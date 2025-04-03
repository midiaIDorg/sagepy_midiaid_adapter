"""
%load_ext autoreload
%autoreload 2
"""
import functools
import json
import os
from pathlib import Path
from warnings import warn
from scipy.interpolate import interp1d

import click
import mokapot
import numpy as np
import numpy.typing as npt
import typing
import pandas as pd
import sagepy.core.scoring
from imspy.algorithm.rescoring import create_feature_space, re_score_psms
from pandas_ops.io import read_df
from pandas_ops.lex_ops import LexicographicIndex
from pandas_ops.stats import sum_real_good
from sagepy.core import Precursor, RawSpectrum, Scorer, SpectrumProcessor, Tolerance
from sagepy.qfdr.tdc import (
    assign_sage_peptide_q,
    assign_sage_protein_q,
    assign_sage_spectrum_q,
    target_decoy_competition_pandas,
)
from sagepy.rescore.utility import transform_psm_to_mokapot_pin, get_features
from sagepy.utility import generate_search_configurations, psm_collection_to_pandas
from sagepy.core.scoring import ScoreType
from tqdm import tqdm
from opentimspy import OpenTIMS
from pandas_ops.io import add_kwargs
from sagepy.core.scoring import Psm as SagepyPsm
import typing


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 4)


generate_search_configurations = add_kwargs(generate_search_configurations)


def sanitize_search_config(search_conf: dict,
    _MAX_PEP_LEN_FOR_INTENSITY_PREDICTION:int=30,
) -> dict:
    assert "enzyme" in search_conf["database"], "Missing database.enzyme settings."

    if "max_len" not in search_conf["database"]["enzyme"]:
        msg = f"Missing database.enzyme.max_len setting. Setting to {_MAX_PEP_LEN_FOR_INTENSITY_PREDICTION}."
        warn(msg)
    else:
        if "rescoring" in search_conf and search_conf["database"]["enzyme"]["max_len"] > _MAX_PEP_LEN_FOR_INTENSITY_PREDICTION:
            msg = f"You are doing rescoring and it must use fragment relative intensity prediction. The default fragment predictor cannot predict fragment intensities of fragments larger than {_MAX_PEP_LEN_FOR_INTENSITY_PREDICTION} amino acids long. Clipping your choice of `{search_conf['database']['enzyme']['max_len']}` to {_MAX_PEP_LEN_FOR_INTENSITY_PREDICTION}."
            warn(msg)
            search_conf["database"]["enzyme"]["max_len"] = _MAX_PEP_LEN_FOR_INTENSITY_PREDICTION

    assert (
        search_conf.get("report_psms", -1) > 0
    ), "You either did not pass `report_psms` or passed in `report_psms<0`. The number of psms should be at least set to 1, and it is suggested to pass in more than 1 for rescoring. But who believes rescoring without theory? Ils sont foux, lex experimentalistes..."

    if "rescoring" in search_conf:
        if search_conf["report_psms"] == 1:
            warn(
                "You have passed in `report_psms = 1` in your config and you want rescoring (even though there is no mathematical theory behind it). Wise people in the field suggest using rescoring with more than 1 psm for rescoring, as if they had any idea how to prove that. Biology will go away. Math will stay."
            )

    search_conf["database"]["static_mods"] = {
        pos: replace_short_unimod_with_long_unimod(mod, verbose)
        for pos, mod in search_conf["database"].get("static_mods", {}).items()
    }
    search_conf["database"]["variable_mods"] = {
        pos: [replace_short_unimod_with_long_unimod(mod, verbose) for mod in mods]
        for pos, mods in search_conf["database"].get("variable_mods", {}).items()
    }

    return search_conf


def sanitized_search_config_to_scorer_kwargs(search_conf: dict) -> dict:
    scorer_kwargs = {
        arg: search_conf[arg]
        for arg in (
            "min_matched_peaks",
            "min_isotope_err",
            "max_isotope_err",
            "chimera",
            "report_psms",
            "wide_window",
            "max_fragment_charge",
        )
        if arg in search_conf
    }
    scorer_kwargs["score_type"] = ScoreType(search_conf.get("score_type", "hyperscore"))

    (
        scorer_kwargs["min_precursor_charge"],
        scorer_kwargs["max_precursor_charge"],
    ) = search_conf["precursor_charge"]
    (
        scorer_kwargs["min_isotope_err"],
        scorer_kwargs["max_isotope_err"],
    ) = search_conf["isotope_errors"]

    get_tol = lambda level: Tolerance(
        **{k: tuple(v) for k, v in search_conf[level].items()}
    )
    for tol in ("precursor_tol", "fragment_tol"):
        if tol in search_conf:
            scorer_kwargs[f"{tol}erance"] = get_tol(tol)

    scorer_kwargs["variable_mods"] = search_conf["database"].get("variable_mods", {})
    scorer_kwargs["static_mods"] = search_conf["database"].get("static_mods", {})
    scorer_kwargs["override_precursor_charge"] = search_conf.get("override_precursor_charge", False)

    return scorer_kwargs


def to_dict(df: pd.DataFrame):
    return {c: df[c].to_numpy() for c in df}


def create_query(
    precursor_stats: pd.DataFrame,
    fragment_stats: pd.DataFrame,
    edges: pd.DataFrame,
    spec_processor: SpectrumProcessor,
    progressbar_desc: str = "Prepping SAGEPY queries.",
    min_peaks: int = 15,
    scan2ce: typing.Callable = lambda x: x,
) -> list:
    prec_stats, frag_stats, matches = map(
        to_dict, (precursor_stats, fragment_stats, edges)
    )
    lx = LexicographicIndex(matches["MS1_ClusterID"])

    # this actually copies into RAM.
    fragment_mzs = frag_stats["mz_wmean"][matches["MS2_ClusterID"]]
    fragment_intensities = frag_stats["intensity"][matches["MS2_ClusterID"]]

    MS1_ClusterIDs = matches["MS1_ClusterID"][lx.idx[:-1]]
    fragment_TICs = lx.map(sum_real_good, fragment_intensities)
    retention_time_wmean_present = "retention_time_wmean" in prec_stats

    prec_mzs = prec_stats["mz_wmean"]
    prec_intensities = prec_stats["intensity"]
    prec_iims = prec_stats["inv_ion_mobility_wmean"]
    prec_scans = prec_stats["scan_wmean"]
    if retention_time_wmean_present:
        prec_rts = prec_stats["retention_time_wmean"]

    queries = []
    for i in tqdm(range(len(lx)), desc=progressbar_desc):
        fragment_peaks_cnt = lx.idx[i + 1] - lx.idx[i]
        if fragment_peaks_cnt >= min_peaks:
            MS1_ClusterID = MS1_ClusterIDs[i]
            precursor = Precursor(
                mz=prec_mzs[MS1_ClusterID],
                charge=None,
                intensity=prec_intensities[MS1_ClusterID],
                inverse_ion_mobility=prec_iims[MS1_ClusterID],
                collision_energy=scan2ce(prec_scans[MS1_ClusterID]),
            )
            raw_spectrum = RawSpectrum(
                file_id=1,
                spec_id=str(MS1_ClusterID),
                total_ion_current=fragment_TICs[i],
                precursors=[precursor],
                mz=fragment_mzs[lx.idx[i] : lx.idx[i + 1]],
                intensity=fragment_intensities[lx.idx[i] : lx.idx[i + 1]],
                scan_start_time=prec_rts[MS1_ClusterID]
                if retention_time_wmean_present
                else None,
                ion_injection_time=prec_rts[MS1_ClusterID]
                if retention_time_wmean_present
                else None,
            )
            queries.append(spec_processor.process(raw_spectrum))

    return queries


def get_core_cnt() -> int:
    cnt = os.cpu_count()
    if cnt is None:
        return 1
    return cnt


cores_cnt = get_core_cnt()


def replace_short_unimod_with_long_unimod(short_unimod, verbose: False):
    long_unimod = short_unimod.replace("U:", "UNIMOD:")
    if verbose and long_unimod != short_unimod:
        print(f"Replaced `{short_unimod}` with `{long_unimod}`.")
    return long_unimod


def iter_merge_split_searches_dropping_target_decoy_collisions(
    psms_in_splits: typing.Iterable[list[SagepyPsm]],
    use_charges: bool = True,
) -> typing.Iterable[SagepyPsm]:
    """Iterate PSMs from different DB splits.

    WARNING: we assume that fasta is split so that different splits do not contain the same protein header.

    Arguments:
        psms_in_splits (Iterable of lists of SagepyPsms): Psms to merge. WARNING!!! TO BE CALLED ON THE SAME SPECTRUM PSMS.
        use_charges (bool): Merge by modified sequence and charge as criterion. When False, only by modified sequence.

    Yields:
        SagepyPsm: A PSM with protein sources adjusted for origins from different splits.
    """
    DELETE_ME = -1
    FIRST_TIME_FOUND = 0

    ion_to_psm_repr = {}
    for psm in chain.from_iterable(psms_in_splits):
        ion = (psm.sequence, psm.sage_feature.charge) if use_charges else psm.sequence

        psm_repr = ion_to_psm_repr.get(ion, FIRST_TIME_FOUND)

        if psm_repr == DELETE_ME:  # already found decoy-target pair
            continue
        elif psm_repr == FIRST_TIME_FOUND:
            ion_to_psm_repr[ion] = psm
        elif psm_repr.decoy == psm.decoy:  # only decoys OK or only targets OK
            psm_repr.proteins.extend(psm.proteins)
            # as different chunks get different proteins.
            # psm_repr.proteins is a list: in place modification
        else:  # one psm is a decoy , another a target: collision to drop
            ion_to_psm_repr[ion] = DELETE_ME

    for ion, psm in ion_to_psm_repr.items():
        if psm != DELETE_ME:
            yield psm


def iter_merge_split_searches_retaining_targets_in_target_decoy_collisions(
    psms_in_splits: typing.Iterable[list[SagepyPsm]],
    use_charges: bool = True,
) -> typing.Iterator[SagepyPsm]:
    """Iterate PSMs from different DB splits.

    WARNING: we assume that fasta is split so that different splits do not contain the same protein header.

    Arguments:
        psms_in_splits (Iterable of lists of SagepyPsms): Psms to merge. WARNING!!! TO BE CALLED ON THE SAME SPECTRUM PSMS.
        use_charges (bool): Merge by modified sequence and charge as criterion. When False, only by modified sequence.

    Yields:
        SagepyPsm: A PSM with protein sources adjusted for origins from different splits.
    """
    FIRST_TIME_FOUND = 0

    ion_to_psm_repr = {}
    for psm in chain.from_iterable(psms_in_splits):
        ion = (psm.sequence, psm.sage_feature.charge) if use_charges else psm.sequence

        psm_repr = ion_to_psm_repr.get(ion, FIRST_TIME_FOUND)

        if psm_repr == FIRST_TIME_FOUND:
            ion_to_psm_repr[ion] = psm
        elif psm_repr.decoy == psm.decoy:  # only decoys OK or only targets OK
            psm_repr.proteins.extend(psm.proteins)
            # as different chunks get different proteins.
            # psm_repr.proteins is a list: in place modification
        elif psm_repr.decoy and not psm.decoy:
            ion_to_psm_repr[ion] = psm

    yield from ion_to_psm_repr.values()


valid_db_splits_merge_strategy: dict[str, typing.Iterator[SagepyPsm]] = dict(
    drop_target_decoy_collisions=iter_merge_split_searches_dropping_target_decoy_collisions,
    retain_targets_if_colliding_with_decoys=iter_merge_split_searches_retaining_targets_in_target_decoy_collisions,
)


def get_fragments_df(
    psm_list,
    fragment_type_as_char: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    fragment_table_size = sum((psm.sage_feature.matched_peaks for psm in psm_list))

    fragment_psm_id = np.zeros(fragment_table_size, dtype=np.int64)
    fragment_type = np.zeros(fragment_table_size, dtype=np.uint8)
    fragment_ordinal = np.zeros(fragment_table_size, dtype=np.int64)
    fragment_charge = np.zeros(fragment_table_size, dtype=np.int64)
    fragment_mz_calculated = np.zeros(fragment_table_size, dtype=np.float64)
    fragment_mz_experimental = np.zeros(fragment_table_size, dtype=np.float64)
    fragment_intensity = np.zeros(fragment_table_size, dtype=np.float64)
    fragment_MS1_ClusterID = np.zeros(fragment_table_size, dtype=np.int64)

    idx = 0
    if verbose:
        progress_bar = tqdm(
            total=fragment_table_size, desc="Writing down fragments table"
        )
    for psm_id, psm in enumerate(psm_list):
        psm_frag_cnt = psm.sage_feature.matched_peaks
        fragments = psm.sage_feature.fragments
        for i in range(psm_frag_cnt):
            fragment_psm_id[idx] = psm_id
            fragment_type[idx] = ord(fragments.ion_types[i].to_string().lower())
            fragment_ordinal[idx] = fragments.fragment_ordinals[i]
            fragment_charge[idx] = fragments.charges[i]
            fragment_mz_calculated[idx] = fragments.mz_calculated[i]
            fragment_mz_experimental[idx] = fragments.mz_experimental[i]
            fragment_intensity[idx] = fragments.intensities[i]
            fragment_MS1_ClusterID[idx] = int(psm.spec_idx)
            progress_bar.update(1)
            idx += 1

    return pd.DataFrame(
        dict(
            psm_id=fragment_psm_id,
            fragment_type=np.char.mod("%c", fragment_type)
            if fragment_type_as_char
            else fragment_type,
            fragment_ordinals=fragment_ordinal,
            fragment_charge=fragment_charge,
            fragment_mz_calculated=fragment_mz_calculated,
            fragment_mz_experimental=fragment_mz_experimental,
            fragment_intensity=fragment_intensity,
            MS1_ClusterID=fragment_MS1_ClusterID,
        ),
        copy=False,
    )


@click.command(context_settings={"show_default": True})
@click.argument("dataset", type=Path)
@click.argument("fasta", type=Path)
@click.argument("search_config", type=Path)
@click.argument("precursor_cluster_stats", type=Path)
@click.argument("fragment_cluster_stats", type=Path)
@click.argument("edges", type=Path)
@click.argument("results_sage_parquet", type=Path)
@click.argument("matched_fragments_sage_parquet", type=Path)
@click.option("--num_threads", type=int, default=cores_cnt)
@click.option("--num_splits", type=int, default=1)
@click.option("--verbose", is_flage=True, help="Drown me with stdout flushes.")
def sagepy_search(
    dataset: Path,
    fasta: Path,
    search_config: Path,
    precursor_cluster_stats: Path,
    fragment_cluster_stats: Path,
    edges: Path,
    results_sage_parquet: Path,
    matched_fragments_sage_parquet: Path,
    num_threads: int = cores_cnt,
    num_splits: int = 1,
    verbose: bool = False,
) -> None:
    """Run sagepy search"""

    assert isinstance(num_splits, int)
    assert num_splits >= 1

    if num_threads > cores_cnt:
        msg = f"You passed in `num_threads={num_threads}` but the max is `{cores_cnt}`. The `{cores_cnt}` will be used."
        warn(msg)

    num_threads = min(num_threads, cores_cnt)


    with open(search_config, "r") as f:
        search_conf = sanitize_search_config(json.load(f))

    scorer_kwargs = sanitized_search_config_to_scorer_kwargs(search_conf)

    if num_splits > 1:
        psm_merge_strategy = search_conf.get("psm_merge_strategy", "drop_target_decoy_collisions")
        assert (
            psm_merge_strategy in valid_db_splits_merge_strategy
        ), f"Currently supported DB splits merge strategies include: {list(valid_db_splits_merge_strategy)}"

    # GET RID OF THIS.
    search_conf["deisotope"] = True
    search_conf["report_psms"] = 1
    # search_conf["predict_rt"] = False

    op = OpenTIMS(dataset)
    DiaFrameMsMsWindows = pd.DataFrame(op.table2dict("DiaFrameMsMsWindows"))
    scan2ce = interp1d(
        DiaFrameMsMsWindows["ScanNumBegin"],
        DiaFrameMsMsWindows["CollisionEnergy"],
        kind="linear",
        fill_value="extrapolate",
    )
    
    precursor_stats = read_df(
        precursor_cluster_stats,
        columns=[
            "mz_wmean",
            "intensity",
            "inv_ion_mobility_wmean",
            "retention_time_wmean",
            "scan_wmean",
            "ClusterID",
        ],
    )
    fragment_stats = read_df(
        fragment_cluster_stats,
        columns=["ClusterID", "mz_wmean", "intensity"],
    )
    fragment_stats["intensity"] = fragment_stats["intensity"].astype(np.float32)
    fragment_stats["mz_wmean"] = fragment_stats["mz_wmean"].astype(np.float32)
    edges = read_df(edges)

    spec_processor = SpectrumProcessor(
        take_top_n=search_conf.get("max_peaks", 1000),
        min_deisotope_mz=search_conf.get("min_deisotope_mz", 0.0),
        deisotope=search_conf.get("deisotope", True),
    )

    queries = create_query(
        precursor_stats,
        fragment_stats,
        edges,
        spec_processor,
        scan2ce=scan2ce,
        **{arg: search_conf[arg] for arg in ("min_peaks",) if arg in search_conf},
    )
    scorer = Scorer(**scorer_kwargs)

    dbs = generate_search_configurations(
        fasta_path=fasta,
        num_splits=num_splits,
        **search_conf["database"],
        **search_conf["database"]["enzyme"],
    )

    if verbose:
        dbs = tqdm(dbs, total=num_splits, desc="Searching database.")
    
    psms = [
        scorer.score_collection_psm(db, queries, num_threads=num_threads) for db in dbs
    ]  # list of spectrum id to list of psms
    for psm_l in psms:
        assert len(psm_l) == len(psms[0])

    if num_splits > 1:
        valid_db_splits_merge_strategy[psm_merge_strategy]


        merged_psms = functools.reduce(
            functools.partial(
                sagepy.core.scoring.merge_psm_dicts,
                max_hits=search_conf["report_psms"],
            ),
            psms,
        )
        psm_list = [psm for psms in merged_psms.values() for psm in psms]

    else
    # from collections import Counter
    # Counter(map(len, merged_psms.values()))
    # Counter(map(len, merged_psms2.values()))

    # for psms_dct in psms
    # Counter(map(sum, zip(*map(lambda psms_dct: map(len, psms_dct.values()), psms)))) == Counter(map(len, merged_psms2.values()))

    # %%timeit
    merge_lists = lambda lists: list(chain.from_iterable(lists))

    from collections import defaultdict
    from itertools import chain

    lists = (psms[0]["100205"], psms[1]["100205"], psms[2]["100205"])

    # %%time
    psm_list2 = list(
        chain.from_iterable(
            map(
                merge_lists_2,
                zip(*(psm_dct.values() for psm_dct in psms)),
            )
        )
    )

    res2 = {(psm.spec_idx, psm.sequence, psm.sage_feature.charge) for psm in psm_list2}
    res1 = {(psm.spec_idx, psm.sequence, psm.sage_feature.charge) for psm in psm_list}

    merged_psms2 = dict(
        zip(
            psms[0].keys(),
            map(
                merge_lists,
                zip(*(psm_dct.values() for psm_dct in psms)),
            ),
        )
    )

    # OK, we can directly drop to a list.

    # for A, B in zip(zip(psms[0].values(), psms[1].values(), psms[2].values()), zip(*(psm_dct.values() for psm_dct in psms))):
    #     assert A==B

    # len(psms[0]["100185"])
    # len(psms[1]["100185"])
    # len(psms[2]["100185"])

    # sorted([psm.peptide_idx for psm in merged_psms["100185"]])
    # sorted([psm.peptide_idx for psm in merged_psms2["100185"]])

    # [len(psms_dct["100184"]) for psms_dct in psms]
    # len(merged_psms["100184"])
    # len(merged_psms2["100184"])

    # assert len(merged_psms) == len(psms[0])

    # TODO: extract and save the SAGE matched peaks???
    for psm in psm_list:
        psm.retention_time /= 60.0

    if "rescoring" in search_conf:
        psm_list = create_feature_space(
            psms=psm_list,
            **search_conf["rescoring"]["feature_space_settings"],
        )

        if "david_teschners_random_combo" in search_conf["rescoring"]["engines"]:
            psm_list = re_score_psms(
                psm_list,
                **search_conf["rescoring"]["engines"]["david_teschners_random_combo"],
            )

        precursors = psm_collection_to_pandas(psm_list, num_threads=num_threads)

        # if "mokapot" in search_conf["rescoring"]["engines"]:
        #     psms_moka = mokapot.read_pin(transform_psm_to_mokapot_pin(precursors))
        #     results, _ = mokapot.brew(psms_moka, **)

        #     if "seed" in search_conf["rescoring"]["engines"]["mokapot"]:
        #         mokapot_kwargs["rng"] = search_conf["rescoring_engines"]["mokapot"][
        #             "seed"
        #         ]

        #     precursors["spec_idx"] = precursors["spec_idx"].astype(int)
        #     precursors = precursors.merge(
        #         results.psms,
        #         how="left",
        #         left_on="spec_idx",
        #         right_on="SpecId",
        #         suffixes=("", "_mokapot"),
        #     )

        # why is this done I don't know how many times????
        psm_list = [
            psm
            for psm in psm_list
            if psm.rank
            <= search_conf[
                "report_psms"
            ]  # top rank == 1, ordered descending with score
        ]

    else:
        # why is this done I don't know how many times????
        psm_list = [
            psm
            for psm in psm_list
            if psm.rank
            <= search_conf[
                "report_psms"
            ]  # top rank == 1, ordered descending with score
        ]
        # assign SAGE q-values
        assign_sage_spectrum_q(psm_list)
        assign_sage_peptide_q(psm_list)
        assign_sage_protein_q(psm_list)
        precursors = psm_collection_to_pandas(psm_list, num_threads=num_threads)

    from imspy.timstof.dbsearch.utility import parse_to_tims2rescore

    # you can also use double competition to get the q-values CREMA style
    TDC_pandas = target_decoy_competition_pandas(
        precursors, method="peptide_psm_peptide", score="hyperscore"
    )

    # parse_to_tims2rescore(precursors)

    from pandas_ops.io import save_df

    save_df(precursors, "F9477_1psm.parquet")

    precursors = precursors.reset_index()
    precursors = precursors.rename(
        columns=dict(
            index="psm_id",
            spec_idx="MS1_ClusterID",
            sequence_modified="peptide",
            decoy="is_decoy",
        )
    )
    precursors["num_proteins"] = precursors.proteins.map(len)
    precursors.proteins = precursors.proteins.str.join(";")
    precursors["filename"] = ""
    precursors["scannr"] = precursors.MS1_ClusterID
    precursors["label"] = precursors.is_decoy.map({False: -1, True: 1})
    precursors["peptide_len"] = precursors.sequence.map(len)

    precursors2 = precursors.set_index("MS1_ClusterID")
    precursors2.loc[list(map(str, found_in_sagepy_not_sage))]
    precursors2.loc["100588"]

    fragments_df = get_fragments_df(
        psm_list,
        fragment_type_as_char=True,  # necessary for downstream procedures.
        verbose=verbose,
    )

    # precursor_cluster_stats
    sageprec.MS1_ClusterID
    venny_cnt(sageprec.MS1_ClusterID, precursors.MS1_ClusterID.astype(int))

    submitted = [int(q.id) for q in queries]
    venny_cnt(sageprec.MS1_ClusterID, submitted)

    print(TDC_pandas)
    # fragment_cluster_stats
    # results_sage_parquet


# this is done in get_features`
# X, Y = get_features(
#     PSM_pandas,
#     score=search_conf.get("score_type", "hyperscore"),
#     replace_nan=True,
# )

# features = [
#     f"{score}",
#     "delta_rt",
#     "delta_ims",
#     "cosine_similarity",
#     "delta_mass",
#     "rank",
#     "isotope_error",
#     "average_ppm",
#     "delta_next",
#     "delta_best",
#     "matched_peaks",
#     "longest_b",
#     "longest_y",
#     "longest_y_pct",
#     "missed_cleavages",
#     "matched_intensity_pct",
#     "poisson",
#     "charge",
#     "intensity_ms1",
#     "intensity_ms2",
#     "collision_energy",
#     "cosine_similarity",
#     "spectral_angle_similarity",
#     "pearson_correlation",
#     "spearman_correlation",
#     "spectral_entropy_similarity",
# ]

# fitting the classifier

# Perform FDR filtering

# HERE GOES MAPPING BACK? Does it really????
# psm = psm_list[0]
# OK, preallocate results of size sum(psm.sage_feature.matched_peaks)
# likely just need a table like the one from hte new sage.
