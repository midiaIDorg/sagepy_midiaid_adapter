"""
%load_ext autoreload
%autoreload 2
"""
import duckdb
import numba
import math
import functools
import json
import os
import heapq
import toml
from pathlib import Path
from warnings import warn
from scipy.interpolate import interp1d
from collections import defaultdict
from collections import Counter
import itertools

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
from sagepy.rescore.rescore import rescore_psms
from sagepy.rescore.utility import transform_psm_to_mokapot_pin, get_features
from sagepy.utility import generate_search_configurations, psm_collection_to_pandas
from sagepy.core.scoring import ScoreType
from tqdm import tqdm
from opentimspy import OpenTIMS
from pandas_ops.io import add_kwargs
from sagepy.core.scoring import Psm as SagepyPsm
from pandas_ops.io import save_df
import typing
from typing import Callable
from typing import Iterable
from typing import Iterator

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 4)


generate_search_configurations = add_kwargs(generate_search_configurations)


def sanitize_search_config(
    search_conf: dict,
    _MAX_PEP_LEN_FOR_INTENSITY_PREDICTION: int = 30,
) -> dict:
    assert "enzyme" in search_conf["database"], "Missing database.enzyme settings."

    if "max_len" not in search_conf["database"]["enzyme"]:
        msg = f"Missing database.enzyme.max_len setting. Setting to {_MAX_PEP_LEN_FOR_INTENSITY_PREDICTION}."
        warn(msg)
    else:
        if (
            "feature_prediction" in search_conf
            and search_conf["database"]["enzyme"]["max_len"]
            > _MAX_PEP_LEN_FOR_INTENSITY_PREDICTION
        ):
            msg = f"You decided for prediction and it encompassed (for now) fragment relative intensity prediction. The default fragment predictor cannot predict fragment intensities of fragments larger than {_MAX_PEP_LEN_FOR_INTENSITY_PREDICTION} amino acids long. Clipping your choice of `{search_conf['database']['enzyme']['max_len']}` to {_MAX_PEP_LEN_FOR_INTENSITY_PREDICTION}."
            warn(msg)
            search_conf["database"]["enzyme"][
                "max_len"
            ] = _MAX_PEP_LEN_FOR_INTENSITY_PREDICTION

    assert (
        search_conf.get("report_psms", -1) > 0
    ), "You either did not pass `report_psms` or passed in `report_psms<0`. The number of psms should be at least set to 1, and it is suggested to pass in more than 1 for rescoring. But who believes rescoring without theory? Ils sont foux, lex experimentalistes..."

    if "feature_prediction" in search_conf:
        if search_conf["report_psms"] == 1:
            warn(
                "You have passed in `report_psms = 1` in your config and you want to predict feature that can typically be used in rescoring even though there IS NO MATHEMATICAL THEORY BEHIND ITS INFLUENCE ON FDR CALCULATIONS (SHAME!!! SHAME!!! SHAME!!!). Wise people in the field suggest using rescoring with more than 1 psm for rescoring, as if they had any idea how to prove that. Biology will go away. Math will stay."
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
    scorer_kwargs["override_precursor_charge"] = search_conf.get(
        "override_precursor_charge", False
    )

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
    scan2ce: Callable = lambda x: x,
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


def iter_merge_psms_per_spectrum(
    psms_in_splits_per_spectrum: Iterable[list[SagepyPsm]],
    use_charges: bool = True,
) -> Iterable[tuple[SagepyPsm, bool]]:
    """Iterate merged PSMs per spectrum from different DB splits while detecting collisions.

    This procedure can report 2 peptides with the same sequence (or sequence and charge) that could originate from both a target or decoy sequence.

    WARNING: we assume that fasta is split so that different splits do not contain the same protein header.

    Arguments:
        psms_in_splits_per_spectrum (Iterable of lists of SagepyPsms): Psms to merge. WARNING!!! TO BE CALLED ON THE SAME SPECTRUM PSMS.
        use_charges (bool): Merge by modified sequence and charge as criterion. When False, only by modified sequence.
        *args, **kwargs: No influence.

    Yields:
        tuple[SagepyPsm, bool]: A PSM with protein sources adjusted for origins from different splits and info of whether it colides with some other peptide in a target-decoy collision.
    """
    final_psms = {}
    final_proteins = defaultdict(set)
    key_to_labels = defaultdict(set)

    for psm in itertools.chain.from_iterable(psms_in_splits_per_spectrum):
        key = (psm.sequence, psm.sage_feature.charge) if use_charges else psm.sequence
        labels = key_to_labels[key]
        if len(labels) < 2:
            final_psms[(key, psm.decoy)] = psm
            labels.add(psm.decoy)
        final_proteins[key].update(psm.proteins)

    for (key, label), psm in final_psms.items():
        labels = key_to_labels[key]
        assert len(labels) > 0
        assert len(labels) <= 2
        detected_collision = len(labels) == 2
        psm.proteins = list(final_proteins[key])
        yield psm, detected_collision


def test_iter_merge_psms_per_spectrum():
    from types import SimpleNamespace as Mock

    psms_in_splits_per_spectrum = (
        [
            Mock(
                decoy=True,
                proteins=["a", "b"],
                sequence="ABC",
                sage_feature=Mock(charge=1),
            ),
            Mock(
                decoy=False,
                proteins=["a", "b"],
                sequence="DE",
                sage_feature=Mock(charge=1),
            ),
            Mock(
                decoy=True,
                proteins=["a", "b"],
                sequence="FG",
                sage_feature=Mock(charge=1),
            ),
        ],
        [
            Mock(
                decoy=True,
                proteins=["c", "b"],
                sequence="ABC",
                sage_feature=Mock(charge=1),
            ),
            Mock(
                decoy=False,
                proteins=["d", "b"],
                sequence="DE",
                sage_feature=Mock(charge=1),
            ),
            Mock(
                decoy=True,
                proteins=["e", "b"],
                sequence="FG",
                sage_feature=Mock(charge=1),
            ),
        ],
        [
            Mock(
                decoy=True,
                proteins=["f", "b"],
                sequence="ABC",
                sage_feature=Mock(charge=1),
            ),
            Mock(
                decoy=False,
                proteins=["g", "b"],
                sequence="DE",
                sage_feature=Mock(charge=1),
            ),
            Mock(
                decoy=False,
                proteins=["h", "b"],
                sequence="FG",
                sage_feature=Mock(charge=1),
            ),
        ],
    )
    expected_outcomes = (
        (
            Mock(
                decoy=True,
                proteins=["a", "b", "c", "f"],
                sequence="ABC",
                sage_feature=Mock(charge=1),
            ),
            False,
        ),
        (
            Mock(
                decoy=False,
                proteins=["a", "b", "d", "g"],
                sequence="DE",
                sage_feature=Mock(charge=1),
            ),
            False,
        ),
        (
            Mock(
                decoy=True,
                proteins=["a", "b", "e", "h"],
                sequence="FG",
                sage_feature=Mock(charge=1),
            ),
            True,
        ),
        (
            Mock(
                decoy=False,
                proteins=["a", "b", "e", "h"],
                sequence="FG",
                sage_feature=Mock(charge=1),
            ),
            True,
        ),
    )
    for (exp_psm, exp_collision), (obs_psm, obs_collision) in zip(
        expected_outcomes,
        iter_merge_psms_per_spectrum(
            psms_in_splits_per_spectrum,
            use_charges=True,
        ),
    ):
        obs_psm.proteins = sorted(obs_psm.proteins)
        exp_psm.proteins = sorted(exp_psm.proteins)
        assert exp_psm == obs_psm
        assert exp_collision == obs_collision


target_decoy_collision_to_filter: dict[str, Callable[[SagepyPsm, bool], bool]] = dict(
    KEEP_BOTH=lambda psm, collision: True,
    KEEP_TARGET_DELETE_DECOY=lambda psm, collision: not psm.decoy,
    DROP_BOTH=lambda psm, collision: not detected_collision,
)


def get_psm_collision_tuples(top_n_per_spectrum: int | float = math.inf):
    if top_n_per_spectrum == math.inf:
        return iter_merge_psms_per_spectrum

    return lambda psms_per_spectrum, use_charges: heapq.nlargest(
        int(top_n_per_spectrum),
        iter_merge_psms_per_spectrum(psms_per_spectrum, use_charges),
        key=lambda x: x[0].hyperscore,
    )


def matteos_happy_merge(
    psms_dcts: list[dict[str, list[SagepyPsm]]],
    use_charges: bool = True,
    collision_decision: str = "KEEP_BOTH",  # "keep target", "drop both"
    top_n_per_spectrum: int | float = math.inf,
) -> tuple[list[SagepyPsm], list[bool]]:
    """Get merged PSMs per spectrum from different DB splits while detecting collisions.

    This procedure can report 2 peptides with the same sequence (or sequence and charge) that could originate from both a target or decoy sequence.

    WARNING: we assume that fasta is split so that different splits do not contain the same protein header.

    Arguments:
        psms_in_splits_per_spectrum (Iterable of lists of SagepyPsms): Psms to merge. WARNING!!! TO BE CALLED ON THE SAME SPECTRUM PSMS.
        use_charges (bool): Merge by modified sequence and charge as criterion. When False, only by modified sequence.
        collision_decision (str): What strategy to use for peptides that have a mixed decoy-target prodigy?

    Yields:
        tuple[SagepyPsm, bool]: A PSM with protein sources adjusted for origins from different splits and info of whether it colides with some other peptide in a target-decoy collision.
    """
    assert (
        collision_decision in target_decoy_collision_to_filter
    ), f"Provide collision_decision from {list(target_decoy_collision_to_filter)}."

    psm_filter = target_decoy_collision_to_filter[collision_decision]
    psm_collision_tuples = get_psm_collision_tuples(top_n_per_spectrum)

    psms = []
    target_decoy_collisions = []
    for psms_per_spectrum in zip(*(psm_dct.values() for psm_dct in psms_dcts)):
        for psm, collision in psm_collision_tuples(psms_per_spectrum, use_charges):
            if psm_filter(psm, collision):
                psms.append(psm)
                target_decoy_collisions.append(collision)

    return psms, target_decoy_collisions


def davids_happy_merge(
    psms_in_splits_per_spectrum: Iterable[list[SagepyPsm]],
    top_n_per_spectrum: int = 1000,
    *args,
    **kwargs,
) -> tuple[list[SagepyPsm], list]:
    merged_psms = functools.reduce(
        functools.partial(
            sagepy.core.scoring.merge_psm_dicts,
            max_hits=top_n_per_spectrum,
        ),
        psms_in_splits_per_spectrum,
    )
    psms = [psm for psms in merged_psms.values() for psm in psms]
    target_decoy_collisions = []
    return psms, target_decoy_collisions


DB_splits_mergers: dict[
    str, Callable[Iterable[list[SagepyPsm]], tuple[list[SagepyPsm], list]]
] = dict(matteos_happy_merge=matteos_happy_merge, davids_happy_merge=davids_happy_merge)


def adjust_precursors_to_match_SAGE_naming_and_types(
    precursors: pd.DataFrame,
    precursor_stats: pd.DataFrame,
) -> None:
    """
    Adjust names and values to have
    """
    precursors = precursors.reset_index()
    precursors = precursors.rename(
        columns=dict(
            index="psm_id",
            spec_idx="MS1_ClusterID",
            sequence_modified="peptide",
            decoy="is_decoy",
            average_ppm="precursor_ppm",
            delta_ims="delta_mobility",
        )
    )
    precursors["num_proteins"] = precursors.proteins.map(len)
    precursors.proteins = precursors.proteins.str.join(";")
    precursors["MS1_ClusterID"] = precursors.MS1_ClusterID.astype(np.uint32)
    precursors = precursors.sort_values("MS1_ClusterID")

    precursors["label"] = precursors.is_decoy.map({False: -1, True: 1})
    precursors["peptide_len"] = precursors.sequence.map(len)
    # ?precursors["semi_enzymatic"]?
    # if "feature_prediction" in search_conf:
    #     precursors.predicted_rt
    #     precursors.rt
    #     plt.hist( precursors.rt / precursors.predicted_rt, bins=100)
    #     plt.show()

    precursor_stats.retention_time_wmean.iloc[
        precursors.MS1_ClusterID
    ].to_numpy() / 60.0

    plt.scatter(
        precursor_stats.retention_time_wmean.iloc[precursors.MS1_ClusterID].to_numpy()
        / 60.0,
        precursors.predicted_rt - precursors.delta_rt_model,
    )
    plt.show()

    precursors.matched_peaks = precursors.matched_peaks.astype(np.int64)
    precursors.longest_b = precursors.longest_b.astype(np.int64)
    precursors.longest_y = precursors.longest_y.astype(np.int64)
    precursors.scored_candidates = precursors.scored_candidates.astype(np.int64)

    precursors["ion_mobility"] = precursor_stats.inv_ion_mobility_wmean.iloc[
        precursors.MS1_ClusterID
    ].to_numpy()
    # delta_mobility = real - predicted
    precursors["predicted_mobility"] = (
        precursors["ion_mobility"] - precursors.delta_mobility
    )


def get_fragments_alla_SAGE(
    psms: list[SagepyPsm],
    fragment_type_as_char: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    fragment_table_size = sum((psm.sage_feature.matched_peaks for psm in psms))

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
    for psm_id, psm in enumerate(psms):
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


@numba.njit
def get_average_diff(xx, yy):
    assert len(xx) == len(yy)
    N = len(xx)
    res = 0.0
    for x, y in zip(xx, yy):
        res += (x - y) / N
    return res


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
        search_conf = sanitize_search_config(
            json.load(f) if search_config.suffix == ".json" else toml.load(f)
        )
    scorer_kwargs = sanitized_search_config_to_scorer_kwargs(search_conf)

    try:
        DB_splits_merger = DB_splits_mergers[
            search_conf.get("psm_merge_strategy", "matteos_happy_merge")
        ]
        DB_splits_merger_kwargs = search_conf.get("psm_merge_strategy_kwargs", {})
    except KeyError as exc:
        raise KeyError(
            f"Currently supported DB splits merge strategies include: {list(DB_splits_mergers)}. You passed in `{exc}`."
        ) from exc

    if not search_conf["deisotope"]:
        warn("Are you sure you do not want SAGEPY to run deisotoping?")

    dbconn = duckdb.connect()
    DiaFrameMsMsWindows = pd.DataFrame(
        OpenTIMS(dataset).table2dict("DiaFrameMsMsWindows")
    )
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
    stats = {}

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

    psms_dcts: list[  # different fasta split results stored here as dicts
        dict[
            str,  # spec_idx: spectrum identification.
            list[SagepyPsm],  # psms per spectrum
        ]
    ] = [
        scorer.score_collection_psm(db, queries, num_threads=num_threads) for db in dbs
    ]

    for psms_dct in psms_dcts:
        assert len(psms_dct) == len(psms_dcts[0])
        assert max(Counter(map(len, psms_dct.values()))) <= scorer_kwargs["report_psms"]

    psms, target_decoy_collisions = (
        DB_splits_merger(psms_dcts, **DB_splits_merger_kwargs)
        if num_splits > 1
        else list(psms_dcts.values())
    )

    target_decoy_collisions_stats = Counter(target_decoy_collisions)
    stats["TARGET_DECOY_COLLISION_CNT"] = target_decoy_collisions_stats[True]
    stats["PSMS_WITHOUT_COLLISION_CNT"] = target_decoy_collisions_stats[False]

    if "feature_prediction" in search_conf:
        # this needs to be extended / replaced by some function that gets midia specific
        # features. Or we do it in another place, which is a good idea.
        psms = create_feature_space(
            psms=psms,
            verbose=verbose,
            **search_conf["feature_prediction"],
        )

    # SAGE operations directly exposed via SAGEPY
    assign_sage_spectrum_q(psms)
    assign_sage_peptide_q(psms)
    assign_sage_protein_q(psms)

    precursors: pd.DataFrame = psm_collection_to_pandas(psms, num_threads=num_threads)
    # TODO: add this to `psm_collection_to_pandas`
    precursors["fragment_ppm"] = [
        get_average_diff(
            psm.sage_feature.fragments.mz_calculated,
            psm.sage_feature.fragments.mz_experimental,
        )
        for psm in (
            tqdm(psms, desc="Getting average framgent m/z error in ppm")
            if verbose
            else psms
        )
    ]

    save_df(precursors, results_sage_parquet)

    fragments: pd.DataFrame = get_fragments_alla_SAGE(
        psms,
        fragment_type_as_char=True,  # downstream necessity
        verbose=verbose,
    )

    save_df(fragments, matched_fragments_sage_parquet)

    # WHAT SAGEPY GIVES US: predictions of intensities actually only
    # others than that: no need to do MGF.


# psms = [
#     psm
#     for psm in psms
#     if psm.rank
#     <= search_conf[
#         "report_psms"
#     ]  # top rank == 1, ordered descending with score
# ]
# assign SAGE q-values

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
# psm = psms[0]
# OK, preallocate results of size sum(psm.sage_feature.matched_peaks)
# likely just need a table like the one from hte new sage.


# if "david_teschners_random_combo" in search_conf["rescoring"]["engines"]:
#     psms = re_score_psms(
#         psms,
#         **search_conf["rescoring"]["engines"]["david_teschners_random_combo"],
#     )

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
# precursors = psm_collection_to_pandas(psms, num_threads=num_threads)

# from imspy.timstof.dbsearch.utility import parse_to_tims2rescore

# you can also use double competition to get the q-values CREMA style
# TDC_pandas = target_decoy_competition_pandas(
#     precursors, method="peptide_psm_peptide", score="hyperscore"
# )

# parse_to_tims2rescore(precursors)
