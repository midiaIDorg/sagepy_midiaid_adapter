# %load_ext autoreload
# %autoreload 2

import functools
import json
import os
from pathlib import Path
from warnings import warn

import click
import mokapot
import numpy as np
import numpy.typing as npt
import pandas as pd
import sagepy.core.scoring
from imspy.algorithm.rescoring import create_feature_space, re_score_psms
from pandas_ops.io import read_df
from pandas_ops.lex_ops import LexicographicIndex
from pandas_ops.stats import sum_real_good
from sagepy.core import (Precursor, RawSpectrum, Scorer, SpectrumProcessor,
                         Tolerance)
from sagepy.qfdr.tdc import (assign_sage_peptide_q, assign_sage_protein_q,
                             assign_sage_spectrum_q,
                             target_decoy_competition_pandas)
from sagepy.rescore.utility import transform_psm_to_mokapot_pin
from sagepy.utility import (generate_search_configurations,
                            psm_collection_to_pandas)
from tqdm import tqdm


def to_dict(df: pd.DataFrame):
    return {c: df[c].to_numpy() for c in df}


def create_query(
    precursor_stats: pd.DataFrame,
    fragment_stats: pd.DataFrame,
    edges: pd.DataFrame,
    spec_processor: SpectrumProcessor,
    progressbar_desc: str = "Prepping SAGEPY queries.",
    min_peaks: int = 15,
) -> list:
    precursor_stats, fragment_stats, edges = map(
        to_dict, (precursor_stats, fragment_stats, edges)
    )
    lx = LexicographicIndex(edges["MS1_ClusterID"])

    # this actually copies into RAM.
    fragment_mzs = fragment_stats["mz_wmean"][edges["MS2_ClusterID"]]
    fragment_intensities = fragment_stats["intensity"][edges["MS2_ClusterID"]]

    MS1_ClusterIDs = edges["MS1_ClusterID"][lx.idx[:-1]]
    fragment_TICs = lx.map(sum_real_good, fragment_intensities)
    retention_time_wmean_present = "retention_time_wmean" in precursor_stats

    queries = []
    for i in tqdm(range(len(lx)), desc=progressbar_desc):
        fragment_peaks_cnt = lx.idx[i + 1] - lx.idx[i]
        if fragment_peaks_cnt >= min_peaks:
            MS1_ClusterID = MS1_ClusterIDs[i]
            precursor = Precursor(
                mz=precursor_stats["mz_wmean"][MS1_ClusterID],
                charge=None,
                intensity=precursor_stats["intensity"][MS1_ClusterID],
                inverse_ion_mobility=precursor_stats["inv_ion_mobility_wmean"][
                    MS1_ClusterID
                ],
            )
            raw_spectrum = RawSpectrum(
                file_id=1,
                spec_id=str(MS1_ClusterID),
                total_ion_current=fragment_TICs[i],
                precursors=[precursor],
                mz=fragment_mzs[lx.idx[i] : lx.idx[i + 1]],
                intensity=fragment_intensities[lx.idx[i] : lx.idx[i + 1]],
                scan_start_time=precursor_stats["retention_time_wmean"][MS1_ClusterID]
                if retention_time_wmean_present
                else None,
                ion_injection_time=precursor_stats["retention_time_wmean"][
                    MS1_ClusterID
                ]
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


# fasta = "fastas/Human_2024_02_16_UniProt_Taxon9606_Reviewed_20434entries_contaminant_tenzer.fasta"
# precursor_cluster_stats = "tmp/clusters/tims/reformated/63/combined_cluster_stats.parquet"
# fragment_cluster_stats = "tmp/clusters/tims/reformated/65/combined_cluster_stats.parquet"
# edges = "tmp/edges/rough/69/rough_edges.startrek"
# search_config = "tmp/configs/sage_config/109.json"
# num_threads = 16
@click.command(context_settings={"show_default": True})
@click.argument("fasta", type=Path)
@click.argument("search_config", type=Path)
@click.argument("precursor_cluster_stats", type=Path)
@click.argument("fragment_cluster_stats", type=Path)
@click.argument("edges", type=Path)
@click.argument("results_sage_parquet", type=Path)
@click.argument("matched_fragments_sage_parquet", type=Path)
@click.option("--num_threads", type=int, default=cores_cnt)
def sagepy_search(
    fasta: Path,
    search_config: Path,
    precursor_cluster_stats: Path,
    fragment_cluster_stats: Path,
    edges: Path,
    results_sage_parquet: Path,
    matched_fragments_sage_parquet: Path,
    num_threads: int = cores_cnt,
) -> None:
    """Run sagepy search"""

    if num_threads > cores_cnt:
        msg = f"You passed in `num_threads={num_threads}` but the max is `{cores_cnt}`. The `{cores_cnt}` will be used."
        warn(msg)

    num_threads = min(num_threads, cores_cnt)

    precursor_stats = read_df(
        precursor_cluster_stats,
        columns=[
            "mz_wmean",
            "intensity",
            "inv_ion_mobility_wmean",
            "retention_time_wmean",
        ],
    )
    fragment_stats = read_df(
        fragment_cluster_stats,
        columns=["ClusterID", "mz_wmean", "intensity"],
    )
    fragment_stats["intensity"] = fragment_stats["intensity"].astype(np.float32)
    fragment_stats["mz_wmean"] = fragment_stats["mz_wmean"].astype(np.float32)
    edges = read_df(edges)

    with open(search_config, "r") as f:
        search_conf = json.load(f)

    # TODO: move to config
    search_conf["rescoring"] = dict(
        feature_space_settings=dict(
            fine_tune_im=True,
            fine_tune_rt=True,
            verbose=True,
        ),
        engines=dict(
            mokapot=dict(level="modified_peptide"),
            david_teschners_random_combo=dict(
                use_logreg=True,
            ),
        ),
    )

    spec_processor = SpectrumProcessor(
        take_top_n=search_conf.get("max_peaks", 150),
        min_deisotope_mz=0.0,
        deisotope=search_conf["deisotope"],
    )

    queries = create_query(
        precursor_stats,
        fragment_stats,
        edges,
        spec_processor,
        **{arg: search_conf[arg] for arg in ("min_peaks",) if arg in search_conf},
    )

    scorer_kwargs = {
        arg: search_conf[arg]
        for arg in (
            "min_matched_peaks",
            "min_isotope_err",
            "max_isotope_err",
            "chimera",
            "report_psms",
            "wide_window",
            "score_type",
            "max_fragment_charge",
        )
        if arg in search_conf
    }

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

    scorer = Scorer(
        variable_mods=search_conf["database"]["variable_mods"],
        static_mods=search_conf["database"]["static_mods"],
        annotate_matches=True,
        override_precursor_charge=False,
        **scorer_kwargs,
    )

    max_peptide_len = search_conf["database"]["enzyme"]["max_len"]
    if "rescoring" in search_conf and max_peptide_len > 30:
        msg = f"You are doing rescoring and it must use fragment relative intensity prediction. The default fragment predictor cannot predict fragment intensities of fragments larger than 30 amino acids long. Clipping your choice of `{max_peptide_len}` to 30."
        warn(msg)
        max_peptide_len = 30

    # iterate over search configurations
    num_splits = 2

    dbs = generate_search_configurations(
        fasta_path=fasta,
        num_splits=num_splits,
        min_len=search_conf["database"]["enzyme"]["min_len"],
        max_len=max_peptide_len,
        cleave_at=search_conf["database"]["enzyme"]["cleave_at"],
        restrict=search_conf["database"]["enzyme"]["restrict"],
        c_terminal=search_conf["database"]["enzyme"]["c_terminal"],
        generate_decoys=search_conf["database"]["generate_decoys"],
        bucket_size=search_conf["database"]["bucket_size"],
        static_mods=search_conf["database"]["static_mods"],
        variable_mods=search_conf["database"]["variable_mods"],
        missed_cleavages=search_conf["database"]["enzyme"]["missed_cleavages"],
    )

    dbs = tqdm(dbs, total=num_splits, desc="Searching database.")
    psms = [
        scorer.score_collection_psm(db, queries, num_threads=num_threads) for db in dbs
    ]

    # max_hits = search_conf["max_retraining_psms"]
    merged_psms = functools.reduce(
        functools.partial(sagepy.core.scoring.merge_psm_dicts, max_hits=25), psms
    )
    psm_list = [psm for psms in merged_psms.values() for psm in psms]

    # TODO: extract and save the SAGE matched peaks
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

        PSM_pandas = psm_collection_to_pandas(psm_list, num_threads=num_threads)

        if "mokapot" in search_conf["rescoring"]["engines"]:
            psms_moka = mokapot.read_pin(transform_psm_to_mokapot_pin(PSM_pandas))
            results, _ = mokapot.brew(psms_moka, **)



            columns_of_interest = [
                "average_ppm",
                "calcmass",
                "charge",
                "collision_energy",
                "cosine_similarity",
                "decoy",
                "delta_best",
                "delta_ims",
                "delta_mass",
                "delta_next",
                "delta_rt",
                "expmass",
                "hyperscore",
                "intensity_ms1",
                "intensity_ms2",
                "isotope_error",
                "longest_b",
                "longest_y",
                "longest_y_pct",
                "matched_intensity_pct",
                "matched_peaks",
                "missed_cleavages",
                "pearson_correlation",
                "poisson",
                "proteins",
                "rank",
                "rt",
                "spec_idx",
                "spearman_correlation",
                "spectral_angle_similarity",
                "spectral_entropy_similarity",
            ]

            # TODO: put it in config:
            match search_conf["rescoring"]["engines"]["mokapot"]["level"]:
                case "peptide":
                    level = ("peptide",)
                case "modified_peptide":
                    level = ("sequence",)
                case "ion":
                    level = (
                        "peptide",
                        "charge",
                    )
                case "modified_ion":
                    level = (
                        "sequence",
                        "charge",
                    )
            columns_of_interest.extend(level)
            PSM_pandas = PSM_pandas[columns_of_interest]

            mokapot_kwargs = {}
            if "seed" in search_conf["rescoring"]["engines"]["mokapot"]:
                mokapot_kwargs["rng"] = search_conf["rescoring_engines"]["mokapot"][
                    "seed"
                ]

            # PSM_pandas["decoy"] = [ 1 if d else -1 for d in PSM_pandas["decoy"]]
            psm_moka = mokapot.LinearPsmDataset(
                psms=PSM_pandas,
                target_column="decoy",
                spectrum_columns="spec_idx",
                peptide_column="sequence",
                protein_column="proteins",
                # group_column=???,
                feature_columns=None,
                calcmass_column="calcmass",
                expmass_column="expmass",
                rt_column="rt",
                charge_column="charge",
                copy_data=True,
                **mokapot_kwargs,
            )
            results, _ = mokapot.brew(psm_moka)
            psm_moka.assign_confidence()

            PSM_pandas["spec_idx"] = PSM_pandas["spec_idx"].astype(int)
            PSM_pandas = PSM_pandas.merge(
                results.psms,
                how="left",
                left_on="spec_idx",
                right_on="SpecId",
                suffixes=("", "_mokapot"),
            )
        psm_list = [
            psm
            for psm in psm_list
            if psm.rank
            <= search_conf[
                "report_psms"
            ]  # top rank == 1, ordered descending with score
        ]

    else:
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

    # Perform FDR filtering

    # HERE GOES MAPPING BACK
    for psm in psm_list:
        # create some key
        psm_id = psm.spec_idx

        # extract the matched peaks
        matched_peaks = psm.sage_feature.matched_peaks

        # save the matched peaks, BREAK for now until we have a way to save the matched peaks
        # psm.sage_feature.fragments.mz_calculated
        # psm.sage_feature.fragments.mz_experimental
        break

    # create a pandas dataframe
    # import pandas as pd
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', 2)

    PSM_pandas = psm_collection_to_pandas(psm_list, num_threads=num_threads)

    # you can also use double competition to get the q-values CREMA style
    TDC_pandas = target_decoy_competition_pandas(
        PSM_pandas, method="peptide_psm_peptide", score="hyperscore"
    )

    # precursor_cluster_stats

    print(TDC_pandas)
    ## SAVe under.
    # fragment_cluster_stats
    # results_sage_parquet
