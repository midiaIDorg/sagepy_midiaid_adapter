import functools
import json
import os
from pathlib import Path

import click
import sagepy.core.scoring
from sagepy.core import Precursor, RawSpectrum, Scorer, SpectrumProcessor
from sagepy.qfdr.tdc import (
    assign_sage_peptide_q,
    assign_sage_protein_q,
    assign_sage_spectrum_q,
    target_decoy_competition_pandas,
)
from sagepy.utility import generate_search_configurations, psm_collection_to_pandas
from tqdm import tqdm

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas_ops.io import read_df
from pandas_ops.lex_ops import LexicographicIndex
from pandas_ops.stats import sum_real_good


def to_dict(df: pd.DataFrame):
    return {c: df[c].to_numpy() for c in df}


def create_query(
    precursor_stats: pd.DataFrame,
    fragment_stats: pd.DataFrame,
    edges: pd.DataFrame,
    spec_processor: SpectrumProcessor,
    progressbar_desc: str = "Prepping SAGEPY queries.",
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

    queries = []
    for i in tqdm(range(len(lx)), desc=progressbar_desc):
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
            scan_start_time=precursor_stats["retention_time_wmean"][MS1_ClusterID],
            ion_injection_time=precursor_stats["retention_time_wmean"][MS1_ClusterID],
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

    spec_processor = SpectrumProcessor(take_top_n=75)
    queries = create_query(precursor_stats, fragment_stats, edges, spec_processor)

    scorer = Scorer(
        report_psms=search_conf["report_psms"],
        min_matched_peaks=search_conf["min_peaks"],
        variable_mods=search_conf["database"]["variable_mods"],
        static_mods=search_conf["database"]["static_mods"],
    )

    # iterate over search configurations
    num_splits = 2

    dbs = generate_search_configurations(
        fasta_path=fasta,
        num_splits=num_splits,
        min_len=search_conf["database"]["enzyme"]["min_len"],
        max_len=search_conf["database"]["enzyme"]["max_len"],
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
    merged_psms = functools.reduce(
        functools.partial(sagepy.core.scoring.merge_psm_dicts, max_hits=25), psms
    )
    psm_list = [
        psm
        for psms in merged_psms.values()
        for psm in psms
        if psm.rank
        <= search_conf["report_psms"]  # top rank == 1, ordered descending with score
    ]

    # TODO: extract and save the SAGE matched peaks

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

    # assign SAGE q-values
    assign_sage_spectrum_q(psm_list)
    assign_sage_peptide_q(psm_list)
    assign_sage_protein_q(psm_list)

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
