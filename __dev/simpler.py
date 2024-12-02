import numpy as np
from pandas_ops.io import read_df
from pandas_ops.lex_ops import LexicographicIndex
from pandas_ops.stats import min_max, sum_real_good
from sagepy.core import (EnzymeBuilder, Precursor, ProcessedSpectrum,
                         RawSpectrum, Representation, SageSearchConfiguration,
                         Scorer, SpectrumProcessor, Tolerance)
from tqdm import tqdm

# configure a trypsin-like digestor of fasta files
enzyme_builder = EnzymeBuilder(
    missed_cleavages=2,
    min_len=5,
    max_len=50,
    cleave_at="KR",
    restrict="P",
    c_terminal=True,
)

# UPDATE: Modification handling is simplified, using canonical UNIMOD notation
static_mods = {"C": "[UNIMOD:4]"}  # static cysteine modification
variable_mods = {"M": ["[UNIMOD:35]"]}

precursor_stats_path = "tmp/clusters/tims/reformated/65/combined_cluster_stats.parquet"
fragment_stats_path = "tmp/clusters/tims/reformated/67/combined_cluster_stats.parquet"
edges_path = "tmp/edges/rough/76/rough_edges.startrek"
fasta_path = "fastas/Human_2024_02_16_UniProt_Taxon9606_Reviewed_20434entries_contaminant_tenzer.fasta"

with open(fasta_path, "r") as infile:
    fasta = infile.read()

# set-up a config for a sage-database
sage_config = SageSearchConfiguration(
    fasta=fasta,
    static_mods=static_mods,
    variable_mods=variable_mods,
    enzyme_builder=enzyme_builder,
    generate_decoys=True,
    bucket_size=2**14,
)


# generate the database for searching against
indexed_db = sage_config.generate_indexed_database()

to_dict = lambda df: {c: df[c].to_numpy() for c in df}


precursor_stats = to_dict(
    read_df(
        precursor_stats_path,
        # columns=[
        #     "ClusterID",
        #     "mz_wmean",
        #     "retention_time_wmean",
        #     "inv_ion_mobility_wmean",
        #     "intensity",
        # ],
    )
)

fragment_stats = to_dict(
    read_df(fragment_stats_path, 
        # columns=["ClusterID", "mz_wmean", "intensity"]
    )
)
fragment_stats["intensity"] = fragment_stats["intensity"].astype(np.float32)
fragment_stats["mz_wmean"] = fragment_stats["mz_wmean"].astype(np.float32)

edges = to_dict(read_df(edges_path))
lx = LexicographicIndex(edges["MS1_ClusterID"])

# this actually copies into RAM.
fragment_mzs = fragment_stats["mz_wmean"][edges["MS2_ClusterID"]]
fragment_intensities = fragment_stats["intensity"][edges["MS2_ClusterID"]]

MS1_ClusterIDs = edges["MS1_ClusterID"][lx.idx[:-1]]

fragment_TICs = lx.map(sum_real_good, fragment_intensities)
len(fragment_TICs)
MS1_ClusterIDs.max()


spec_processor = SpectrumProcessor(take_top_n=75)

queries = []
for i in tqdm(range(len(lx))):
    MS1_ClusterID = MS1_ClusterIDs[i]
    precursor = Precursor(
        mz=precursor_stats["mz_wmean"][MS1_ClusterID],
        charge=None,
        intensity=precursor_stats["intensity"][MS1_ClusterID],
        inverse_ion_mobility=precursor_stats["inv_ion_mobility_wmean"][MS1_ClusterID],
    )
    raw_spectrum = RawSpectrum(
        file_id=1,
        spec_id=str(MS1_ClusterID),
        total_ion_current=fragment_TICs[i],
        precursors=[precursor],
        mz=fragment_mzs[lx.idx[i] : lx.idx[i + 1]],
        intensity=fragment_intensities[lx.idx[i] : lx.idx[i + 1]],
    )
    queries.append(spec_processor.process(raw_spectrum))

# the number of reported ptms?

potential_ptms -> each row = one PTM
paired_fragments = each group of rows indexed by row number from potential_ptms is a group of fragments backing the PTM.
