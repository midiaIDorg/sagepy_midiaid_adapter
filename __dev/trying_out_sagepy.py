from sagepy.core import (EnzymeBuilder, Precursor, ProcessedSpectrum,
                         RawSpectrum, Representation, SageSearchConfiguration,
                         Scorer, SpectrumProcessor, Tolerance)
from tqdm import tqdm

import numpy as np
from pandas_ops.io import read_df
from pandas_ops.lex_ops import LexicographicIndex
from pandas_ops.stats import min_max, sum_real_good

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


### Example search of a sage database
precursor = Precursor(
    charge=2,
    mz=506.77,
)

intensity = np.array(
    [
        202.0,
        170.0,
        205.0,
        152.0,
        1069.0,
        595.0,
        198.0,
        805.0,
        187.0,
        194.0,
        197.0,
        169.0,
        196.0,
        209.0,
        638.0,
        372.0,
        235.0,
        399.0,
        194.0,
        185.0,
        181.0,
        170.0,
        407.0,
        150.0,
        157.0,
        175.0,
        273.0,
        1135.0,
        881.0,
        337.0,
        311.0,
        243.0,
        310.0,
        153.0,
        162.0,
        210.0,
        277.0,
        206.0,
        189.0,
        259.0,
        658.0,
        383.0,
        166.0,
        169.0,
        219.0,
        186.0,
        221.0,
        193.0,
        367.0,
        283.0,
        237.0,
        157.0,
        372.0,
        1276.0,
        1618.0,
        1102.0,
        404.0,
        232.0,
        456.0,
        765.0,
        507.0,
        223.0,
        258.0,
        402.0,
        187.0,
        158.0,
        153.0,
        304.0,
        218.0,
        223.0,
        156.0,
        1605.0,
        1165.0,
        1062.0,
        434.0,
        208.0,
        155.0,
        197.0,
        221.0,
        697.0,
        397.0,
        180.0,
        195.0,
        512.0,
        252.0,
        367.0,
        305.0,
        335.0,
        175.0,
        174.0,
        296.0,
        212.0,
    ],
    dtype=np.float32,
)

mz = np.array(
    [
        272.16873692,
        356.16844797,
        406.71079396,
        406.71396814,
        406.71714233,
        406.72031653,
        407.21246768,
        407.21564382,
        407.21881996,
        407.22199612,
        407.7144506,
        407.71762869,
        488.27537883,
        488.28581266,
        499.29228981,
        499.29580676,
        499.29932372,
        499.30284069,
        506.75478369,
        507.26157767,
        541.26272227,
        553.29188809,
        577.30432041,
        577.30810217,
        595.32672633,
        597.2907525,
        603.27568881,
        614.32036769,
        614.32426881,
        614.32816995,
        615.3272682,
        615.33117252,
        616.33108578,
        617.33572156,
        636.30924838,
        637.30619081,
        637.31016425,
        665.36284673,
        666.36197292,
        674.35335834,
        674.35744565,
        674.36153297,
        675.35511968,
        675.36330039,
        679.3531909,
        680.35044702,
        680.35455247,
        687.36822726,
        687.37648041,
        688.37547678,
        697.3616813,
        700.3617026,
        715.36157366,
        715.36578342,
        715.36999319,
        715.37420297,
        715.37841277,
        715.38262258,
        716.36384605,
        716.37227148,
        716.38069696,
        717.37103577,
        725.35228543,
        749.39291293,
        749.39722166,
        750.38424802,
        786.44692356,
        786.45575152,
        787.4492132,
        787.45804678,
        795.39284711,
        812.41777208,
        812.42225834,
        812.42674462,
        812.4312309,
        812.44020351,
        813.40504794,
        813.41851494,
        813.42300396,
        813.427493,
        813.43198205,
        813.44544927,
        814.43784098,
        828.42202737,
        828.4265576,
        851.43464868,
        899.45327427,
        899.46271517,
        912.45278821,
        913.44673363,
        915.45053417,
        915.46482091,
    ],
    dtype=np.float32,
)

raw_spectrum = RawSpectrum(
    file_id=1,
    spec_id="DEMO-SPEC",
    total_ion_current=12667.0,
    precursors=[precursor],
    mz=mz,
    intensity=intensity,
)

spec_processor = SpectrumProcessor(take_top_n=75)
query = spec_processor.process(raw_spectrum)


# UPDATE: pass modifications to the scorer, necessary for PTM handling
scorer = Scorer(
    report_psms=100,
    min_matched_peaks=5,
    variable_mods=variable_mods,
    static_mods=static_mods,
)
results = scorer.score(db=indexed_db, spectrum=query)
len(results)

pepidx = results[-1].peptide_idx
indexed_db[pepidx]

raw_spectrum_view = RawSpectrum(
    file_id=1,
    spec_id="DEMO-SPEC",
    total_ion_current=12667.0,
    precursors=[precursor],
    mz=mz[:],
    intensity=intensity[:],
)

spec_processor = SpectrumProcessor(take_top_n=75)
query_view = spec_processor.process(raw_spectrum_view)
results_view = scorer.score(db=indexed_db, spectrum=query_view)


# OK, now make mz and intensity vectors for general case.
precursor_stats_path = "tmp/clusters/tims/reformated/65/combined_cluster_stats.parquet"
fragment_stats_path = "tmp/clusters/tims/reformated/67/combined_cluster_stats.parquet"
edges_path = "tmp/edges/rough/76/rough_edges.startrek"

to_dict = lambda df: {c: df[c].to_numpy() for c in df}

precursor_stats = to_dict(
    read_df(
        precursor_stats_path,
        columns=[
            "ClusterID",
            "mz_wmean",
            "retention_time_wmean",
            "inv_ion_mobility_wmean",
            "intensity",
        ],
    )
)

fragment_stats = to_dict(
    read_df(fragment_stats_path, columns=["ClusterID", "mz_wmean", "intensity"])
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
        total_ion_current=fragment_TICs[MS1_ClusterID],
        precursors=[precursor],
        mz=fragment_mzs[lx.idx[i] : lx.idx[i + 1]],
        intensity=fragment_intensities[lx.idx[i] : lx.idx[i + 1]],
    )
    queries.append(spec_processor.process(raw_spectrum))

# the number of reported ptms?

potential_ptms -> each row = one PTM
paired_fragments = each group of rows indexed by row number from potential_ptms is a group of fragments backing the PTM.
