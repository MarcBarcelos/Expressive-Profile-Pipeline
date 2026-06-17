from dataclasses import dataclass

@dataclass
class MetricsConfig:
    # lingustic features
    spacy_model: str = "en_core_web_sm" # spacy model to use for calculating linguistic features
    mattr_window: int = 100             # changes size of the window for calculating mattr
    entropy_window: int = 100           # changes size of the window for calculating windowed unigram entropy
    trigram_test_frac: float = 0.2      # fraction of trigrams to test for calculating trigram entropy
    trigram_alpha: float = 0.1          # alpha value for calculating trigram entropy

    # text descriptives configs
    td_metrics = ["descriptive_stats", "readability", "dependency_distance", "coherence", "pos_proportions", "quality"] 
                                        # excluded "information_theory" because already have other measures of entropy, and easy to fail on short texts; None would give all metrics

    # embedding features
    embed_chunk_size: int = 100         # chunk size for calculating embedding features
    embed_overlap: int = 3              # overlap size for calculating embedding features
    batch_size: int = 128               # batch size for calculating embedding features
    e5_prefix: str = "passage: "        # prefix to add to text when calculating e5 embeddings

    # structural features
    n_jobs: int = 8                     # number of parallel jobs able to be used
    checkpoint_every: int = 50          # checkpoint every n documents when calculating features
    seed: int = 2001                    # random seed for reproducibility

    # efa features
    n_permutations = 500                #eigenvalues aquired from 500 permutation of random data
    top_n = 6                           # Number of loadings to retreive from each dimension
