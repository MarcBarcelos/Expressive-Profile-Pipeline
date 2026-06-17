CATEGORIES = {
    # --- linguistic (TextDescriptives + custom) ---
    "Lexical Richness": (
        "n_unique_tokens", "proportion_unique_tokens", "mattr",
        "oov_ratio", "alpha_ratio",
    ),
    "Word Form Complexity": (
        "mean_word_length", "token_length_mean", "token_length_median",
        "token_length_std", "syllables_per_token_mean",
        "syllables_per_token_median", "syllables_per_token_std",
    ),
    "Document Scale": (
        "doc_length", "n_tokens", "n_characters", "n_sentences",
        "sentence_length_mean", "sentence_length_median", "sentence_length_std",
    ),
    "Repetition": (
        "duplicate_line_chr_fraction", "duplicate_paragraph_chr_fraction",
        "duplicate_ngram_chr_fraction_5", "duplicate_ngram_chr_fraction_6",
        "duplicate_ngram_chr_fraction_7", "duplicate_ngram_chr_fraction_8",
        "duplicate_ngram_chr_fraction_9", "duplicate_ngram_chr_fraction_10",
        "top_ngram_chr_fraction_2", "top_ngram_chr_fraction_3",
        "top_ngram_chr_fraction_4",
    ),
    "POS Distribution": (
        "pos_prop_ADJ", "pos_prop_ADP", "pos_prop_ADV", "pos_prop_AUX",
        "pos_prop_CCONJ", "pos_prop_DET", "pos_prop_INTJ", "pos_prop_NOUN",
        "pos_prop_NUM", "pos_prop_PART", "pos_prop_PRON", "pos_prop_PROPN",
        "pos_prop_PUNCT", "pos_prop_SCONJ", "pos_prop_SYM", "pos_prop_VERB",
        "pos_prop_X",
    ),
    "Syntax": (
        "dependency_distance_mean", "dependency_distance_std",
        "prop_adjacent_dependency_relation_mean",
        "prop_adjacent_dependency_relation_std",
    ),
    "Readability": (
        "flesch_reading_ease", "flesch_kincaid_grade", "smog", "gunning_fog",
        "automated_readability_index", "coleman_liau_index", "lix", "rix",
    ),
    "LM Entropy": (
        "H_unigram_win_mean_nats", "H_unigram_win_std_nats", "PPL_unigram_win_mean",
        "H_3gram_self_nats", "PPL_3gram_self_nats",
    ),

    # --- semantic (embedding-based) ---
    "Coherence & Trajectory": (
        "first_order_coherence", "second_order_coherence",
        "semantic_drift_start_end", "cumulative_semantic_trajectory_length", "semantic_trajectory_length_per_word",
        "segment_transition_variability", "semantic_range_max_dist",
    ),
    "On-topic vs. Dispersion": (
        "on_topic_consistency", "semantic_dispersion", "semantic_concentration",
    ),
    "Topic Strength & Volume": (
        "pc1_topic_strength", "pc_top5_strength", "semantic_volume_logdet",
    ),
    "Clustering & Diversity": (
        "kmeans_k", "topic_cluster_entropy", "topic_diversity_effective_k",
        "max_cluster_share", "cluster_focus_norm",
    ),
    "Book Size / Shape": (
        "n_chunks", "embedding_dim",
    ),
}

CATEGORY_OF = {metric: cat for cat, metrics in CATEGORIES.items() for metric in metrics}

DROP_THESE = {
    # Lexical Richness
    "n_unique_tokens", "proportion_unique_tokens", "alpha_ratio",
    # Word Form Complexity
    "token_length_mean", "token_length_median", "token_length_std",
    "syllables_per_token_median",
    # Document Scale
    "doc_length", "n_characters", "n_sentences",
    "sentence_length_median", "sentence_length_std",
    # Repetition
    "duplicate_line_chr_fraction", "duplicate_paragraph_chr_fraction",
    "duplicate_ngram_chr_fraction_5", "duplicate_ngram_chr_fraction_6",
    "duplicate_ngram_chr_fraction_7", "duplicate_ngram_chr_fraction_9",
    "duplicate_ngram_chr_fraction_10", "top_ngram_chr_fraction_3",
    "top_ngram_chr_fraction_4",
    # POS Distribution  (proportions sum to 1 -> drop one reference category)
    "pos_prop_X",
    # Readability
    "automated_readability_index", "coleman_liau_index", "flesch_kincaid_grade",
    "flesch_reading_ease", "lix", "rix", "smog",
    # LM Entropy  (drop perplexities, keep the entropies)
    "PPL_unigram_win_mean", "PPL_3gram_self_nats",
    # Coherence & Trajectory  (r=0.98 with first_order_coherence -> singular corr matrix)
    "second_order_coherence", "cumulative_semantic_trajectory_length",
    # On-topic vs. Dispersion
    "on_topic_consistency", "semantic_concentration",
    # Clustering & Diversity
    "kmeans_k", "topic_diversity_effective_k", "cluster_focus_norm",
    # Book Size / Shape  (not content)
    "n_chunks", "embedding_dim",
}

def select_features(columns, drop=()):
    drop = set(drop)
    unknown = drop - CATEGORY_OF.keys()
    if unknown:
        raise ValueError(f"drop names not in the taxonomy (typo?): {sorted(unknown)}")

    cols = set(columns)
    feature_cols = [m for m in CATEGORY_OF if m in cols and m not in drop]
    category_map = {m: CATEGORY_OF[m] for m in feature_cols}
    return feature_cols, category_map

#### IN USE ####
# feature_cols, category_map = select_features(full_df.columns, drop=DROP_THESE)
