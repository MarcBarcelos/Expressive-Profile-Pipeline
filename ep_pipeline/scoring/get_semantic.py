import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# semantic metrics based on sentence embeddings, inspired by Elkins et al. (2023) "Thematic and semantic coherence in discourse: A computational analysis of 1000 texts" https://arxiv.org/abs/2305.14392
SEMANTIC_KEYS = (
    "first_order_coherence", "second_order_coherence", "semantic_drift_start_end",
    "cumulative_semantic_trajectory_length", "semantic_trajectory_length_per_word",
    "segment_transition_variability", "semantic_range_max_dist", "semantic_dispersion",
    "pc1_topic_strength", "pc_top5_strength", "semantic_volume_logdet",
    "topic_cluster_entropy", "max_cluster_share",
)

def _empty_metrics():
    """The defined output for text that yields no chunks: all-NaN.
    This is part of the metric's contract, not failure handling."""
    return {k: np.nan for k in SEMANTIC_KEYS}

# split text into chunks of a specified size with a specified overlap, which is used for calculating semantic metrics based on sentence embeddings
def chunk_by_words(text, chunk_size=100, overlap=3):
    words = text.split()                              # split the input text into a list of words based on whitespace
    step = max(1, chunk_size - overlap)               # calculate the step size for sliding the window of words to create chunks, ensuring that there is at least one word in each chunk even if the specified overlap is greater than or equal to the chunk size
    return [" ".join(words[i:i + chunk_size]).strip() # create chunks of words by joining the appropriate slice of the list of words based on the calculated step size and specified chunk size, and stripping any leading or trailing whitespace from the resulting chunk
            for i in range(0, len(words), step)
            if words[i:i + chunk_size]]

# L2-normalize a vector, which is used for calculating cosine similarity between sentence embeddings when computing semantic metrics based on sentence embeddings
def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

# calculate the adjusted cosine similarity between adjacent chunks of sentence embeddings, which is used as part of the calculation of first-order and second-order coherence metrics in the semantic analysis of texts
def adj_cos(E, lag=1):
    return np.sum(E[:-lag] * E[lag:], axis=1) if E.shape[0] > lag else np.array([])

# calculate the Shannon entropy of a distribution given by counts, which is used for calculating the topic cluster entropy metric in the semantic analysis of texts based on sentence embeddings
def shannon_entropy(counts, eps=1e-12):
    p = counts / (counts.sum() + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())

# compute the semantic metrics for a given text using a specified embedding model and parameters for chunking and analysis, which includes calculating various coherence, drift, range, dispersion, topic strength, volume, and clustering metrics
def compute_semantic_metrics(text, model, chunk_size=100, overlap=3,
                             batch_size=128, prefix="passage: ", seed=0):
    sub_chunks = chunk_by_words(text, chunk_size=chunk_size, overlap=overlap)
    if not sub_chunks:
        return _empty_metrics()

    E = model.encode([prefix + c for c in sub_chunks], batch_size=batch_size,
                     show_progress_bar=False, convert_to_numpy=True,
                     normalize_embeddings=True).astype(np.float32)
    n = E.shape[0]

    fo_vals = adj_cos(E, 1)
    so_vals = adj_cos(E, 2)
    fo_mean = float(np.mean(fo_vals)) if fo_vals.size else np.nan
    so_mean = float(np.mean(so_vals)) if so_vals.size else np.nan
    step_dists = 1.0 - fo_vals if fo_vals.size else np.array([])
    traj_len = float(np.sum(step_dists)) if step_dists.size else np.nan
    traj_per_word = float(traj_len / (len(text.split())))
    trans_var = float(np.std(step_dists)) if step_dists.size else np.nan
    drift = float(1.0 - np.dot(E[0], E[-1])) if n > 1 else np.nan
    sem_range = 1.0 - float(np.min(E @ E.T)) if n >= 2 else np.nan
    doc_vec = l2_normalize(E.mean(axis=0))
    dispersion = float(np.mean(1.0 - (E @ doc_vec)))

    pca_max = min(20, E.shape[1], max(1, n - 1))
    if pca_max >= 1:
        pca = PCA(n_components=pca_max, random_state=seed).fit(E)
        pc1_str = float(pca.explained_variance_ratio_[0])
        pc5_str = float(pca.explained_variance_ratio_[:min(5, pca_max)].sum())
        X = pca.transform(E)
        if n >= 3 and pca_max >= 2:
            cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
            sign, logdet = np.linalg.slogdet(cov)
            vol_logdet = float(logdet) if sign > 0 else np.nan
        else:
            vol_logdet = np.nan
    else:
        X = E
        pc1_str = pc5_str = vol_logdet = np.nan

    if n >= 4:
        k = int(np.clip(np.sqrt(n), 2, 10))
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        counts = np.bincount(labels, minlength=k).astype(float)
        clust_ent = shannon_entropy(counts)
        max_cluster = float(counts.max() / counts.sum())
    else:
        clust_ent = max_cluster = np.nan

    return {
        "first_order_coherence": fo_mean, "second_order_coherence": so_mean,
        "semantic_drift_start_end": drift,
        "cumulative_semantic_trajectory_length": traj_len,
        "semantic_trajectory_length_per_word": traj_per_word,
        "segment_transition_variability": trans_var,
        "semantic_range_max_dist": sem_range, "semantic_dispersion": dispersion,
        "pc1_topic_strength": pc1_str, "pc_top5_strength": pc5_str,
        "semantic_volume_logdet": vol_logdet,
        "topic_cluster_entropy": clust_ent, "max_cluster_share": max_cluster,
    }