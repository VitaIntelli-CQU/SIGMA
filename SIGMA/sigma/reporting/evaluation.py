"""
Clustering evaluation helpers for SIGMA notebooks.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import chisquare
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors

from sigma.toolkit import clustering


def clustering_metric_scores(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "Homo": homogeneity_score(y_true, y_pred),
        "V-measure": v_measure_score(y_true, y_pred),
        "FMI": fowlkes_mallows_score(y_true, y_pred),
        "MI": mutual_info_score(y_true, y_pred),
    }


def print_metric_scores(scores):
    for key in ["ARI", "NMI", "AMI", "Homo", "V-measure", "FMI", "MI"]:
        print(f"{key}: {scores[key]:.4f}")


def report_clustering_metrics(
    adata,
    label_key,
    cluster_key,
    embedding_key="SIGMA",
    n_clusters=7,
    method="mclust",
    use_pca=True,
):
    clustering(
        adata,
        key=embedding_key,
        add_key=cluster_key,
        n_clusters=n_clusters,
        method=method,
        use_pca=use_pca,
    )
    scores = clustering_metric_scores(adata.obs[label_key], adata.obs[cluster_key])
    print_metric_scores(scores)
    return scores


def cluster_and_score_ari(
    adata,
    label_key,
    cluster_key="SIGMA",
    embedding_key="SIGMA",
    n_clusters=7,
    method="mclust",
    use_pca=True,
    drop_missing_labels=False,
):
    clustering(
        adata,
        key=embedding_key,
        add_key=cluster_key,
        n_clusters=n_clusters,
        method=method,
        use_pca=use_pca,
    )
    adata_eval = adata[~adata.obs[label_key].isna()] if drop_missing_labels else adata
    ari = adjusted_rand_score(adata_eval.obs[label_key], adata_eval.obs[cluster_key])
    print(ari)
    return ari


def compute_ilisi_score_fallback(X, batch_labels, n_neighbors=90):
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)[:, 1:]
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    if n_batches <= 1:
        return np.nan
    lisi_values = []
    for neigh in indices:
        neigh_batches = batch_labels[neigh]
        _, counts = np.unique(neigh_batches, return_counts=True)
        probs = counts / counts.sum()
        lisi_values.append(1.0 / np.sum(probs ** 2))
    ilisi_raw = float(np.mean(lisi_values))
    return (ilisi_raw - 1.0) / (n_batches - 1.0)


def compute_kbet_acceptance_fallback(X, batch_labels, n_neighbors=90, alpha=0.05):
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    unique_batches, global_counts = np.unique(batch_labels, return_counts=True)
    global_probs = global_counts / global_counts.sum()
    if len(unique_batches) <= 1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)[:, 1:]
    accepted = []
    for neigh in indices:
        neigh_batches = batch_labels[neigh]
        obs_counts = np.array([(neigh_batches == b).sum() for b in unique_batches], dtype=float)
        exp_counts = global_probs * obs_counts.sum()
        keep = exp_counts > 0
        if keep.sum() <= 1:
            continue
        _, p_value = chisquare(obs_counts[keep], f_exp=exp_counts[keep])
        accepted.append(p_value >= alpha)
    return float(np.mean(accepted)) if accepted else np.nan


def run_cluster_range_benchmark(
    adata,
    method_name="SIGMA",
    embedding_key="SIGMA",
    batch_key="batch",
    label_key="anno",
    cluster_range=(4, 5, 6, 7, 8),
    clustering_method="mclust",
    use_pca=True,
):
    bio_metric_order = ["ARI", "FMI", "NMI", "AMI", "MI", "V-Measure", "Homo"]
    batch_metric_order = ["iLISI", "kBET"]

    if embedding_key not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain {embedding_key!r}.")
    if batch_key not in adata.obs:
        raise KeyError(f"adata.obs does not contain {batch_key!r}.")
    if label_key not in adata.obs:
        raise KeyError(f"adata.obs does not contain {label_key!r}.")

    results = []
    for n_cluster in cluster_range:
        print(f"\n===== Clustering: {n_cluster} =====")
        cluster_key = f"{method_name}_{n_cluster}"
        clustering(
            adata,
            key=embedding_key,
            add_key=cluster_key,
            n_clusters=n_cluster,
            method=clustering_method,
            use_pca=use_pca,
        )
        adata_labeled = adata[~adata.obs[label_key].isna()].copy()
        y_true = adata_labeled.obs[label_key].astype(str)
        y_pred = adata_labeled.obs[cluster_key].astype(str)
        metrics_dict = {
            "n_clusters": n_cluster,
            "cluster_key": cluster_key,
            "ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
            "AMI": adjusted_mutual_info_score(y_true, y_pred),
            "Homo": homogeneity_score(y_true, y_pred),
            "V-Measure": v_measure_score(y_true, y_pred),
            "FMI": fowlkes_mallows_score(y_true, y_pred),
            "MI": mutual_info_score(y_true, y_pred),
        }
        results.append(metrics_dict)
        for k, v in metrics_dict.items():
            if k not in ["n_clusters", "cluster_key"]:
                print(f"{k}: {v:.6f}")

    result_df = pd.DataFrame(results)
    print("\n==============================")
    print("All Biological Results")
    print("==============================")
    print(result_df)
    mean_result = result_df.drop(columns=["n_clusters", "cluster_key"]).mean()
    print("\n==============================")
    print("Mean Biological Results (4~8 clusters)")
    print("==============================")
    for k, v in mean_result.items():
        print(f"{k}: {v:.6f}")

    batch_source = "fallback"
    try:
        import scib

        adata_batch_eval = adata[~adata.obs[label_key].isna()].copy()
        sc.pp.neighbors(adata_batch_eval, use_rep=embedding_key, n_neighbors=30)
        ilisi = scib.me.ilisi_graph(
            adata_batch_eval,
            batch_key=batch_key,
            type_="embed",
            use_rep=embedding_key,
        )
        kbet = scib.me.kBET(
            adata_batch_eval,
            batch_key=batch_key,
            label_key=label_key,
            type_="embed",
            embed=embedding_key,
        )
        batch_source = "scib"
    except Exception as e:
        print(f"\nscIB batch metrics failed; using fallback implementation. Reason: {e}")
        adata_batch_eval = adata[~adata.obs[label_key].isna()].copy()
        X = np.asarray(adata_batch_eval.obsm[embedding_key])
        batches = adata_batch_eval.obs[batch_key].astype(str).to_numpy()
        ilisi = compute_ilisi_score_fallback(X, batches, n_neighbors=90)
        kbet = compute_kbet_acceptance_fallback(X, batches, n_neighbors=90, alpha=0.05)

    summary_result = mean_result.copy()
    summary_result["iLISI"] = float(ilisi)
    summary_result["kBET"] = float(kbet)
    summary_result["Bio conservation"] = summary_result[bio_metric_order].mean()
    summary_result["Batch correction"] = summary_result[batch_metric_order].mean()
    summary_result["Total"] = summary_result[bio_metric_order + batch_metric_order].mean()
    summary_df = pd.DataFrame(summary_result, columns=[method_name]).T
    summary_df = summary_df[
        bio_metric_order
        + batch_metric_order
        + ["Bio conservation", "Batch correction", "Total"]
    ]

    print("\n==============================")
    print("Batch Metrics")
    print("==============================")
    print(f"Batch metric source: {batch_source}")
    print(f"iLISI: {summary_result['iLISI']:.6f}")
    print(f"kBET: {summary_result['kBET']:.6f}")
    print("\n==============================")
    print("Final Summary Metrics")
    print("==============================")
    for k, v in summary_result.items():
        print(f"{k}: {v:.6f}")
    print("\n==============================")
    print("Summary Table")
    print("==============================")
    print(summary_df)

    return result_df, mean_result, summary_result, summary_df

