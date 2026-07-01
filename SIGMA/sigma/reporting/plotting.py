"""
Small plotting helpers for SIGMA notebooks.
"""

import pandas as pd

from sigma.toolkit import getcolordict


def default_matched_palettes(adata, cluster_key, label_key, base_colors):
    label_categories = list(pd.Categorical(adata.obs[label_key]).categories)
    cluster_categories = list(pd.Categorical(adata.obs[cluster_key]).categories)
    label_palette = {
        str(cat): base_colors[i % len(base_colors)]
        for i, cat in enumerate(label_categories)
    }
    adata.uns[f"{label_key}_colors"] = [
        label_palette[str(cat)]
        for cat in label_categories
    ]
    table = pd.crosstab(
        adata.obs[cluster_key].astype(str),
        adata.obs[label_key].astype(str),
    )
    cluster_to_label = table.idxmax(axis=1).to_dict()
    adata.uns[f"{cluster_key}_colors"] = [
        label_palette[cluster_to_label[str(cat)]]
        for cat in cluster_categories
    ]
    return label_palette


def mapped_annotation_palettes(adata, cluster_key, label_key, colors):
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
    adata.obs[label_key] = adata.obs[label_key].astype(str)
    label_palette = dict(zip(adata.obs[label_key].unique(), colors))
    cluster_palette = getcolordict(
        adata,
        cluster_key,
        label_key,
        label_palette,
    )
    return label_palette, cluster_palette

