"""
Input preparation routines shared by SIGMA experiments.
"""

import numpy as np
import scanpy as sc
import torch
from muon import atac as ac
from muon import prot as pt
from scipy.sparse import block_diag

from sigma.data.spatial_graph import (
    build_spatial_network,
    add_rna_similarity_edges,
    add_rna_similarity_edges_same_slice,
)
from sigma.data.triplet_mining import mine_cross_batch_triplets, mine_within_dataset_triplets
from sigma.toolkit import harmony, pca


def prepare_rna_protein_features(
    adata_rna,
    adata_protein,
    min_cells=10,
    n_top_genes=3000,
    rna_n_comps=30,
    protein_n_comps=30,
):
    sc.pp.filter_genes(adata_rna, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=rna_n_comps)

    adata_protein = adata_protein[adata_rna.obs_names].copy()
    pt.pp.clr(adata_protein)
    sc.pp.scale(adata_protein)
    adata_protein.obsm["feat"] = pca(adata_protein, n_comps=protein_n_comps)

    return adata_rna, adata_protein, [adata_rna, adata_protein]


def prepare_rna_protein_atac_features(
    adata_rna,
    adata_protein,
    adata_atac,
    min_cells=10,
    n_top_genes=3000,
    rna_n_comps=30,
    protein_n_comps=30,
    atac_n_comps=30,
    atac_scale_factor=1e4,
    atac_counts_per_cell_after=1e4,
):
    adata_rna, adata_protein, _ = prepare_rna_protein_features(
        adata_rna=adata_rna,
        adata_protein=adata_protein,
        min_cells=min_cells,
        n_top_genes=n_top_genes,
        rna_n_comps=rna_n_comps,
        protein_n_comps=protein_n_comps,
    )

    adata_atac = adata_atac[adata_rna.obs_names].copy()
    ac.pp.tfidf(adata_atac, scale_factor=atac_scale_factor)
    sc.pp.normalize_per_cell(adata_atac, counts_per_cell_after=atac_counts_per_cell_after)
    sc.pp.log1p(adata_atac)
    adata_atac.obsm["feat"] = pca(adata_atac, n_comps=atac_n_comps)
    return adata_rna, adata_protein, adata_atac, [adata_rna, adata_protein, adata_atac]


def prepare_rna_atac_features(
    adata_rna,
    adata_atac,
    min_cells=10,
    n_top_genes=3000,
    rna_n_comps=30,
    atac_n_comps=60,
    align_atac=True,
    atac_scale_factor=1e4,
    atac_counts_per_cell_after=1e4,
):
    sc.pp.filter_genes(adata_rna, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=rna_n_comps)

    if align_atac:
        adata_atac = adata_atac[adata_rna.obs_names].copy()
    ac.pp.tfidf(adata_atac, scale_factor=atac_scale_factor)
    sc.pp.normalize_per_cell(adata_atac, counts_per_cell_after=atac_counts_per_cell_after)
    sc.pp.log1p(adata_atac)
    adata_atac.obsm["feat"] = pca(adata_atac, n_comps=atac_n_comps)
    return adata_rna, adata_atac, [adata_rna, adata_atac]


def prepare_harmony_rna_protein_features(
    adata_rna,
    adata_protein,
    batch_key="batch",
    min_cells=10,
    n_top_genes=5000,
    rna_n_comps=50,
    protein_n_comps=30,
):
    sc.pp.filter_genes(adata_rna, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    adata_rna.raw = adata_rna
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]]
    adata_rna.obsm["X_pca"] = pca(adata_rna, n_comps=rna_n_comps)
    harmony(adata_rna, "X_pca", batch_key)

    pt.pp.clr(adata_protein)
    adata_protein.obsm["X_pca"] = pca(adata_protein, n_comps=protein_n_comps)
    harmony(adata_protein, "X_pca", batch_key)
    return adata_rna, adata_protein, [adata_rna, adata_protein]


def build_spatial_triplet_inputs(
    adata_list,
    rna_adata,
    device,
    n_neighbors,
    edge_ratio=0.005,
    feature_key="feat",
    triplet_neighbors=3,
    farthest_ratio=0.6,
):
    for adata in adata_list:
        build_spatial_network(adata, model="KNN", n_neighbors=n_neighbors)

    add_rna_similarity_edges(
        adata_list=adata_list,
        adata_rna=rna_adata,
        ratio=edge_ratio,
    )

    features = [torch.FloatTensor(adata.obsm[feature_key]).to(device) for adata in adata_list]
    edges = [torch.LongTensor(adata.uns["edgeList"]).to(device) for adata in adata_list]
    triplets = [
        mine_within_dataset_triplets(
            adata,
            key=feature_key,
            n_nearest_neighbors=triplet_neighbors,
            farthest_ratio=farthest_ratio,
        )
        for adata in adata_list
    ]
    return features, edges, triplets


def build_batch_triplet_inputs(
    adata_list,
    device,
    feature_key="X_pca_harmony",
    batch_key="batch",
    far_frac=0.8,
    top_k=1,
):
    features = [torch.FloatTensor(adata.obsm[feature_key]).to(device) for adata in adata_list]
    edges = [torch.LongTensor(adata.uns["edgeList"]).to(device) for adata in adata_list]
    triplets = [
        mine_cross_batch_triplets(
            adata.obsm[feature_key],
            adata.obs[batch_key],
            far_frac=far_frac,
            top_k=top_k,
        )
        for adata in adata_list
    ]
    return features, edges, triplets


def add_local_rna_similarity_edges(
    adata_list,
    rna_adata,
    ratio=0.005,
    min_cells=10,
    n_top_genes=5000,
    n_comps=50,
):
    adata_rna_tmp = rna_adata.copy()
    sc.pp.filter_genes(adata_rna_tmp, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata_rna_tmp, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata_rna_tmp, target_sum=1e4)
    sc.pp.log1p(adata_rna_tmp)
    adata_rna_tmp = adata_rna_tmp[:, adata_rna_tmp.var["highly_variable"]]
    adata_rna_tmp.obsm["feat_local"] = pca(adata_rna_tmp, n_comps=n_comps)
    add_rna_similarity_edges_same_slice(
        adata_list=adata_list,
        rna_feature=adata_rna_tmp.obsm["feat_local"],
        ratio=ratio,
    )


def stitch_slice_graphs(rna_adatas, adt_adatas, rna_adata, adt_adata):
    rna_adj = block_diag([adata.uns["adj"] for adata in rna_adatas.values()])
    adt_adj = block_diag([adata.uns["adj"] for adata in adt_adatas.values()])
    rna_adata.uns["edgeList"] = np.array(np.nonzero(rna_adj))
    adt_adata.uns["edgeList"] = np.array(np.nonzero(adt_adj))

