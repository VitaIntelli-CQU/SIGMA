import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def build_spatial_network(adata, radius=None, n_neighbors=None, model='KNN', verbose=True, include_self=False):
    """
    Construct spatial neighbor graph from spatial coordinates.
    """
    spatial = adata.obsm['spatial']
    if model == 'KNN':
        adata.uns['adj'] = kneighbors_graph(spatial, n_neighbors=n_neighbors, mode='connectivity', include_self=include_self)
    elif model == 'Radius':
        adata.uns['adj'] = radius_neighbors_graph(spatial, radius=radius, mode='connectivity', include_self=include_self)

    edgeList = np.nonzero(adata.uns['adj'])
    adata.uns['edgeList'] = np.array([edgeList[0], edgeList[1]])

    if verbose:
        print('The graph contains %d edges, %d cells.' % (adata.uns['edgeList'].shape[1], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (adata.uns['edgeList'].shape[1] / adata.n_obs))


def add_rna_similarity_edges(adata_list, adata_rna, ratio=0.01):
    n_obs = adata_rna.n_obs
    rna_feat = adata_rna.obsm["feat"]
    sim_matrix = cosine_similarity(rna_feat)

    original_edge_count = adata_list[0].uns["edgeList"].shape[1]
    n_new_edges = int(original_edge_count * ratio)

    print(f"Original edges: {original_edge_count}, planned new edges: {n_new_edges}")

    existing_edges = set()
    for adata in adata_list:
        edges = adata.uns["edgeList"]
        for i in range(edges.shape[1]):
            e = tuple(sorted([edges[0, i], edges[1, i]]))
            existing_edges.add(e)

    upper_tri_indices = np.triu_indices(n_obs, k=1)
    similarities = sim_matrix[upper_tri_indices]
    top_indices = np.argsort(similarities)[-n_new_edges:]

    new_edges_list = []
    for idx in top_indices:
        i = upper_tri_indices[0][idx]
        j = upper_tri_indices[1][idx]
        edge = tuple(sorted([i, j]))
        if edge not in existing_edges:
            new_edges_list.append([i, j])
            new_edges_list.append([j, i])

    print(f"Actual directed new edges after deduplication: {len(new_edges_list)}")

    for adata in adata_list:
        old_edges = adata.uns["edgeList"]
        new_edges = np.array(new_edges_list).T
        combined_edges = np.hstack([old_edges, new_edges])
        adata.uns["edgeList"] = combined_edges

        rows = combined_edges[0]
        cols = combined_edges[1]
        data = np.ones(combined_edges.shape[1])
        adj = csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
        adj = adj + adj.T
        adata.uns["adj"] = (adj > 0).astype(int)

        print(f"  Final modality edge count: {combined_edges.shape[1]}")


def add_rna_similarity_edges_same_slice(
    adata_list,
    rna_feature,
    ratio=0.005,
):
    n_obs = rna_feature.shape[0]
    sim_matrix = cosine_similarity(rna_feature)

    original_edge_count = adata_list[0].uns["edgeList"].shape[1]
    n_new_edges_target = int(original_edge_count * ratio)

    print(
        f"Original edges: {original_edge_count}, "
        f"planned new edges: {n_new_edges_target}"
    )

    existing_edges = set()
    for adata in adata_list:
        edges = adata.uns["edgeList"]
        for i in range(edges.shape[1]):
            a, b = int(edges[0, i]), int(edges[1, i])
            edge = tuple(sorted([a, b]))
            existing_edges.add(edge)

    upper_tri_indices = np.triu_indices(n_obs, k=1)
    similarities = sim_matrix[upper_tri_indices]
    sorted_idx = np.argsort(similarities)[::-1]

    new_undirected_edges = []
    for idx in sorted_idx:
        i = int(upper_tri_indices[0][idx])
        j = int(upper_tri_indices[1][idx])
        edge = tuple(sorted([i, j]))

        if edge in existing_edges:
            continue

        new_undirected_edges.append(edge)
        existing_edges.add(edge)

        if len(new_undirected_edges) >= n_new_edges_target:
            break

    new_edges_list = []
    for i, j in new_undirected_edges:
        new_edges_list.append([i, j])
        new_edges_list.append([j, i])

    print(f"Actual undirected new edges: {len(new_undirected_edges)}")
    print(f"Actual directed new edges: {len(new_edges_list)}")

    if len(new_edges_list) == 0:
        return

    new_edges = np.array(new_edges_list).T

    for adata in adata_list:
        old_edges = adata.uns["edgeList"]
        combined_edges = np.hstack([old_edges, new_edges])
        adata.uns["edgeList"] = combined_edges

        rows = combined_edges[0]
        cols = combined_edges[1]
        data = np.ones(combined_edges.shape[1], dtype=np.int8)
        adj = csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
        adata.uns["adj"] = (adj > 0).astype(int)

        print(f"  Final modality edge count: {combined_edges.shape[1]}")

