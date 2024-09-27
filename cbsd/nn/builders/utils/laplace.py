# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math

import torch
import torch_sparse
from torch_geometric.utils import degree, spmm

'''def dirichlet_energy(L, f, size):
    """Returns the Dirichlet energy of the signal f under the sheaf Laplacian L."""
    right = torch_sparse.spmm(L[0], L[1], size, size, f)
    energy = f.t() @ right
    return energy.item()'''


def sheaf_dirichlet_energy(L, x, normalised=False):
    """Returns the Dirichlet energy of the signal x under the sheaf Laplacian L."""
    right = spmm(L, x)
    energy = torch.trace(x.t() @ right)
    if normalised:
        norm = torch.linalg.norm(x, ord="fro")
        return energy / norm**2
    else:
        return energy


def dirichlet_energy(x, edge_index, edge_weights=1, normalised=True):
    edge_difference = edge_weights * (
        torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1) ** 2
    )
    energy = edge_difference.sum() / 2
    if normalised:
        norm = torch.linalg.norm(x, ord="fro")
        return energy / norm**2
    else:
        return energy


def compute_left_right_map_index(edge_index, full_matrix=False):
    """Computes indices for lower triangular matrix or full matrix"""
    edge_to_idx = dict()
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        edge_to_idx[(source, target)] = e

    left_index, right_index = [], []
    row, col = [], []
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        if source < target or full_matrix:
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])

            row.append(source)
            col.append(target)

    left_index = torch.tensor(
        left_index, dtype=torch.long, device=edge_index.device
    )
    right_index = torch.tensor(
        right_index, dtype=torch.long, device=edge_index.device
    )
    left_right_index = torch.vstack([left_index, right_index])

    row = torch.tensor(row, dtype=torch.long, device=edge_index.device)
    col = torch.tensor(col, dtype=torch.long, device=edge_index.device)
    new_edge_index = torch.vstack([row, col])

    if full_matrix:
        assert len(left_index) == edge_index.size(1)
    else:
        assert len(left_index) == edge_index.size(1) // 2

    return left_right_index, new_edge_index


def compute_learnable_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])

    row, col = edge_index
    device = edge_index.device
    row_template = (
        torch.arange(0, learned_d, device=device)
        .view(1, -1, 1)
        .tile(1, 1, learned_d)
    )
    col_template = torch.transpose(row_template, dim0=1, dim1=2)

    non_diag_row_indices = (
        row_template + total_d * row.reshape(-1, 1, 1)
    ).reshape(1, -1)
    non_diag_col_indices = (
        col_template + total_d * col.reshape(-1, 1, 1)
    ).reshape(1, -1)
    non_diag_indices = torch.cat(
        (non_diag_row_indices, non_diag_col_indices), dim=0
    )

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (
        row_template + total_d * diag.reshape(-1, 1, 1)
    ).reshape(1, -1)
    diag_col_indices = (
        col_template + total_d * diag.reshape(-1, 1, 1)
    ).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def compute_learnable_diag_laplacian_indices(
    size, edge_index, learned_d, total_d
):
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(0, learned_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d * row.unsqueeze(1)).reshape(
        1, -1
    )
    non_diag_col_indices = (col_template + total_d * col.unsqueeze(1)).reshape(
        1, -1
    )
    non_diag_indices = torch.cat(
        (non_diag_row_indices, non_diag_col_indices), dim=0
    )

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d * diag.unsqueeze(1)).reshape(
        1, -1
    )
    diag_col_indices = (col_template + total_d * diag.unsqueeze(1)).reshape(
        1, -1
    )
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def compute_fixed_diag_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(learned_d, total_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d * row.unsqueeze(1)).reshape(
        1, -1
    )
    non_diag_col_indices = (col_template + total_d * col.unsqueeze(1)).reshape(
        1, -1
    )
    non_diag_indices = torch.cat(
        (non_diag_row_indices, non_diag_col_indices), dim=0
    )

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d * diag.unsqueeze(1)).reshape(
        1, -1
    )
    diag_col_indices = (col_template + total_d * diag.unsqueeze(1)).reshape(
        1, -1
    )
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrices: A batch of matrices.
        p: Power.
        positive_definite: If positive definite
    Returns:
        Power of each matrix in the batch.
    """
    # vals, vecs = torch.linalg.eigh(matrices)
    # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
    vecs, vals, _ = torch.linalg.svd(matrices)
    good = (
        vals
        > vals.max(-1, True).values
        * vals.size(-1)
        * torch.finfo(vals.dtype).eps
    )
    vals = vals.pow(p).where(
        good, torch.zeros((), device=matrices.device, dtype=matrices.dtype)
    )
    matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
    return matrix_power


def mergesp(index1, value1, index2, value2):
    """Merges two sparse matrices with disjoint indices into one."""
    assert index1.dim() == 2 and index2.dim() == 2
    assert value1.dim() == 1 and value2.dim() == 1
    assert index1.size(1) == value1.numel()
    assert index2.size(1) == value2.numel()
    assert index1.size(0) == 2 and index2.size(0) == 2

    index = torch.cat([index1, index2], dim=1)
    val = torch.cat([value1, value2])
    return index, val


### Build dense laplacian


def get_edge_index_dict(edge_index, undirected=True):
    """Computes a dictionary mapping the undirected edges in edge_index to an ID."""
    assert edge_index.size(1) % 2 == 0

    E = edge_index.size(1)
    edge_idx_dict = dict()
    next_id = 0

    for e in range(E):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        if undirected:
            edge = tuple(sorted([source, target]))
        else:
            edge = tuple([source, target])

        # Generate or retrieve the edge index
        if edge not in edge_idx_dict:
            edge_idx_dict[edge] = next_id
            next_id += 1

    return edge_idx_dict


def compute_incidence_index(edge_index, d):
    """Computes the indices of a sheaf coboundary matrix from the edge_index of the graph."""
    assert edge_index.size(1) % 2 == 0

    edge_idx_dict = get_edge_index_dict(edge_index)
    index = []

    for edge in range(edge_index.size(1)):
        source = edge_index[0, edge].item()
        target = edge_index[1, edge].item()
        edge_key = tuple(sorted([source, target]))

        top_x = edge_idx_dict[edge_key] * d
        top_y = source * d
        for i, j in itertools.product(range(d), range(d)):
            index.append([top_x + i, top_y + j])

    incidence_index = torch.tensor(index, dtype=torch.long).T
    assert list(incidence_index.size()) == [2, edge_index.size(1) * (d**2)]
    return incidence_index


def build_dense_laplacian(
    size,
    edge_index,
    maps,
    d,
    normalised=False,
    diagonal_maps=False,
    values=None,
    edge_weights=None,
):
    """Builds a sheaf laplacian from a given graph using naive dense computations (used for testing)."""
    assert edge_index.size(1) % 2 == 0
    if diagonal_maps:
        assert len(maps.size()) == 2
        assert maps.size(1) == d

    E = edge_index.size(1) // 2
    N = size
    Delta = torch.zeros(size=(E * d, N * d), dtype=torch.float64)
    undirected_edge_idx_dict = get_edge_index_dict(edge_index)
    directed_edge_idx_dict = get_edge_index_dict(edge_index, undirected=False)

    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        edge_key = tuple(sorted([source, target]))

        # Generate the positions in the block matrix
        top_x = undirected_edge_idx_dict[edge_key] * d
        top_y = source * d

        orient = -1 if edge_key[0] == source else 1
        if edge_weights is not None:
            factor1_idx, factor2_idx = (
                directed_edge_idx_dict[(source, target)],
                directed_edge_idx_dict[(target, source)],
            )
            assert edge_weights[factor1_idx] == edge_weights[factor2_idx]
            maps[e] = maps[e] * edge_weights[factor1_idx]
        if diagonal_maps:
            diag_idx = torch.arange(0, d)
            Delta[top_x + diag_idx, top_y + diag_idx] = orient * maps[e]
        else:
            Delta[top_x : top_x + d, top_y : top_y + d] = orient * maps[e]

    # Compute non-normalised Laplacian.
    L_dense = Delta.T @ Delta

    """if values is not None:
        # Append extra entries to the diagonal of the parallel transport maps and update the stalk dimension.
        L_dense, d = append_diag_maps_to_existent_laplacian(
            size, d, L_dense, edge_index, values
        )
"""
    if not normalised:
        return L_dense

    # Build normalised Laplacian.
    D_sqrt_inv = torch.zeros((N * d, N * d), dtype=torch.float64)
    for i in range(N):
        low = i * d
        high = low + d

        D_i = L_dense[low:high, low:high]
        D_i = D_i + torch.eye(d)
        D_i_sqrt_inv = sym_matrix_pow(D_i, -0.5)
        D_sqrt_inv[low:high, low:high] = D_i_sqrt_inv

    return D_sqrt_inv @ L_dense @ D_sqrt_inv


def append_diag_maps_to_existent_laplacian(
    size, learnable_d, L, edge_index, values
):
    extra_d = len(values)
    total_d = learnable_d + extra_d

    deg = degree(edge_index[0], num_nodes=size, dtype=L.dtype)
    values = torch.tensor(values, dtype=L.dtype)
    new_L = torch.zeros(
        (size * (extra_d + learnable_d), size * (extra_d + learnable_d)),
        dtype=L.dtype,
    )

    for idx in range(edge_index.size(1)):
        i, j = edge_index[0][idx], edge_index[1][idx]
        assert i != j

        # Add to the new Laplacian the entries of the existent Laplacian
        new_low_i, new_high_i = i * total_d, i * total_d + learnable_d
        new_low_j, new_high_j = j * total_d, j * total_d + learnable_d

        low_i, high_i = i * learnable_d, i * learnable_d + learnable_d
        low_j, high_j = j * learnable_d, j * learnable_d + learnable_d

        new_L[new_low_i:new_high_i, new_low_j:new_high_j] = L[
            low_i:high_i, low_j:high_j
        ]

        # Append extra entries to each parallel transport map
        extra_diag_idx = torch.arange(learnable_d, total_d)
        new_L[new_low_i + extra_diag_idx, new_low_j + extra_diag_idx] = values

    for i in range(size):
        # Add to the new Laplacian diagonal, the diagonal entries of the existent Laplacian
        new_low_i, new_high_i = i * total_d, i * total_d + learnable_d
        low_i, high_i = i * learnable_d, i * learnable_d + learnable_d
        new_L[new_low_i:new_high_i, new_low_i:new_high_i] = L[
            low_i:high_i, low_i:high_i
        ]

        # Append the degree on the diagonal for the extra entries
        extra_diag_idx = torch.arange(learnable_d, total_d)
        new_L[i * total_d + extra_diag_idx, i * total_d + extra_diag_idx] = deg[
            i
        ]

    return new_L, total_d


def sym_matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: a batch of matrices
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.linalg.eigh(matrix)
    vals[vals > 0] = vals[vals > 0].pow(p)
    matrix_pow = vecs @ torch.diag(vals) @ vecs.T
    return matrix_pow
