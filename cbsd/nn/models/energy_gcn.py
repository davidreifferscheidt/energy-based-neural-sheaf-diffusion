import copy
from typing import Any, Callable, Dict, Optional, Union

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import torch_sparse
from torchtyping import TensorType


def dirichlet_energy_edge_distr(x, edge_index, edge_attr=None):
    edge_index, edge_attr = pyg_utils.remove_self_loops(edge_index, edge_attr)

    # Compute the squared differences in node values and divide by the degrees
    degree = pyg_utils.degree(edge_index[0], num_nodes=x.size(0))
    diff = torch.mul(
        1 / torch.sqrt(degree[edge_index[0]].unsqueeze(1)), x[edge_index[0]]
    )
    diff -= torch.mul(
        1 / torch.sqrt(degree[edge_index[1]].unsqueeze(1)), x[edge_index[1]]
    )

    energy = torch.norm(diff, dim=1).pow(2)

    # Multiply by the edge attributes
    if edge_attr == None:
        edge_attr = 1
    energy = edge_attr * energy
    norm = torch.norm(x)
    return energy / (2 * norm.pow(2))


def dirichlet_energy_node_distr(x, edge_index, edge_attr=None):
    edge_index, edge_attr = pyg_utils.remove_self_loops(edge_index, edge_attr)

    degree = pyg_utils.degree(edge_index[0], num_nodes=x.size(0))
    diff_matrix = torch.mul(
        1 / torch.sqrt(degree[edge_index[0]].unsqueeze(1)), x[edge_index[0]]
    )
    diff_matrix -= torch.mul(
        1 / torch.sqrt(degree[edge_index[1]].unsqueeze(1)), x[edge_index[1]]
    )
    dirichlet_energy = torch.norm(diff_matrix, dim=1).pow(2)

    # Scatter the edge energies to the nodes
    dirichlet_energy_per_node = torch.zeros_like(x[:, 0])
    dirichlet_energy_per_node.scatter_add_(0, edge_index[0], dirichlet_energy)
    dirichlet_energy_per_node.scatter_add_(0, edge_index[1], dirichlet_energy)

    # dirichlet_energy_per_node now contains the Dirichlet energy for each node
    norm = torch.norm(x)
    return dirichlet_energy_per_node / (2 * norm.pow(2))


def dirichlet_energy(L, f, size, normalized=False):
    """Returns the Dirichlet energy of the signal f under the (normalized) Laplacian L."""
    right = torch_sparse.spmm(L[0], L[1], size, size, f)
    energy = torch.trace(f.t() @ right)
    if normalized:
        norm = torch.norm(f)
        return energy.item() / (norm.pow(2))
    else:
        return energy.item()


class EnergyGCN(nn.Module):
    # Graph Convolution Network: as proposed in [Kipf et al. 2017](https://arxiv.org/abs/1609.02907).
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        edge_index: [TensorType[torch.long, 2, "num_edges"]],
        num_nodes: int,
        out_channels: Optional[int] = None,
        dropout: Optional[float] = 0.0,
        input_dropout: Optional[float] = 0.0,
        right_weights: Optional[bool] = True,
        use_act: Optional[bool] = True,
        residual: Optional[bool] = True,
    ):
        super(EnergyGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act
        self.right_weights = right_weights
        self.residual = residual
        self.edge_index = pyg_utils.add_self_loops(edge_index=edge_index)[0]
        self.num_nodes = num_nodes

        # Save diffusion matrices
        self.laplacian = pyg_utils.get_laplacian(edge_index=self.edge_index)
        self.laplacian_rw = pyg_utils.get_laplacian(
            edge_index=self.edge_index, normalization="rw"
        )
        self.laplacian_sym = pyg_utils.get_laplacian(
            edge_index=self.edge_index, normalization="sym"
        )

        edge_index, edge_attr = self.laplacian_rw
        dense_rw = pyg_utils.to_dense_adj(
            edge_index=self.laplacian_rw[0], edge_attr=self.laplacian_rw[1]
        )
        self.adjacency_rw = pyg_utils.dense_to_sparse(
            torch.eye(self.num_nodes, device=edge_index.device) - dense_rw
        )
        dense_sym = pyg_utils.to_dense_adj(
            edge_index=self.laplacian_sym[0], edge_attr=self.laplacian_sym[1]
        )
        self.adjacency_sym = pyg_utils.dense_to_sparse(
            torch.eye(self.num_nodes, device=edge_index.device) - dense_sym
        )

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        self.lin_left_weights = nn.ModuleList()
        self.lin_right_weights = nn.ModuleList()

        if self.right_weights:
            for l in range(self.num_layers):
                self.lin_right_weights.append(
                    nn.Linear(
                        self.hidden_channels,
                        self.hidden_channels,
                        bias=False,
                    )
                )
                """self.lin_right_weights.append(
                    nn.utils.parametrizations.spectral_norm(
                        nn.Linear(
                            self.hidden_channels,
                            self.hidden_channels,
                            bias=False,
                        )
                    )
                )"""
                # To ensure initial spectral norm of c (>1)
                # nn.init.eye_(self.lin_right_weights[-1].weight.data)
                """
                self.lin_right_weights[-1].weight.data = (
                    self.lin_right_weights[-1].weight.data * 3
                )"""

        self.epsilons = nn.ParameterList()
        self.gammas = nn.ParameterList()
        for _ in range(self.num_layers):
            self.epsilons.append(nn.Parameter(torch.zeros((1, 1))))
            self.gammas.append(nn.Parameter(torch.tensor([2.0])))

        self.energy = []
        self.sheaf_energy = []

    def forward(self, x):
        """
        Forward method.

        Parameters
        ----------
        data: data
        dataset from pyg containing feature matrix x and edge_index

        Returns
        ---------
        logits: torch.tensor
            The result of the last message passing step (i.e. the logits)
        """
        edge_index = self.edge_index
        num_nodes = x.size(0)
        self.laplacian = pyg_utils.get_laplacian(
            edge_index, normalization="sym"
        )
        lap = pyg_utils.to_scipy_sparse_matrix(
            self.laplacian[0].detach(),
            self.laplacian[1].detach() * torch.sigmoid(self.gammas[0]).detach(),
            num_nodes,
        )
        lambda_max = scipy.sparse.linalg.eigsh(
            lap, k=1, which="LM", return_eigenvectors=False
        )
        self.lambda_max = float(lambda_max.real)
        # dense_L = to_dense_adj(edge_index=L[0], edge_attr=L[1])
        s = scipy.sparse.linalg.svds(lap, return_singular_vectors=False)
        self.sigma_max = s[0].item()

        dense_lap = pyg_utils.to_dense_adj(
            edge_index=self.laplacian[0], edge_attr=self.laplacian[1]
        )
        # A_norm = I - L_norm.
        A_norm = torch.eye(num_nodes, device=edge_index.device) - dense_lap
        IplusL = torch.eye(num_nodes, device=edge_index.device) + dense_lap
        sparse_A_norm = pyg_utils.dense_to_sparse(A_norm)
        sparse_IplusL = pyg_utils.dense_to_sparse(IplusL)

        """assert self.laplacian == pyg_utils.get_laplacian(
            edge_index=edge_index, edge_weight=edge_weight, normalization=None
        )"""

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        self.starting_energy_edge_distr = dirichlet_energy_edge_distr(
            x, edge_index
        ).cpu()
        self.starting_energy_node_distr = dirichlet_energy_node_distr(
            x, edge_index
        ).cpu()
        if self.use_act:
            x = F.relu(x)

        self.energy.append(dirichlet_energy(self.laplacian, x, num_nodes))
        x0 = x
        for layer in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.right_weights:
                x = self.lin_right_weights[layer](x)
                # Compute the spectral norm of the weight matrices
                right_weights = self.lin_right_weights[layer].weight
                ru, rs, rv = torch.svd(right_weights, compute_uv=False)
                self.lin_right_weights[layer].spectral_norm = rs[0].item()

            # apply diffusion matrix
            x = torch_sparse.spmm(
                sparse_IplusL[0],
                sparse_IplusL[1] * 3,  # * torch.sigmoid(self.gammas[layer]),
                x.size(0),
                x.size(0),
                x,
            )

            self.energy.append(dirichlet_energy(self.laplacian, x, num_nodes))
            """
            self.energy_normalised.append(
                dirichlet_energy(x, edge_index, normalised=True).item()
            )
            self.sheaf_energy_normalised.append(
                sheaf_dirichlet_energy(
                    sheaf_laplacian, x, normalised=True
                ).item()
            )"""
            if self.residual:
                coeff = 1 + torch.tanh(self.epsilons[layer]).tile(num_nodes, 1)
                x0 = coeff * x0 + x
                x = x0
            if self.use_act:
                x = F.relu(x)
        self.energy_edge_distr = dirichlet_energy_edge_distr(
            x, edge_index
        ).cpu()
        node_energy = dirichlet_energy_node_distr(x, edge_index)
        self.energy_node_distr = node_energy.cpu()
        # To detect the numerical instabilities of SVD.
        # x = torch.cat((x, node_energy.unsqueeze(1)), dim=1)
        assert torch.all(torch.isfinite(x))
        return self.lin2(x)
