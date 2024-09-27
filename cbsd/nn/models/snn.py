import copy
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import torch_sparse
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import spmm
from torch_geometric.utils.sparse import is_sparse
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from cbsd.nn.builders.utils.laplace import (
    dirichlet_energy,
    sheaf_dirichlet_energy,
)

patch_typeguard()


@typechecked
class SNN(nn.Module):
    r"""The Sheaf Neural Network from the `"Neural Sheaf Diffusion: A Topological
    Perspective on Heterophily and Oversmoothing in GNNs"
    <https://openreview.net/forum?id=HtLzqEb1aec>`_ paper.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        d: int,
        num_layers: int,
        num_nodes: int,
        edge_index: Optional[TensorType[torch.long, 2, "num_edges"]] = None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        use_act: bool = True,
        add_lp: bool = False,
        add_hp: bool = False,
        left_weights: bool = True,
        right_weights: bool = True,
    ):
        super().__init__()

        if d <= 1:
            raise ValueError("d must be greater than 1.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.d = d
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act
        self.add_lp = add_lp
        self.add_hp = add_hp
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.out_channels = (
            hidden_channels if out_channels is None else out_channels
        )

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        self.hidden_channels = int(self.in_channels / self.final_d)

        for _ in range(self.num_layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            # nn.init.orthogonal_(
            #    self.lin_right_weights[-1].weight.data
            # )

        for _ in range(self.num_layers):
            self.lin_left_weights.append(
                nn.Linear(self.final_d, self.final_d, bias=False)
            )
            # nn.init.eye_(
            #    self.lin_left_weights[-1].weight.data
            # )  # ? check if/why this is necessary

        self.epsilons = nn.ParameterList()
        for _ in range(self.num_layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(
            self.in_channels, self.hidden_channels * self.final_d
        )
        self.lin2 = nn.Linear(
            self.hidden_channels * self.final_d, self.out_channels
        )

        # Monitor the energies
        self.energy = []
        self.sheaf_energy = []
        self.energy_normalised = []
        self.sheaf_energy_normalised = []

    def forward(
        self,
        x: TensorType[torch.float, "num_nodes", "in_channels"],
        sheaf_laplacian: torch.Tensor,  # sparse tensor
    ) -> TensorType["num_nodes", "out_channels"]:
        num_nodes = self.num_nodes
        edge_index = self.edge_index
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(num_nodes * self.final_d, -1)

        self.energy.append(
            dirichlet_energy(x, edge_index, normalised=False).item()
        )
        self.sheaf_energy.append(
            sheaf_dirichlet_energy(sheaf_laplacian, x, normalised=False).item()
        )
        """
        self.energy_normalised.append(
            dirichlet_energy(x, edge_index, normalised=True).item()
        )
        self.sheaf_energy_normalised.append(
            sheaf_dirichlet_energy(sheaf_laplacian, x, normalised=True).item()
        )"""

        x0 = x
        for layer in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, num_nodes * self.final_d).t()
                # Compute spectral norm of weight matrices
                left_weights = self.lin_left_weights[layer].weight
                lu, ls, lv = torch.svd(left_weights, compute_uv=False)
                self.lin_left_weights[layer].spectral_norm = ls[0].item()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)
                # Compute the spectral norm of the weight matrices
                right_weights = self.lin_right_weights[layer].weight
                ru, rs, rv = torch.svd(right_weights, compute_uv=False)
                self.lin_right_weights[layer].spectral_norm = rs[0].item()

            L = pyg_utils.to_edge_index(sheaf_laplacian)
            if not self.training:
                lap = pyg_utils.to_scipy_sparse_matrix(
                    L[0].detach(),
                    L[1].detach(),
                    self.num_nodes * self.final_d,
                )
                lambda_max = eigsh(
                    lap, k=1, which="LM", return_eigenvectors=False
                )
                self.lambda_max = float(lambda_max.real)
                print(self.lambda_max)
            # Sheaf Diffusion
            dense_lap = pyg_utils.to_dense_adj(edge_index=L[0], edge_attr=L[1])
            A_norm = (
                torch.eye(
                    self.num_nodes * self.final_d, device=self.edge_index.device
                )
                - dense_lap
            )
            # Diffusion step
            sparse_A_norm = pyg_utils.dense_to_sparse(A_norm)

            # x = torch_sparse.spmm(
            #     sparse_A_norm[0], sparse_A_norm[1], x.size(0), x.size(0), x
            # )
            x = spmm(sheaf_laplacian, x)
            # x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            # if is_sparse(sheaf_laplacian):
            #     x = spmm(sheaf_laplacian, x)
            # # x = torch_sparse.spmm(
            # #    sheaf_laplacian[0], sheaf_laplacian[1], x.size(0), x.size(0), x
            # # )

            # else:
            #     x = sheaf_laplacian.float() @ x.float()

            # self.energy.append(
            #    dirichlet_energy(sheaf_laplacian, x, normalised=True)
            # )
            if self.use_act:
                x = F.elu(x)

            self.energy.append(
                dirichlet_energy(x, edge_index, normalised=False).item()
            )
            self.sheaf_energy.append(
                sheaf_dirichlet_energy(
                    sheaf_laplacian, x, normalised=False
                ).item()
            )
            """
            self.energy_normalised.append(
                dirichlet_energy(x, edge_index, normalised=True).item()
            )
            self.sheaf_energy_normalised.append(
                sheaf_dirichlet_energy(
                    sheaf_laplacian, x, normalised=True
                ).item()
            )"""
            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(num_nodes, 1)
            x0 = coeff * x0 - x
            x = x0
        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(num_nodes, -1)
        return self.lin2(x)
