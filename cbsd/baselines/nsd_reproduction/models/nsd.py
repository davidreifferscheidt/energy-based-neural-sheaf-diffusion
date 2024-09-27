import copy
from typing import Any, Callable, Dict, Optional, Union

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import torch_sparse
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from torch_geometric.utils.sparse import dense_to_sparse, to_torch_csr_tensor
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from cbsd.baselines.nsd_reproduction.lib.laplace import dirichlet_energy
from cbsd.baselines.nsd_reproduction.models import laplacian_builders as lb
from cbsd.baselines.nsd_reproduction.models.sheaf_models import (
    EdgeWeightLearner,
    LocalConcatSheafLearner,
    LocalConcatSheafLearnerVariant,
)

"""
from cbsd.nn.builders.utils.laplace import (
    dirichlet_energy,
    sheaf_dirichlet_energy,
)"""


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
    return energy  # / (2 * norm.pow(2))


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


patch_typeguard()


@typechecked
class DiscreteDiagSheafDiffusion(nn.Module):
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
        add_lp: bool = True,
        add_hp: bool = False,
        left_weights: bool = True,
        right_weights: bool = True,
        sparse_learner: bool = False,
        linear: bool = False,
        normalised: bool = True,
        deg_normalised: bool = False,
        sheaf_act: str = "tanh",
        second_linear: bool = False,
    ):
        super().__init__()
        if d <= 1:
            raise ValueError("d must be greater than 1.")

        self.d = d
        self.add_lp = add_lp
        self.add_hp = add_hp

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act
        self.out_channels = (
            hidden_channels if out_channels is None else out_channels
        )
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.sparse_learner = sparse_learner
        self.use_act = use_act
        self.nonlinear = not linear
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.sheaf_act = sheaf_act
        self.second_linear = second_linear

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.num_layers):
                self.lin_right_weights.append(
                    nn.Linear(
                        self.hidden_channels, self.hidden_channels, bias=False
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
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.num_layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                """self.lin_left_weights.append(
                    nn.utils.parametrizations.spectral_norm(
                        nn.Linear(self.final_d, self.final_d, bias=False)
                    )
                )"""
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(
            self.num_layers, self.num_layers if self.nonlinear else 1
        )
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(
                        self.hidden_channels * self.final_d,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                    )
                )
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.num_nodes,
            edge_index,
            d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
        )

        self.epsilons = nn.ParameterList()
        for i in range(self.num_layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(
            self.in_channels, self.hidden_channels * self.final_d
        )
        if self.second_linear:
            self.lin12 = nn.Linear(
                self.hidden_channels * self.final_d,
                self.hidden_channels * self.final_d,
            )
        self.lin2 = nn.Linear(
            self.hidden_channels * self.final_d, self.out_channels
        )

        # Monitor the energies
        self.energy = []
        self.sheaf_energy = []
        self.energy_normalised = []
        self.sheaf_energy_normalised = []

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        self.starting_energy_edge_distr = dirichlet_energy_edge_distr(
            x, self.edge_index
        ).cpu()
        self.starting_energy_node_distr = dirichlet_energy_node_distr(
            x, self.edge_index
        ).cpu()
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.num_nodes * self.final_d, -1)

        x0 = x
        for layer in range(self.num_layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(
                    x,
                    p=self.dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(self.num_nodes, -1), self.edge_index
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

                if not self.training:
                    # Compute lambda max
                    lap = to_scipy_sparse_matrix(
                        L[0].detach(),
                        L[1].detach(),
                        self.num_nodes * self.final_d,
                    )
                    lambda_max = eigsh(
                        lap, k=1, which="LM", return_eigenvectors=False
                    )
                    self.sheaf_learners[layer].lambda_max = float(
                        lambda_max.real
                    )
                    # dense_L = to_dense_adj(edge_index=L[0], edge_attr=L[1])
                    s = scipy.sparse.linalg.svds(
                        lap, return_singular_vectors=False
                    )
                    # u, s, v = torch.svd(dense_L, compute_uv=False)
                    self.sheaf_learners[layer].sigma_max = s[0].item()
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.num_nodes * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            if self.right_weights:
                # Compute the spectral norm of the weight matrices
                right_weights = self.lin_right_weights[layer].weight
                ru, rs, rv = torch.svd(right_weights, compute_uv=False)
                self.lin_right_weights[layer].spectral_norm = rs[0].item()

            if self.left_weights:
                left_weights = self.lin_left_weights[layer].weight
                lu, ls, lv = torch.svd(left_weights, compute_uv=False)
                self.lin_left_weights[layer].spectral_norm = ls[0].item()

            # Sheaf Diffusion
            dense_lap = to_dense_adj(edge_index=L[0], edge_attr=L[1])
            A_norm = (
                torch.eye(
                    self.num_nodes * self.final_d, device=self.edge_index.device
                )
                - dense_lap
            )
            # Diffusion step
            sparse_A_norm = dense_to_sparse(A_norm)
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            # self.energy.append(dirichlet_energy(L, x, x.size(0)))
            """self.sheaf_energy.append(
                sheaf_dirichlet_energy(
                    to_torch_csr_tensor(L[0], L[1]), x, normalised=False
                ).item()
            )"""
            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(self.num_nodes, 1)
            x0 = coeff * x0 - x
            x = x0

        self.energy.append(dirichlet_energy(L, x, x.size(0)))
        x = x.reshape(self.num_nodes, -1)
        self.energy_edge_distr = dirichlet_energy_edge_distr(
            x, self.edge_index
        ).cpu()
        node_energy = dirichlet_energy_node_distr(x, self.edge_index)
        self.energy_node_distr = node_energy.cpu()
        # To detect the numerical instabilities of SVD.
        # x = torch.cat((x, node_energy.unsqueeze(1)), dim=1)
        assert torch.all(torch.isfinite(x))
        return self.lin2(x)


@typechecked
class DiscreteBundleSheafDiffusion(nn.Module):
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
        add_lp: bool = True,
        add_hp: bool = False,
        left_weights: bool = True,
        right_weights: bool = True,
        sparse_learner: bool = False,
        linear: bool = False,
        normalised: bool = True,
        deg_normalised: bool = False,
        sheaf_act: str = "tanh",
        second_linear: bool = False,
        orth_trans: str = "householder",
        use_edge_weights: bool = True,
    ):
        super().__init__()
        if d <= 1:
            raise ValueError("d must be greater than 1.")
        assert not deg_normalised

        self.d = d
        self.add_lp = add_lp
        self.add_hp = add_hp

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act
        self.out_channels = (
            hidden_channels if out_channels is None else out_channels
        )
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.sparse_learner = sparse_learner
        self.use_act = use_act
        self.nonlinear = not linear
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.sheaf_act = sheaf_act
        self.second_linear = second_linear
        self.orth_trans = orth_trans
        self.use_edge_weights = use_edge_weights

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.num_layers):
                """self.lin_right_weights.append(
                    nn.Linear(
                        self.hidden_channels, self.hidden_channels, bias=False
                    )
                )"""
                self.lin_right_weights.append(
                    nn.utils.parametrizations.spectral_norm(
                        nn.Linear(
                            self.hidden_channels,
                            self.hidden_channels,
                            bias=False,
                        )
                    )
                )
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.num_layers):
                """self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )"""
                self.lin_left_weights.append(
                    nn.utils.parametrizations.spectral_norm(
                        nn.Linear(self.final_d, self.final_d, bias=False)
                    )
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
                """self.lin_left_weights[-1].weight.data = (
                    self.lin_left_weights[-1].weight.data * 0.5
                )"""

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(
            self.num_layers, self.num_layers if self.nonlinear else 1
        )
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.get_param_size(),),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(
                        self.hidden_channels * self.final_d,
                        out_shape=(self.get_param_size(),),
                        sheaf_act=self.sheaf_act,
                    )
                )
            if self.use_edge_weights:
                self.weight_learners.append(
                    EdgeWeightLearner(
                        self.hidden_channels * self.final_d, edge_index
                    )
                )
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.num_nodes,
            edge_index,
            d=self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_trans,
        )

        self.epsilons = nn.ParameterList()
        for i in range(self.num_layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(
            self.in_channels, self.hidden_channels * self.final_d
        )
        if self.second_linear:
            self.lin12 = nn.Linear(
                self.hidden_channels * self.final_d,
                self.hidden_channels * self.final_d,
            )
        self.lin2 = nn.Linear(
            self.hidden_channels * self.final_d, self.out_channels
        )

        # Monitor the energies
        self.energy = []
        self.sheaf_energy = []
        self.energy_normalised = []
        self.sheaf_energy_normalised = []

    def get_param_size(self):
        if self.orth_trans in ["matrix_exp", "cayley"]:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.num_nodes * self.final_d, -1)

        x0 = x
        for layer in range(self.num_layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(
                    x,
                    p=self.dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(self.num_nodes, -1), self.edge_index
                )
                edge_weights = (
                    self.weight_learners[layer](
                        x_maps.reshape(self.num_nodes, -1), self.edge_index
                    )
                    if self.use_edge_weights
                    else None
                )
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

                if not self.training:
                    # Compute lambda max
                    lap = to_scipy_sparse_matrix(
                        L[0].detach(),
                        L[1].detach(),
                        self.num_nodes * self.final_d,
                    )
                    lambda_max = eigsh(
                        lap, k=1, which="LM", return_eigenvectors=False
                    )
                    self.sheaf_learners[layer].lambda_max = float(
                        lambda_max.real
                    )
                    # dense_L = to_dense_adj(edge_index=L[0], edge_attr=L[1])
                    s = scipy.sparse.linalg.svds(
                        lap, return_singular_vectors=False
                    )
                    # u, s, v = torch.svd(dense_L, compute_uv=False)
                    self.sheaf_learners[layer].sigma_max = s[0].item()
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.num_nodes * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            if self.right_weights:
                # Compute the spectral norm of the weight matrices
                right_weights = self.lin_right_weights[layer].weight
                ru, rs, rv = torch.svd(right_weights, compute_uv=False)
                self.lin_right_weights[layer].spectral_norm = rs[0].item()

            if self.left_weights:
                left_weights = self.lin_left_weights[layer].weight
                lu, ls, lv = torch.svd(left_weights, compute_uv=False)
                self.lin_left_weights[layer].spectral_norm = ls[0].item()

            # Sheaf Diffusion
            dense_lap = to_dense_adj(edge_index=L[0], edge_attr=L[1])
            A_norm = (
                torch.eye(
                    self.num_nodes * self.final_d, device=self.edge_index.device
                )
                - dense_lap
            )
            sparse_A_norm = dense_to_sparse(A_norm)
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            self.energy.append(dirichlet_energy(L, x, x.size(0)))
            """self.sheaf_energy.append(
                sheaf_dirichlet_energy(
                    to_torch_csr_tensor(L[0], L[1]), x, normalised=False
                ).item()
            )"""
            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(self.num_nodes, 1)
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.num_nodes, -1)
        x = self.lin2(x)
        return x  # F.log_softmax(x, dim=1)


@typechecked
class DiscreteGeneralSheafDiffusion(nn.Module):
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
        add_lp: bool = True,
        add_hp: bool = False,
        left_weights: bool = True,
        right_weights: bool = True,
        sparse_learner: bool = False,
        linear: bool = False,
        normalised: bool = True,
        deg_normalised: bool = False,
        sheaf_act: str = "tanh",
        second_linear: bool = False,
    ):
        super().__init__()
        if d <= 1:
            raise ValueError("d must be greater than 1.")

        self.d = d
        self.add_lp = add_lp
        self.add_hp = add_hp

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act
        self.out_channels = (
            hidden_channels if out_channels is None else out_channels
        )
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.sparse_learner = sparse_learner
        self.use_act = use_act
        self.nonlinear = not linear
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.sheaf_act = sheaf_act
        self.second_linear = second_linear

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.num_layers):
                self.lin_right_weights.append(
                    nn.Linear(
                        self.hidden_channels, self.hidden_channels, bias=False
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
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.num_layers):
                """self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )"""
                self.lin_left_weights.append(
                    nn.utils.parametrizations.spectral_norm(
                        nn.Linear(self.final_d, self.final_d, bias=False)
                    )
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(
            self.num_layers, self.num_layers if self.nonlinear else 1
        )
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(
                        self.hidden_channels * self.final_d,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                    )
                )
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.num_nodes,
            edge_index,
            d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
        )

        self.epsilons = nn.ParameterList()
        for i in range(self.num_layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(
            self.in_channels, self.hidden_channels * self.final_d
        )
        if self.second_linear:
            self.lin12 = nn.Linear(
                self.hidden_channels * self.final_d,
                self.hidden_channels * self.final_d,
            )
        self.lin2 = nn.Linear(
            self.hidden_channels * self.final_d, self.out_channels
        )
        # Monitor the energies
        self.energy = []
        self.sheaf_energy = []
        self.energy_normalised = []
        self.sheaf_energy_normalised = []

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.num_nodes * self.final_d, -1)

        x0 = x
        for layer in range(self.num_layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(
                    x,
                    p=self.dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(self.num_nodes, -1), self.edge_index
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

                if not self.training:
                    # Compute lambda max
                    lap = to_scipy_sparse_matrix(
                        L[0].detach(),
                        L[1].detach(),
                        self.num_nodes * self.final_d,
                    )
                    lambda_max = eigsh(
                        lap, k=1, which="LM", return_eigenvectors=False
                    )
                    self.sheaf_learners[layer].lambda_max = float(
                        lambda_max.real
                    )
                    # dense_L = to_dense_adj(edge_index=L[0], edge_attr=L[1])
                    s = scipy.sparse.linalg.svds(
                        lap, return_singular_vectors=False
                    )
                    # u, s, v = torch.svd(dense_L, compute_uv=False)
                    self.sheaf_learners[layer].sigma_max = s[0].item()
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.num_nodes * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            if self.right_weights:
                # Compute the spectral norm of the weight matrices
                right_weights = self.lin_right_weights[layer].weight
                ru, rs, rv = torch.svd(right_weights, compute_uv=False)
                self.lin_right_weights[layer].spectral_norm = rs[0].item()

            if self.left_weights:
                left_weights = self.lin_left_weights[layer].weight
                lu, ls, lv = torch.svd(left_weights, compute_uv=False)
                self.lin_left_weights[layer].spectral_norm = ls[0].item()

            # Sheaf Diffusion
            dense_lap = to_dense_adj(edge_index=L[0], edge_attr=L[1])
            A_norm = (
                torch.eye(
                    self.num_nodes * self.final_d, device=self.edge_index.device
                )
                - dense_lap
            )
            sparse_A_norm = dense_to_sparse(A_norm)
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            self.energy.append(
                dirichlet_energy(x, self.edge_index, normalised=False).item()
            )
            """self.sheaf_energy.append(
                sheaf_dirichlet_energy(
                    to_torch_csr_tensor(L[0], L[1]), x, normalised=False
                ).item()
            )"""
            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(self.num_nodes, 1)
            x0 = coeff * x0 - x
            # x = x0

        x = x.reshape(self.num_nodes, -1)
        x = self.lin2(x)
        return x  # F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusionEnergy(nn.Module):
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
        add_lp: bool = True,
        add_hp: bool = False,
        left_weights: bool = True,
        right_weights: bool = True,
        sparse_learner: bool = False,
        linear: bool = False,
        normalised: bool = True,
        deg_normalised: bool = False,
        sheaf_act: str = "tanh",
        second_linear: bool = False,
        normalized_energy: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        if d <= 1:
            raise ValueError("d must be greater than 1.")

        self.d = d
        self.add_lp = add_lp
        self.add_hp = add_hp

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act
        self.out_channels = (
            hidden_channels if out_channels is None else out_channels
        )
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.sparse_learner = sparse_learner
        self.use_act = use_act
        self.nonlinear = not linear
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.sheaf_act = sheaf_act
        self.second_linear = second_linear
        self.normalized_energy = normalized_energy
        self.residual = residual

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.num_layers):
                self.lin_right_weights.append(
                    nn.Linear(
                        self.hidden_channels, self.hidden_channels, bias=False
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
                # nn.init.eye_(self.lin_right_weights[-1].weight.data)
                # self.lin_right_weights[-1].weight.data = (
                #     self.lin_right_weights[-1].weight.data * 0.5
                # )
        if self.left_weights:
            for i in range(self.num_layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                # self.lin_left_weights.append(
                #     nn.utils.parametrizations.spectral_norm(
                #         nn.Linear(self.final_d, self.final_d, bias=False)
                #     )
                # )
                # nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(
            self.num_layers, self.num_layers if self.nonlinear else 1
        )
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(
                        self.hidden_channels * self.final_d,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                    )
                )
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.num_nodes,
            edge_index,
            d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
        )

        self.epsilons = nn.ParameterList()
        for i in range(self.num_layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(
            self.in_channels, self.hidden_channels * self.final_d
        )
        if self.second_linear:
            self.lin12 = nn.Linear(
                self.hidden_channels * self.final_d,
                self.hidden_channels * self.final_d,
            )
        self.lin2 = nn.Linear(
            self.hidden_channels * self.final_d, self.out_channels
        )

        self.laplacian_sym = pyg_utils.get_laplacian(
            edge_index=self.edge_index, normalization="sym"
        )
        self.laplacian = pyg_utils.get_laplacian(edge_index=self.edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.num_nodes * self.final_d, -1)

        sheaf_energies = []
        energies = []

        x0 = x
        for layer in range(self.num_layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(
                    x,
                    p=self.dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(self.num_nodes, -1), self.edge_index
                )
                L, trans_maps = self.laplacian_builder(maps)

                sheaf_energies.append(
                    dirichlet_energy(
                        L, x, x.size(0), normalized=self.normalized_energy
                    )
                )
                energies.append(
                    dirichlet_energy(
                        self.laplacian,
                        x.view(self.num_nodes, -1),
                        self.num_nodes,
                        normalized=self.normalized_energy,
                    )
                )
                self.sheaf_learners[layer].set_L(trans_maps)
                # Compute lambda max
                if not self.training:
                    lap = to_scipy_sparse_matrix(
                        L[0].detach(),
                        L[1].detach(),
                        self.num_nodes * self.final_d,
                    )
                    lambda_max = eigsh(
                        lap, k=1, which="LM", return_eigenvectors=False
                    )
                    self.sheaf_learners[layer].lambda_max = float(
                        lambda_max.real
                    )
                    print(self.sheaf_learners[layer].lambda_max)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.num_nodes * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            # Sheaf Diffusion
            dense_lap = to_dense_adj(edge_index=L[0], edge_attr=L[1])
            A_norm = (
                torch.eye(
                    self.num_nodes * self.final_d, device=self.edge_index.device
                )
                - dense_lap
            )
            # Diffusion step
            sparse_A_norm = dense_to_sparse(A_norm)

            # x = torch_sparse.spmm(
            #     sparse_A_norm[0], sparse_A_norm[1], x.size(0), x.size(0), x
            # )
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            if self.residual:
                coeff = 1 + torch.tanh(self.epsilons[layer]).tile(
                    self.num_nodes, 1
                )
                x0 = coeff * x0 - x
                x = x0
                
            sheaf_energies.append(
                dirichlet_energy(
                    L, x, x.size(0), normalized=self.normalized_energy
                )
            )
            energies.append(
                dirichlet_energy(
                    self.laplacian,
                    x.view(self.num_nodes, -1),
                    self.num_nodes,
                    normalized=self.normalized_energy,
                )
            )

        x = x.reshape(self.num_nodes, -1)
        x = self.lin2(x)
        return sheaf_energies, energies  # F.log_softmax(x, dim=1)
