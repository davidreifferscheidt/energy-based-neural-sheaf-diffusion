import copy
import math
from typing import Any, Callable, Dict, Optional, Union

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import torch_sparse
from torchtyping import TensorType

"""def dirichlet_energy(x, edge_index):
    edge_difference = (
        torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1) ** 2
    )
    return edge_difference.sum() / 2"""


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


def dirichlet_energy(L, f):
    """Returns the Dirichlet energy of the signal f under the (normalized) Laplacian L."""
    right = pyg_utils.spmm(L, f)
    energy = torch.trace(f.t() @ right)
    norm = torch.norm(f)
    return energy.item() / (norm.pow(2))


class EnergyACM(nn.Module):
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
        super(EnergyACM, self).__init__()

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

        device = edge_index.device

        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.att = 0
        self.att_vec = nn.ParameterList()
        for i in range(2 * num_layers):
            self.att_vec.append(
                nn.Parameter(
                    torch.FloatTensor(1 * hidden_channels, 1).to(device)
                )
            )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            nn.Parameter(torch.FloatTensor(1 * hidden_channels, 1).to(device)),
            nn.Parameter(torch.FloatTensor(1 * hidden_channels, 1).to(device)),
            nn.Parameter(torch.FloatTensor(1 * hidden_channels, 1).to(device)),
        )
        self.att_mix = nn.Parameter(
            torch.FloatTensor(2 * self.num_layers, 2 * self.num_layers).to(
                device
            )
        )
        #############################################################
        # Save diffusion matrices
        # high-pass filter diffusion matrices
        laplacian = pyg_utils.get_laplacian(edge_index=self.edge_index)
        self.laplacian = pyg_utils.to_torch_coo_tensor(
            edge_index=laplacian[0], edge_attr=laplacian[1], size=self.num_nodes
        )
        laplacian_rw = pyg_utils.get_laplacian(
            edge_index=self.edge_index, normalization="rw"
        )
        self.laplacian_rw = pyg_utils.to_torch_coo_tensor(
            edge_index=laplacian_rw[0],
            edge_attr=laplacian_rw[1],
            size=self.num_nodes,
        )
        laplacian_sym = pyg_utils.get_laplacian(
            edge_index=self.edge_index, normalization="sym"
        )
        self.laplacian_sym = pyg_utils.to_torch_coo_tensor(
            edge_index=laplacian_sym[0],
            edge_attr=laplacian_sym[1],
            size=self.num_nodes,
        )
        # low-pass filter diffusion matrices
        self.adjacency = pyg_utils.to_torch_coo_tensor(
            edge_index=self.edge_index, size=self.num_nodes
        )

        dense_rw = pyg_utils.to_dense_adj(
            edge_index=laplacian_rw[0], edge_attr=laplacian_rw[1]
        )
        adjacency_rw = pyg_utils.dense_to_sparse(
            torch.eye(self.num_nodes, device=edge_index.device) - dense_rw
        )
        self.adjacency_rw = pyg_utils.to_torch_coo_tensor(
            edge_index=adjacency_rw[0],
            edge_attr=adjacency_rw[1],
            size=self.num_nodes,
        )
        dense_sym = pyg_utils.to_dense_adj(
            edge_index=laplacian_sym[0], edge_attr=laplacian_sym[1]
        )
        adjacency_sym = pyg_utils.dense_to_sparse(
            torch.eye(self.num_nodes, device=edge_index.device) - dense_sym
        )
        self.adjacency_sym = pyg_utils.to_torch_coo_tensor(
            edge_index=adjacency_sym[0],
            edge_attr=adjacency_sym[1],
            size=self.num_nodes,
        )
        ##################################################################
        # Linear layers to map into latent space
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        self.lin_right = nn.Linear(
            self.hidden_channels,
            self.hidden_channels,
            bias=True,
        )

        # weight matrices applied from the right to adjust in latent space
        self.linear_right_weights = nn.ParameterList()
        self.linear_right_weights2 = nn.ParameterList()
        for l in range(self.num_layers):
            self.linear_right_weights.append(
                nn.Parameter(
                    torch.FloatTensor(
                        self.hidden_channels, self.hidden_channels
                    ).to(device)
                )
            )
            self.linear_right_weights2.append(
                nn.Parameter(
                    torch.FloatTensor(
                        self.hidden_channels, self.hidden_channels
                    ).to(device)
                )
            )
        """self.linear_right_weights = nn.ModuleList()
        for l in range(self.num_layers):
            self.linear_right_weights.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )"""
        # learnable paramter to adjust magnitude of residual connection
        """self.epsilons = nn.ParameterList()
        self.gammas = nn.ParameterList()
        self.betas = nn.ParameterList()
        for _ in range(self.num_layers):
            self.epsilons.append(nn.Parameter(torch.zeros((1, 1))))
            self.gammas.append(nn.Parameter(torch.tensor([-2.0])))
            self.betas.append(nn.Parameter(torch.tensor([1.0])))"""

        self.energy = []
        self.sheaf_energy = []
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_channels)
        std_att = 1.0 / math.sqrt(self.hidden_channels)
        std_att_mix = 1.0 / math.sqrt(self.num_layers)

        for i in range(self.num_layers):
            self.att_vec[i].data.uniform_(-std_att, std_att)
            self.linear_right_weights[i].data.uniform_(-stdv, stdv)
            self.linear_right_weights2[i].data.uniform_(-stdv, stdv)

        self.att_mix.data.uniform_(-std_att_mix, std_att_mix)

    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        """if self.model_type == "acmgcn+" or self.model_type == "acmgcn++":
            output_low, output_high, output_mlp = (
                self.layer_norm_low(output_low),
                self.layer_norm_high(output_high),
                self.layer_norm_mlp(output_mlp),
            )"""
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        dim=1,
                    )
                ),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def attention(self, output_list):
        T = len(output_list)
        vec_list = []
        for i in range(T):
            vec_list.append(torch.mm(output_list[i], self.att_vec[i]))
        logits = (
            torch.mm(torch.sigmoid(torch.cat(vec_list, dim=1)), self.att_mix)
            / T
        )
        att = torch.softmax(logits, dim=1)
        return att
        # return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

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
        num_nodes = self.num_nodes

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        # maps x to latent space
        x = self.lin1(x)
        self.starting_energy_edge_distr = dirichlet_energy_edge_distr(
            x, edge_index
        ).cpu()
        self.starting_energy_node_distr = dirichlet_energy_node_distr(
            x, edge_index
        ).cpu()
        self.energy.append(dirichlet_energy(self.laplacian, x))
        if self.use_act:
            x = F.relu(x)

        output = []
        for layer in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.right_weights:
                out1 = torch.mm(
                    x, self.linear_right_weights[layer]
                )  # self.linear_right_weights[layer](x)
                out2 = torch.mm(x, self.linear_right_weights2[layer])
            if layer != 0:
                out1 = pyg_utils.spmm(self.laplacian_rw.pow(layer), out1)
                out2 = pyg_utils.spmm(self.adjacency_rw.pow(layer), out2)
            if layer != self.num_layers - 1:
                out1 = pyg_utils.spmm(self.adjacency_rw, out1)
                out2 = pyg_utils.spmm(self.laplacian_rw, out2)
            out1 = F.relu(out1)
            out2 = F.relu(out2)
            output.append(out1)
            output.append(out2)

        self.att = self.attention(output)
        """sum = 0
        for i in range(len(output)):
            sum += self.att[:, i][:, None] * output[i]
        x = 1 / len(output) * sum"""

        x = 1 / len(output) * sum(output)

        self.energy_edge_distr = dirichlet_energy_edge_distr(
            x, edge_index
        ).cpu()
        node_energy = dirichlet_energy_node_distr(x, edge_index)
        self.energy_node_distr = node_energy.cpu()
        # To detect the numerical instabilities of SVD.
        # x = torch.cat((x, node_energy.unsqueeze(1)), dim=1)
        assert torch.all(torch.isfinite(x))
        return self.lin2(x)
