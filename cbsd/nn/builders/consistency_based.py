import itertools
from typing import Optional

import torch
import torch.nn as nn
import torch_sparse
import tqdm
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from torch_geometric.utils import degree, is_undirected, to_torch_csr_tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .utils.laplace import (
    batched_sym_matrix_pow,
    build_dense_laplacian,
    compute_fixed_diag_laplacian_indices,
    compute_learnable_diag_laplacian_indices,
    compute_learnable_laplacian_indices,
    compute_left_right_map_index,
    get_edge_index_dict,
    mergesp,
)

patch_typeguard()


@typechecked
class ConsistencyBasedLaplacianBuilder(nn.Module):
    def __init__(self, edge_index, config: DictConfig) -> None:
        super().__init__()

        if not hasattr(config, "d"):
            raise ValueError(
                "The config must contain a sheaf_laplacian_dim attribute."
            )
        self.device = edge_index.device
        self.d = config.d
        self.normalised = config.normalised
        self.deg_normalised = config.deg_normalised
        self.add_lp = config.add_lp
        self.add_hp = config.add_hp
        self.augmented = config.augmented
        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.edge_index = edge_index
        self.num_edges = self.edge_index.shape[1]
        self.num_nodes = maybe_num_nodes(edge_index)

        self._init_maps(config.init)

    def _init_maps(
        self,
        init: str = "random",
    ) -> None:
        size = (self.num_edges, self.d, self.d)
        # TODO: init with ones causes that gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
        if init.lower() == "gaussian":
            tensor = torch.randn(size, dtype=torch.float32, device=self.device)
        elif init.lower() == "random":
            tensor = torch.rand(size, dtype=torch.float32, device=self.device)
        else:
            raise NotImplementedError(f"Init {init} not implemented.")
        self.restriction_maps = nn.Parameter(tensor, requires_grad=True)

    def dim_reduction(
        self,
        x: TensorType[torch.float, "num_nodes", "num_features"],
        d: int = 2,
        num_channels: int = 32,
    ) -> TensorType[torch.float, "num_nodes", "reduced_features"]:
        # TODO: implement multiple dimensionality reduction methods
        # Apply PCA to x
        pca = PCA(n_components=d * num_channels)
        x = pca.fit_transform(x.detach().numpy())
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def dirichlet_energy(
        self,
        x: TensorType[torch.float, "num_nodes", "d", "num_features"],
        normalize: bool = False,
    ) -> TensorType[torch.float, ()]:
        loss = 0.0
        num_edges = self.edge_index.shape[1]

        directed_edge_index_dict = get_edge_index_dict(
            self.edge_index, undirected=False
        )
        for edge in range(num_edges):
            source = self.edge_index[0, edge].item()  # u
            target = self.edge_index[1, edge].item()  # v
            assert edge == directed_edge_index_dict[(source, target)]
            Fuv = self.restriction_maps[edge]
            Fvu = self.restriction_maps[
                directed_edge_index_dict[(target, source)]
            ]

            # TODO: normalization
            # [x] FIX: save sheaf degree matrix D in self instead of compute at every energy call
            # [x] TODO: check eps -> initialization with ones?
            if normalize:
                row, _ = self.edge_index
                # Compute the diagonal block entries of the sheaf laplacian $\mathbf{L}_{\mathcal{F}_{v, v}}=\sum_{v \unlhd e} \mathcal{F}_{v \unlhd e}^{\top} \mathcal{F}_{v \unlhd e}$
                diag_maps = torch.bmm(
                    torch.transpose(self.restriction_maps, dim0=-1, dim1=-2),
                    self.restriction_maps,
                )
                diag_maps = scatter_add(
                    diag_maps, row, dim=0, dim_size=self.num_nodes
                )
                if self.training:
                    # During training, we perturb the matrices to ensure they have different singular values.
                    # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
                    eps = (
                        torch.FloatTensor(self.d)
                        .uniform_(-0.001, 0.001)
                        .to(device=self.device)
                    )
                else:
                    eps = torch.zeros(self.d, device=self.device)

                to_be_inv_diag_maps = (
                    diag_maps + torch.diag(1.0 + eps).unsqueeze(0)
                    if self.augmented
                    else diag_maps
                )
                d_sqrt_inv = batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)
                assert torch.all(torch.isfinite(d_sqrt_inv))
                # diag_maps = (d_sqrt_inv @ diag_maps @ d_sqrt_inv).clamp(min=-1, max=1)
                Du_sqrt_inv = d_sqrt_inv[source]
                Dv_sqrt_inv = d_sqrt_inv[target]
            else:
                Du_sqrt_inv = Dv_sqrt_inv = torch.eye(self.d, device=x.device)

            term1 = torch.matmul(Fvu, Dv_sqrt_inv @ x[target])
            term2 = torch.matmul(Fuv, Du_sqrt_inv @ x[source])
            loss += torch.norm(term1 - term2, p=2) ** 2
        return loss

    def forward(
        self,
        x: TensorType[torch.float, "num_nodes", "d", "num_features"],
        normalize: bool = False,
    ) -> TensorType[torch.float, ()]:
        return self.dirichlet_energy(x, normalize=normalize)

    def build_from_maps(
        self,
        # maps: Optional[TensorType[torch.float, "num_edges", "stalk_dim", "stalk_dim"]],
        # edge_index: Optional[TensorType[torch.long, 2, "num_edges"]] = None,
        # num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        maps = self.restriction_maps
        edge_index = self.edge_index
        num_nodes = self.num_nodes
        self.deg = degree(edge_index[0], num_nodes=num_nodes)
        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.full_left_right_idx, _ = compute_left_right_map_index(
            edge_index, full_matrix=True
        )
        (
            self.left_right_idx,
            self.vertex_tril_idx,
        ) = compute_left_right_map_index(edge_index)
        (
            self.diag_indices,
            self.tril_indices,
        ) = compute_learnable_laplacian_indices(
            num_nodes, self.vertex_tril_idx, self.d, self.final_d
        )

        if self.add_lp or self.add_hp:
            (
                self.fixed_diag_indices,
                self.fixed_tril_indices,
            ) = compute_fixed_diag_laplacian_indices(
                num_nodes, self.vertex_tril_idx, self.d, self.final_d
            )

        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = edge_index

        # Compute transport maps.
        assert torch.all(torch.isfinite(maps))
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(
            torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps
        )
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = torch.bmm(torch.transpose(maps, dim0=-1, dim1=-2), maps)
        diag_maps = scatter_add(diag_maps, row, dim=0, dim_size=num_nodes)

        # Normalise the transport maps.
        diag_maps, tril_maps = self.normalise(
            diag_maps, tril_maps, tril_row, tril_col
        )
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (
            tril_indices,
            tril_maps,
        ) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps
        )

        # Add the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )
        laplacian = to_torch_csr_tensor(edge_index, weights)

        return laplacian

    def normalise(self, diag_maps, non_diag_maps, tril_row, tril_col):
        if self.normalised:
            # Normalise the entries if the normalised Laplacian is used.
            """if self.training:
                # During training, we perturb the matrices to ensure they have different singular values.
                # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
                eps = (
                    torch.FloatTensor(self.d)
                    .uniform_(-0.001, 0.001)
                    .to(device=self.device)
                )
            else:"""
            eps = torch.zeros(self.d, device=self.device)

            to_be_inv_diag_maps = (
                diag_maps + torch.diag(1.0 + eps).unsqueeze(0)
                if self.augmented
                else diag_maps
            )
            d_sqrt_inv = batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)
            assert torch.all(torch.isfinite(d_sqrt_inv))
            left_norm = d_sqrt_inv[tril_row]
            right_norm = d_sqrt_inv[tril_col]
            non_diag_maps = (left_norm @ non_diag_maps @ right_norm).clamp(
                min=-1, max=1
            )
            diag_maps = (d_sqrt_inv @ diag_maps @ d_sqrt_inv).clamp(
                min=-1, max=1
            )
            assert torch.all(torch.isfinite(non_diag_maps))
            assert torch.all(torch.isfinite(diag_maps))
        elif self.deg_normalised:
            # These are general d x d maps so we need to divide by 1 / sqrt(deg * d), their maximum possible norm.
            deg_sqrt_inv = (
                (self.deg * self.d + 1).pow(-1 / 2)
                if self.augmented
                else (self.deg * self.d + 1).pow(-1 / 2)
            )
            deg_sqrt_inv = deg_sqrt_inv.view(-1, 1, 1)
            left_norm = deg_sqrt_inv[tril_row]
            right_norm = deg_sqrt_inv[tril_col]
            non_diag_maps = left_norm * non_diag_maps * right_norm
            diag_maps = deg_sqrt_inv * diag_maps * deg_sqrt_inv
        return diag_maps, non_diag_maps

    def append_fixed_maps(
        self, size, diag_indices, diag_maps, tril_indices, tril_maps
    ):
        if not self.add_lp and not self.add_hp:
            return (diag_indices, diag_maps), (tril_indices, tril_maps)

        fixed_diag, fixed_non_diag = self.get_fixed_maps(size, tril_maps.dtype)
        tril_row, tril_col = self.vertex_tril_idx

        # Normalise the fixed parts.
        if self.normalised:
            fixed_diag, fixed_non_diag = self.scalar_normalise(
                fixed_diag, fixed_non_diag, tril_row, tril_col
            )
        fixed_diag, fixed_non_diag = fixed_diag.view(-1), fixed_non_diag.view(
            -1
        )
        # Combine the learnable and fixed parts.
        tril_indices, tril_maps = mergesp(
            self.fixed_tril_indices, fixed_non_diag, tril_indices, tril_maps
        )
        diag_indices, diag_maps = mergesp(
            self.fixed_diag_indices, fixed_diag, diag_indices, diag_maps
        )

        return (diag_indices, diag_maps), (tril_indices, tril_maps)

    def get_fixed_maps(self, size, dtype):
        assert self.add_lp or self.add_hp

        fixed_diag, fixed_non_diag = [], []
        if self.add_lp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(
                torch.ones(size=(size, 1), device=self.device, dtype=dtype)
            )
        if self.add_hp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(
                -torch.ones(size=(size, 1), device=self.device, dtype=dtype)
            )

        fixed_diag = torch.cat(fixed_diag, dim=1)
        fixed_non_diag = torch.cat(fixed_non_diag, dim=1)

        assert self.fixed_tril_indices.size(1) == fixed_non_diag.numel()
        assert self.fixed_diag_indices.size(1) == fixed_diag.numel()

        return fixed_diag, fixed_non_diag

    def scalar_normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float("inf"), 0)
        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if tril.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if diag.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps

    def build_sheaf_laplacian(self):
        """
        Builds a sheaf laplacian given the edge_index and the restriction maps

        Args:
            N: The number of nodes in the graph
            K: The dimensionality of the Stalks
            edge_index: Edge index of the graph without duplicate edges. We assume that edge i has orientation
                edge_index[0, i] --> edge_index[1, i].
            maps: Tensor of shape [edge_index.size(1), 2 (source/target), K, K] containing the restriction maps of the sheaf
        Returns:
            (index, value): The sheaf Laplacian as a sparse matrix of size (N*K, N*K)
        """
        index = []
        values = []

        for e in range(self.num_edges):
            source = self.edge_index[0, e]
            target = self.edge_index[1, e]

            top_x = e * self.d
            # Generate the positions in the block matrix
            top_y = source * self.d
            for i, j in itertools.product(range(self.d), range(self.d)):
                index.append([top_x + i, top_y + j])
                values.append(-self.restriction_maps[e, 0, i, j])

            top_y = target * self.d
            for i, j in itertools.product(range(self.d), range(self.d)):
                index.append([top_x + i, top_y + j])
                values.append(self.restriction_maps[e, 1, i, j])

        index = torch.tensor(index, dtype=torch.long).T
        print(f"index={index}")
        values = torch.tensor(values)
        print(f"values={values}")

        index_t, values_t = torch_sparse.transpose(
            index, values, self.num_edges * self.d, self.num_nodes * self.d
        )
        index, value = torch_sparse.spspmm(
            index_t,
            values_t,
            index,
            values,
            self.num_nodes * self.d,
            self.num_edges * self.d,
            self.num_nodes * self.d,
            coalesced=True,
        )
        index, value = torch_sparse.coalesce(
            index, value, self.num_nodes * self.d, self.num_nodes * self.d
        )
        return torch.sparse_coo_tensor(index, value)

    def train(
        self,
        data,
        lr: float,
        eps: float,
        log_every: int,
        reg: Optional[str] = "matrix",  # options are l1, l2, matrix, or None
        lambda_reg: float = 0.5,
        normalize: bool = True,
    ):
        device = next(self.parameters()).device
        data = data.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr,
            max_lr=lr * 10,
            step_size_up=1000,
            mode="exp_range",
            gamma=0.95,
            cycle_momentum=False,
        )
        print(f"Starting training on {device}")
        normalized_energy = float("inf")
        epoch = 0
        with tqdm.tqdm(desc="Training progress") as pbar:
            while eps > 0 and eps < normalized_energy:
                epoch += 1
                # for epoch in range(num_epochs):
                optimizer.zero_grad()
                loss = self(data, normalize=normalize)
                norm = torch.linalg.norm(
                    data.reshape(self.num_nodes * self.d, -1), ord="fro"
                )
                normalized_energy = loss.item() / norm**2
                if reg is not None:
                    loss_reg = 0
                    if reg.lower() == "matrix":
                        loss_reg += torch.norm(self.restriction_maps, p="fro")
                    elif reg.lower() in ["l1", "l2"]:
                        for p in self.parameters():
                            if reg.lower() == "l1":
                                loss_reg += p.abs().sum()
                            elif reg.lower() == "l2":
                                loss_reg += p.pow(2).sum()
                    else:
                        raise ValueError(f"Unknown norm_reg: {reg}")
                    loss_total = (1 - lambda_reg) * loss - lambda_reg * loss_reg
                else:
                    loss_total = loss
                loss_total.backward()
                optimizer.step()
                scheduler.step()
                if epoch % log_every == 0:
                    message = f"Epoch {epoch+1}"  # /{num_epochs}"
                    if reg is not None:
                        message += (
                            f", Loss Total: {loss_total.item():.4f}"
                            + f", Loss: {loss.item():.4f}"
                            + f", Loss Reg: {loss_reg:.4f}"
                            + f", Sheaf Dirichlet Energy normalized: {loss.item()/norm**2:.4f}"
                        )
                    else:
                        message += (
                            f", Loss: {loss_total.item():.4f}"
                            + f", Sheaf Dirichlet Energy normalized: {loss.item()/norm**2:.4f}"
                        )
                    tqdm.tqdm.write(message)
                pbar.update(1)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(d={self.d})"


def remove_duplicate_edges(edge_index):
    processed_edges = set()
    new_edge_index = []

    for e in range(edge_index.size(1)):
        source, target = sorted(
            (edge_index[0, e].item(), edge_index[1, e].item())
        )
        if (source, target) in processed_edges:
            continue
        processed_edges.add((source, target))
        new_edge_index.append([source, target])
    return torch.tensor(new_edge_index, dtype=torch.long).t()
