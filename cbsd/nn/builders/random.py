from typing import Optional

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .base import BaseLaplacianBuilder

patch_typeguard()


@typechecked
class RandomLaplacianBuilder(BaseLaplacianBuilder):
    def build(
        self,
        x: Optional[TensorType[torch.float, "num_nodes", "num_features"]] = None,
        edge_index: Optional[TensorType[torch.long, 2, "num_edges"]] = None,
    ) -> torch.Tensor:
        n = edge_index.max().item() + 1
        d = self.d

        nd = n * d
        L = torch.zeros((nd, nd))
        for i, j in zip(*edge_index):
            L[i * d : (i + 1) * d, j * d : (j + 1) * d] = torch.randn((d, d))
            L[j * d : (j + 1) * d, i * d : (i + 1) * d] = L[
                i * d : (i + 1) * d, j * d : (j + 1) * d
            ].T
        return L.to_sparse_csr()
