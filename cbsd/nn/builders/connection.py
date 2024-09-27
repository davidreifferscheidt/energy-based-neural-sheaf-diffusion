from typing import Optional

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .base import BaseLaplacianBuilder

patch_typeguard()


@typechecked
class ConnectionLaplacianBuilder(BaseLaplacianBuilder):
    def build(
        self,
        x: Optional[TensorType[torch.float, "num_nodes", "num_features"]] = None,
        edge_index: Optional[TensorType[torch.long, 2, "num_edges"]] = None,
    ) -> torch.Tensor:
        return None
