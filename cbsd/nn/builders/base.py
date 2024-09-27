from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
from omegaconf import DictConfig
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
class BaseLaplacianBuilder(ABC):
    def __init__(self, config: DictConfig):
        super().__init__()

        if not hasattr(config, "d"):
            raise ValueError(
                "The config must contain a sheaf_laplacian_dim attribute."
            )

        self.d = config.d

    @abstractmethod
    def build(
        self,
        x: Optional[
            TensorType[torch.float, "num_nodes", "num_features"]
        ] = None,
        edge_index: Optional[TensorType[torch.long, 2, "num_edges"]] = None,
        **kwargs,
    ) -> Any:
        raise NotImplementedError
