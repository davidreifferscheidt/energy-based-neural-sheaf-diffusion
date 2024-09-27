from typing import Optional, Union

import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.random_projection import GaussianRandomProjection
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops
from torchtyping import TensorType


@functional_transform("remove_self_loops")
class RemoveSelfLoops(BaseTransform):
    def __init__(self, attr: Optional[str] = "edge_weight"):
        self.attr = attr

    def __call__(
        self, data: Union[Data, HeteroData]
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or "edge_index" not in store:
                continue

            store.edge_index, edge_weight = remove_self_loops(
                store.edge_index, getattr(store, self.attr, None)
            )

            setattr(store, self.attr, edge_weight)
        return data


@functional_transform("dimension_reduction")
class DimensionReduction(BaseTransform):
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def __call__(self, data):
        self.data_device = data.x.device
        x = data.x

        print(f"Dimensionality reduction...")
        x = dimred(
            x=x,
            dim=self.config.dr_dimension,
            method=self.config.dr_method,
        ).to(self.data_device)
        data.x = x
        print("Done.")
        return data


def dimred(
    x: TensorType[torch.float, "num_nodes", "num_features"],
    dim: int = 128,
    method: str = "pca",
) -> TensorType[torch.float, "num_nodes", "d*num_channels"]:
    if method.lower() == "pca":
        # TODO n_components must be between 0 and min(n_samples, n_features)=183 with svd_solver='full'
        transform = PCA(dim)
    elif method.lower() == "tsne":
        transform = TSNE(dim)
    elif method.lower() == "isomap":
        transform = Isomap(n_components=dim)
    elif method.lower() == "random_projection":
        transform = GaussianRandomProjection(n_components=dim)
    else:
        pass  # TODO attribute error?
    x = torch.tensor(transform.fit_transform(x), dtype=torch.float32)
    return x
