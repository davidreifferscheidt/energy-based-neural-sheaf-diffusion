from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from cbsd.data.utils import get_dataset

patch_typeguard()


@typechecked
class NodeClassificationDataModule(pl.LightningDataModule):
    """General PyTorch Lightning DataModule for node classification datasets."""

    def __init__(
        self,
        root: str,
        collection: str,
        name: str,
        split_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.args = {
            "root": root,
            "collection": collection,
            "name": name,
            "split_name": split_name,
            "transform": transform,
            "pre_transform": pre_transform,
        }

    def prepare_data(self) -> None:
        get_dataset(**self.args)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = get_dataset(**self.args)
        data = self.dataset[0]
        assert ~torch.any(
            torch.any(data.train_mask & data.val_mask, dim=0)
        ), "train and val masks must be disjoint"
        assert ~torch.any(
            torch.any(data.train_mask & data.test_mask, dim=0)
        ), "train and test masks must be disjoint"
        assert ~torch.any(
            torch.any(data.val_mask & data.test_mask, dim=0)
        ), "val and test masks must be disjoint"

    @property
    def x(self) -> TensorType[torch.float, "num_nodes", "num_features"]:
        return self.dataset[0].x

    @property
    def edge_index(self) -> TensorType[2, "num_edges"]:
        return self.dataset[0].edge_index

    @property
    def num_nodes(self) -> int:
        return self.dataset[0].num_nodes

    @property
    def num_features(self) -> int:
        return self.dataset[0].num_features

    @property
    def num_classes(self) -> int:
        return self.dataset[0].y.max().item() + 1

    @property
    def num_splits(self) -> int:
        shape = self.dataset[0].train_mask.shape
        if len(shape) == 1:
            return 1
        else:
            num_splits = shape[1]
            assert (
                num_splits == self.dataset[0].val_mask.shape[1]
            ), "num splits in train and val mask must be equal"
            assert (
                num_splits == self.dataset[0].test_mask.shape[1]
            ), "num splits in train and test mask must be equal"
            return num_splits

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)
