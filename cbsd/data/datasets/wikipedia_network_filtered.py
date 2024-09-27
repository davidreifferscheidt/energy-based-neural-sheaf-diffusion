import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data


class WikipediaNetworkFiltered(InMemoryDataset):
    url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{}.npz"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        self.name = self.name.replace("-", "_")
        assert self.name in [
            "chameleon_filtered",
            "chameleon_filtered_directed",
            "squirrel_filtered",
            "squirrel_filtered_directed",
        ]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        splits = self.name.split("_")
        return f"{''.join([s.capitalize() for s in splits])}()"


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(f):
    x = torch.tensor(f["node_features"]).to(torch.float)
    y = torch.tensor(f["node_labels"]).to(torch.long)
    edge_index = torch.tensor(f["edges"]).T
    edge_index, _ = remove_self_loops(edge_index, None)

    train_mask = torch.tensor(f["train_masks"]).T.to(torch.bool)
    val_mask = torch.tensor(f["val_masks"]).T.to(torch.bool)
    test_mask = torch.tensor(f["test_masks"]).T.to(torch.bool)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
