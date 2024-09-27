from typing import Callable, List, Optional, Union

import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_geometric.datasets import (
    Actor,
    Amazon,
    AmazonProducts,
    CitationFull,
    Coauthor,
    DeezerEurope,
    FacebookPagePage,
    Flickr,
    HeterophilousGraphDataset,
    LastFMAsia,
    LINKXDataset,
    Planetoid,
    Reddit,
    Reddit2,
    Twitch,
    WebKB,
    WikiCS,
    WikipediaNetwork,
    Yelp,
)
from torchtyping import patch_typeguard
from typeguard import typechecked

from cbsd.data.datasets import OBGNDataset, WikipediaNetworkFiltered

patch_typeguard()


NC_DATASETS = {
    "planetoid": {
        "class_name": Planetoid,
        "root": "/Planetoid",
        "names": ["Cora", "CiteSeer", "PubMed"],
        "split_names": ["public", "full", "geom-gcn", "random"],
    },
    "citationfull": {
        "class_name": CitationFull,
        "root": "/CitationFull",
        "names": ["Cora", "Cora_ML", "CiteSeer", "PubMed", "DBLP"],
    },
    "coauthor": {
        "class_name": Coauthor,
        "root": "/Coauthor",
        "names": ["CS", "Physics"],
    },
    "amazon": {
        "class_name": Amazon,
        "root": "/Amazon",
        "names": ["Computers", "Photo"],
    },
    "reddit": {
        "class_name": Reddit,
        "root": "/Reddit",
        "names": None,
    },
    "reddit2": {
        "class_name": Reddit2,
        "root": "/Reddit2",
        "names": None,
    },
    "flickr": {
        "class_name": Flickr,
        "root": "/Flickr",
        "names": None,
    },
    "yelp": {
        "class_name": Yelp,
        "root": "/Yelp",
        "names": None,
    },
    "amazonproducts": {
        "class_name": AmazonProducts,
        "root": "/AmazonProducts",
        "names": None,
    },
    "wikics": {
        "class_name": WikiCS,
        "root": "/WikiCS",
        "names": None,
    },
    "webkb": {
        "class_name": WebKB,
        "root": "/WebKB",
        "names": ["Cornell", "Texas", "Wisconsin"],
    },
    "wikipedianetwork": {
        "class_name": WikipediaNetwork,
        "root": "/WikipediaNetwork",
        "names": ["Chameleon", "Squirrel", "Crocodile"],
    },
    "wikipedianetworkfiltered": {
        "class_name": WikipediaNetworkFiltered,
        "root": "/WikipediaNetwork",
        "names": [
            "chameleon-filtered",
            "chameleon-filtered-directed",
            "squirrel-filtered",
            "squirrel-filtered-directed",
        ],
    },
    "heterophilousgraphdataset": {
        "class_name": HeterophilousGraphDataset,
        "root": "/HeterophilousGraphDataset",
        "names": [
            "Roman-Empire",
            "Amazon-Ratings",
            "Minesweeper",
            "Tolokers",
            "Questions",
        ],
    },
    "actor": {
        "class_name": Actor,
        "root": "/Actor",
        "names": None,
    },
    "facebookpagepage": {
        "class_name": FacebookPagePage,
        "root": "/FacebookPagePage",
        "names": None,
    },
    "lastfmasia": {
        "class_name": LastFMAsia,
        "root": "/LastFMAsia",
        "names": None,
    },
    "deezereurope": {
        "class_name": DeezerEurope,
        "root": "/DeezerEurope",
        "names": None,
    },
    "twitch": {
        "class_name": Twitch,
        "root": "/Twitch",
        "names": ["de", "en", "es", "fr", "pt", "ru"],
    },
    "linkxdataset": {
        "class_name": LINKXDataset,
        "root": "/LINKXDataset",
        "names": [
            "Penn84",
            "Reed98",
            "Amherst41",
            "Cornell5",
            "JohnsHopkins55",
            "Genius",
        ],
    },
    "ogbn": {
        "class_name": OBGNDataset,
        "root": "/OGBN",
        "names": ["ogbn-arxiv"],
    },
}


@typechecked
def to_lower(strings: List[str]) -> List[str]:
    return [s.lower() for s in strings]


@typechecked
def get_dataset(
    root: str,
    collection: str,
    name: Optional[str] = None,
    split_name: Optional[str] = None,
    transform: Optional[Union[Callable, List[Callable]]] = None,
    pre_transform: Optional[Union[Callable, List[Callable]]] = None,
    **kwargs,
) -> Dataset:
    if collection.lower() not in NC_DATASETS.keys():
        raise ValueError(f"Collection {collection} not recognized.")

    if name is not None:
        if NC_DATASETS[collection.lower()]["names"] is not None:
            if name.lower() not in to_lower(
                NC_DATASETS[collection.lower()]["names"]
            ):
                raise ValueError(
                    f"Name {name} for collection {collection} not recognized."
                )
            else:
                name_kwargs = {
                    "name": [
                        n
                        for n in NC_DATASETS[collection.lower()]["names"]
                        if n.lower() == name.lower()
                    ][0]
                }
        else:
            name_kwargs = {}

    if collection.lower() == "ogbn":
        if transform is not None:
            transform = T.Compose([transform, T.ToSparseTensor()])

    if (
        collection.lower() in ["planetoid", "planetoid_geom_gcn"]
        and split_name is not None
    ):
        if split_name.lower() not in to_lower(
            NC_DATASETS[collection.lower()]["split_names"]
        ):
            raise ValueError(
                f"Split {split_name} for collection {collection} not recognized."
            )
        split_kwargs = {
            "split": [
                s
                for s in NC_DATASETS[collection.lower()]["split_names"]
                if s.lower() == split_name.lower()
            ][0]
        }
    else:
        split_kwargs = {}

    return NC_DATASETS[collection.lower()]["class_name"](
        root=root + NC_DATASETS[collection.lower()]["root"],
        transform=transform,
        pre_transform=pre_transform,
        **name_kwargs,
        **split_kwargs,
        **kwargs,
    )
