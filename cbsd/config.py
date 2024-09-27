from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from omegaconf import DictConfig
from torch_geometric.nn.models import GAT, GCN, MLP, GraphSAGE
from torchtyping import patch_typeguard

from cbsd.baselines.nsd_reproduction.models.nsd import (
    DiscreteBundleSheafDiffusion,
    DiscreteDiagSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
)
from cbsd.data.datamodules import NodeClassificationDataModule
from cbsd.data.transforms import DimensionReduction, RemoveSelfLoops
from cbsd.nn.builders import (
    ConnectionLaplacianBuilder,
    ConsistencyBasedLaplacianBuilder,
    RandomLaplacianBuilder,
)
from cbsd.nn.models import SNN, EnergyACM, EnergyGCN
from cbsd.nn.tasks import NodeClassificationTask

patch_typeguard()


def instantiate_datamodule(data_config: DictConfig):
    split_name = (
        None
        if not hasattr(data_config, "split_name")
        else data_config.split_name
    )
    transforms = []
    pretransform = None
    root = data_config.root
    if data_config.remove_self_loops:
        transforms.append(RemoveSelfLoops())
    if data_config.to_undirected:
        transforms.append(T.ToUndirected())
    if data_config.dim_reduction:
        pretransform = DimensionReduction(data_config)
        root = (
            data_config.root
            + "_"
            + data_config.dr_method
            + "_"
            + str(data_config.dr_dimension)
        )
    elif (
        data_config.normalize_features
    ):  # reduced dimension + normalize features doesn't work on Cora
        transforms.append(T.NormalizeFeatures())

    return NodeClassificationDataModule(
        root=root,
        collection=data_config.collection,
        name=data_config.name,
        split_name=split_name,
        transform=T.Compose(transforms),
        pre_transform=pretransform,
    )


def instantiate_sheaf_laplacian(
    config: DictConfig,
    datamodule: pl.LightningDataModule,
    model: nn.Module,
    accelerator: str,
) -> None:
    device = (
        "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    if config.type == "random":
        builder = RandomLaplacianBuilder(config)
        sheaf_laplacian = builder.build(
            x=datamodule.x, edge_index=datamodule.edge_index
        )
        model.sheaf_laplacian = sheaf_laplacian
    elif config.type == "connection":
        builder = ConnectionLaplacianBuilder(config)
        sheaf_laplacian = builder.build(
            x=datamodule.x, edge_index=datamodule.edge_index
        )
        model.sheaf_laplacian = sheaf_laplacian
    elif config.type == "consistent":
        builder = ConsistencyBasedLaplacianBuilder(
            edge_index=datamodule.edge_index.to(device), config=config
        )
        x = datamodule.x
        assert x.size(-1) % config.d == 0
        x = x.reshape(datamodule.num_nodes, config.d, -1)
        builder.train(
            data=x,
            lr=config.pretraining.lr,
            eps=config.pretraining.eps,
            log_every=1,
            reg=config.pretraining.reg,
            lambda_reg=config.pretraining.lambda_reg,
            normalize=config.pretraining.normalize,
        )
        sheaf_laplacian = builder.build_from_maps().detach()
        model.sheaf_laplacian = sheaf_laplacian
        assert model.sheaf_laplacian.grad_fn == None
    else:
        raise NotImplementedError(f"Builder {config.type} not implemented.")


def instantiate_model(
    config: DictConfig, datamodule: pl.LightningDataModule, accelerator: str
):
    device = (
        "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    in_channels = datamodule.num_features
    out_channels = datamodule.num_classes
    num_nodes = datamodule.num_nodes
    edge_index = datamodule.edge_index.to(device)
    if config.name.lower() == "energy-acm":
        return EnergyACM(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            out_channels=out_channels,
            edge_index=edge_index,
            num_nodes=num_nodes,
            dropout=config.dropout,
            use_act=config.use_act,
            right_weights=config.right_weights,
            residual=config.residual,
        )
    if config.name.lower() == "energy-gcn":
        return EnergyGCN(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            out_channels=out_channels,
            edge_index=edge_index,
            num_nodes=num_nodes,
            dropout=config.dropout,
            use_act=config.use_act,
            right_weights=config.right_weights,
            residual=config.residual,
        )
    if config.name.lower() == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif config.name.lower() == "gat":
        return GAT(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif config.name.lower() == "mlp":
        return MLP(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif config.name.lower() == "graphsage":
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif config.name.lower() == "snn":
        return SNN(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=out_channels,
            num_layers=config.num_layers,
            edge_index=edge_index,
            num_nodes=num_nodes,
            input_dropout=config.input_dropout,
            dropout=config.dropout,
            d=config.sheaf_laplacian.d,
            use_act=config.use_act,
            add_hp=config.sheaf_laplacian.add_hp,
            add_lp=config.sheaf_laplacian.add_lp,
            left_weights=config.left_weights,
            right_weights=config.right_weights,
        )
    elif config.name.lower() == "diag-nsd":
        return DiscreteDiagSheafDiffusion(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            d=config.d,
            num_layers=config.num_layers,
            num_nodes=num_nodes,
            edge_index=edge_index,
            out_channels=out_channels,
            input_dropout=config.input_dropout,
            dropout=config.dropout,
            use_act=config.use_act,
            add_lp=config.add_lp,
            add_hp=config.add_hp,
            left_weights=config.left_weights,
            right_weights=config.right_weights,
            sparse_learner=config.sparse_learner,
            linear=config.linear,
            normalised=config.normalised,
            deg_normalised=config.deg_normalised,
            sheaf_act=config.sheaf_act,
            second_linear=config.second_linear,
        )
    elif config.name.lower() == "bundle-nsd":
        return DiscreteBundleSheafDiffusion(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            d=config.d,
            num_layers=config.num_layers,
            num_nodes=num_nodes,
            edge_index=edge_index,
            out_channels=out_channels,
            input_dropout=config.input_dropout,
            dropout=config.dropout,
            use_act=config.use_act,
            add_lp=config.add_lp,
            add_hp=config.add_hp,
            left_weights=config.left_weights,
            right_weights=config.right_weights,
            sparse_learner=config.sparse_learner,
            linear=config.linear,
            normalised=config.normalised,
            deg_normalised=config.deg_normalised,
            sheaf_act=config.sheaf_act,
            second_linear=config.second_linear,
            orth_trans=config.orth_trans,
            use_edge_weights=config.use_edge_weights,
        )
    elif config.name.lower() == "gen-nsd":
        return DiscreteGeneralSheafDiffusion(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            d=config.d,
            num_layers=config.num_layers,
            num_nodes=num_nodes,
            edge_index=edge_index,
            out_channels=out_channels,
            input_dropout=config.input_dropout,
            dropout=config.dropout,
            use_act=config.use_act,
            add_lp=config.add_lp,
            add_hp=config.add_hp,
            left_weights=config.left_weights,
            right_weights=config.right_weights,
            sparse_learner=config.sparse_learner,
            linear=config.linear,
            normalised=config.normalised,
            deg_normalised=config.deg_normalised,
            sheaf_act=config.sheaf_act,
            second_linear=config.second_linear,
        )
    else:
        raise NotImplementedError(f"Model {config.name} not implemented.")


def instantiate_task(
    config: DictConfig,
    model: nn.Module,
    datamodule: pl.LightningDataModule,
    split_idx: Optional[int] = None,
):
    if config.name.lower() == "nodeclassification":
        return NodeClassificationTask(
            model,
            num_classes=datamodule.num_classes,
            optimizer_config=config.optimizer,
            split_idx=split_idx,
        )
