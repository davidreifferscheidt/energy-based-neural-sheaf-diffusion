#!/usr/bin/env python

import os
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

import wandb
from cbsd.config import (
    instantiate_datamodule,
    instantiate_model,
    instantiate_sheaf_laplacian,
    instantiate_task,
)
from cbsd.utils import (
    WandbModelCheckpoint,
    WandbSummaries,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

# Stop pytorch-lightning from pestering us about things we already know
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)
# If data loading is really not a bottleneck for you, you uncomment this to silence the
# warning about it
warnings.filterwarnings(
    "ignore",
    "The dataloader, [^,]+, does not have many workers",
    module="pytorch_lightning",
)

log = get_logger()


def get_callbacks(config):
    monitor = {"monitor": "val/top1", "mode": "max"}
    callbacks = [
        WandbSummaries(**monitor),
        WandbModelCheckpoint(
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            filename="best",
            **monitor,
        ),
        TQDMProgressBar(refresh_rate=1),
    ]
    if config.early_stopping is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    return callbacks


@hydra.main(config_path="config", config_name="exec", version_base=None)
@print_exceptions
def main(config: DictConfig):
    rng = set_seed(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)
    wandb.init(
        **config.wandb, resume=(config.wandb.mode == "online") and "allow"
    )

    log.info("Loading data")
    datamodule = instantiate_datamodule(config.data)
    datamodule.prepare_data()
    datamodule.setup()

    log.info("Instantiating model")
    model = instantiate_model(
        config.model, datamodule, accelerator=config.trainer.accelerator
    )

    if hasattr(config.model, "sheaf_laplacian"):
        log.info("Precomputing Sheaf Laplacian")
        instantiate_sheaf_laplacian(
            config.model.sheaf_laplacian,
            datamodule,
            model,
            accelerator=config.trainer.accelerator,
        )

    log.info("Instantiating task")
    task = instantiate_task(
        config.task, model, datamodule, split_idx=config.data.split_idx
    )

    logger = WandbLogger()
    log_hyperparameters(logger, config, model)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config)
    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )
    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    # Track energy/sheaf energy over each layer in each step
    data_energy = [[i, value] for i, value in enumerate(model.energy)]
    table = wandb.Table(
        data=data_energy, columns=["step*num_layers", "Dirichlet Energy"]
    )
    data_sheaf_energy = [
        [i, value] for i, value in enumerate(model.sheaf_energy)
    ]
    table2 = wandb.Table(
        data=data_sheaf_energy,
        columns=["step*num_layers", "Sheaf Dirichlet Energy"],
    )
    if model.__class__.__name__ in [
        "EnergyGCN",
        "EnergyACM",
        "DiscreteDiagSheafDiffusion",
    ]:
        # Track the starting energy and end energy edge-wise

        data_dict_start_e = {
            "index": range(model.starting_energy_edge_distr.size(0)),
            "energy": model.starting_energy_edge_distr,
        }
        dataframe_edge_start = pd.DataFrame(data_dict_start_e)
        dataframe_edge_start = dataframe_edge_start.sort_values(
            "energy", ascending=False
        )
        dataframe_edge_start = dataframe_edge_start.assign(
            sorted=range(len(model.starting_energy_edge_distr))
        )
        table_edge_start = wandb.Table(data=dataframe_edge_start)

        data_dict_end_e = {
            "index": range(len(model.energy_edge_distr)),
            "energy": model.energy_edge_distr,
        }
        dataframe_edge_end = pd.DataFrame(data_dict_end_e)
        dataframe_edge_end = dataframe_edge_end.sort_values(
            "energy", ascending=False
        )
        dataframe_edge_end = dataframe_edge_end.assign(
            sorted=range(len(model.energy_edge_distr))
        )
        table_edge_end = wandb.Table(data=dataframe_edge_end)

        # Track the starting energy and end energy node-wise
        # get predictions of best model for all nodes
        with torch.no_grad():
            predictions = trainer.predict(
                datamodule=datamodule, ckpt_path="best"
            )
        # dictionary with true, predicted labels and node energy
        data_dict_start = {
            "index": range(datamodule.num_nodes),
            "energy": model.starting_energy_node_distr,
            "predicted": predictions[0].eq(datamodule.dataset.y),
            "guess": predictions[0],
            "truth": datamodule.dataset.y,
        }
        dataframe_start = pd.DataFrame(data_dict_start)
        dataframe_start = dataframe_start.sort_values("energy", ascending=False)
        dataframe_start = dataframe_start.assign(
            sorted=range(datamodule.num_nodes)
        )
        table_node_start = wandb.Table(data=dataframe_start)

        data_dict_end = {
            "index": range(datamodule.num_nodes),
            "energy": model.energy_node_distr,
            "predicted": predictions[0].eq(datamodule.dataset.y),
            "guess": predictions[0],
            "truth": datamodule.dataset.y,
        }

        dataframe = pd.DataFrame(data_dict_end)
        dataframe = dataframe.sort_values("energy", ascending=False)
        dataframe = dataframe.assign(sorted=range(datamodule.num_nodes))
        table_node_end = wandb.Table(data=dataframe)
        wandb.log(
            {
                "Energy edge-wise start": wandb.plot.line(
                    table_edge_start,
                    x="index",
                    y="energy",
                    title="Energy Starting Distribution edge-wise",
                )
            }
        )

        wandb.log(
            {
                "Energy edge-wise end": wandb.plot.line(
                    table_edge_end,
                    x="index",
                    y="energy",
                    title="Energy End Distribution edge-wise",
                )
            }
        )
        wandb.log(
            {
                "Energy node-wise start": wandb.plot.line(
                    table_node_start,
                    x="index",
                    y="energy",
                    title="Energy Starting Distribution node-wise",
                )
            }
        )

        wandb.log(
            {
                "Energy node-wise end": wandb.plot.line(
                    table_node_end,
                    x="index",
                    y="energy",
                    title="Energy End Distribution node-wise",
                )
            }
        )
    wandb.log(
        {
            "Dirichlet Energy": wandb.plot.line(
                table,
                x="step*num_layers",
                y="Dirichlet Energy",
                stroke=None,
                title="Dirichlet Energy",
            )
        }
    )
    wandb.log(
        {
            "Sheaf Dirichlet Energy": wandb.plot.line(
                table2,
                x="step*num_layers",
                y="Sheaf Dirichlet Energy",
                stroke=None,
                title="Sheaf Dirichlet Energy",
            )
        }
    )

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    # Print learned parameters of sheaf laplacian etc.
    if model.__class__.__name__ in [
        "DiscreteDiagSheafDiffusion",
        "DiscreteBundleSheafDiffusion",
        "DiscreteGeneralSheafDiffusion",
    ]:
        for i in range(len(model.sheaf_learners)):
            L_max = model.sheaf_learners[i].L.detach().max().item()
            L_min = model.sheaf_learners[i].L.detach().min().item()
            L_avg = model.sheaf_learners[i].L.detach().mean().item()
            L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            L_lambda_max = model.sheaf_learners[i].lambda_max
            print(
                f"Laplacian {i}: Max: {L_max:.8f}, Min: {L_min:.8f}, Avg: {L_avg:.8f}, Abs avg: {L_abs_avg:.8f}, Lambda_max: {L_lambda_max:.8f}"
            )

    with np.printoptions(precision=3, suppress=True):
        for i in range(0, model.num_layers):
            if hasattr(model, "epsilons"):
                print(
                    f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}"
                )
            if hasattr(
                model, "gammas"
            ):  # model.__class__.__name__ == "EnergyGCN":
                print(
                    f"Gammas {i}: {model.gammas[i].detach().cpu().numpy().flatten()}"
                )
            if hasattr(model, "betas"):
                print(
                    f"Betas {i}: {model.betas[i].detach().cpu().numpy().flatten()}"
                )

    wandb.finish()
    log.info(
        f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}"
    )

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
