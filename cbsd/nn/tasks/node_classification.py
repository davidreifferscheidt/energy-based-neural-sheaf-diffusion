from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
class NodeClassificationTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optimizer_config: DictConfig,
        split_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.loss = nn.CrossEntropyLoss()

        task = "multiclass" if num_classes > 2 else "binary"

        metrics = tm.MetricCollection(
            {
                "top1": tm.Accuracy(
                    task=task, num_classes=num_classes, top_k=1
                ),
                "top3": tm.Accuracy(
                    task=task, num_classes=num_classes, top_k=3
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.split_idx = split_idx

    def classify(
        self, logits: TensorType[float, "num_nodes", "num_classes"]
    ) -> TensorType[float, "num_nodes", "num_classes"]:
        return nn.functional.softmax(logits, dim=-1)

    def compute_logits(
        self, batch: Batch
    ) -> TensorType[float, "num_nodes", "num_classes"]:
        x = batch.x

        model_name = self.model.__class__.__name__

        if model_name in [
            "MLP",
            "DiscreteDiagSheafDiffusion",
            "DiscreteBundleSheafDiffusion",
            "DiscreteGeneralSheafDiffusion",
            "EnergyGCN",
            "EnergyACM",
        ]:
            model_inputs = {"x": x}
        elif model_name == "SNN":
            if hasattr(self.model, "sheaf_laplacian"):
                model_inputs = {
                    "x": x,
                    "sheaf_laplacian": self.model.sheaf_laplacian.to(x.device),
                }
            else:
                raise RuntimeError(
                    "SNN model requires a sheaf Laplacian. "
                    "Please instantiate the model with a sheaf Laplacian."
                )
        else:
            model_inputs = {"x": x, "edge_index": batch.edge_index}

        return self.model(**model_inputs)

    def set_phase(self, phase: str, batch: Batch) -> None:
        if phase == "train":
            mask = batch["train_mask"].bool()
        elif phase == "val":
            mask = batch["val_mask"].bool()
        elif phase == "test":
            mask = batch["test_mask"].bool()

        if self.split_idx is not None:
            self.mask = mask[:, self.split_idx]
        else:
            self.mask = mask

    def step(
        self, phase: str, batch: Batch
    ) -> Tuple[TensorType[float, ...], Optional[Dict]]:
        self.set_phase(phase, batch)
        y = batch.y[self.mask]
        logits = self.compute_logits(batch)[self.mask]
        loss = self.loss(logits, y)
        self.log(f"{phase}/loss", loss.mean(), batch_size=batch.x.shape[0])
        if hasattr(self.model, "sheaf_learners"):
            for l in range(len(self.model.sheaf_learners)):
                self.log(
                    f"Lambda max^2 Laplacian {l}",
                    self.model.sheaf_learners[l].lambda_max ** 2,
                )
                self.log(
                    f"Sigma max^2 Laplacian {l}",
                    self.model.sheaf_learners[l].sigma_max ** 2,
                )
        elif hasattr(self.model, "lambda_max"):
            self.log(
                f"Lambda max^2 Laplacian",
                self.model.lambda_max**2,
            )
        elif hasattr(self.model, "sigma_max"):
            self.log(
                f"Sigma max^2 Laplacian",
                self.model.sigma_max**2,
            )
        if hasattr(self.model.lin_right_weights[0], "spectral_norm"):
            for l in range(len(self.model.lin_right_weights)):
                self.log(
                    f"Spectral norm^2 right weights {l}",
                    self.model.lin_right_weights[l].spectral_norm ** 2,
                )
            for l in range(len(self.model.lin_left_weights)):
                self.log(
                    f"Spectral norm^2 left weights {l}",
                    self.model.lin_left_weights[l].spectral_norm ** 2,
                )

        if phase == "train":
            metrics = self.train_metrics(self.classify(logits), y)
        elif phase == "val":
            metrics = self.val_metrics(self.classify(logits), y)
        elif phase == "test":
            metrics = self.test_metrics(self.classify(logits), y)

        return loss, metrics

    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, TensorType[float, ...]]:
        loss, metrics = self.step("train", batch)
        self.log_dict(
            metrics,
            prog_bar=True,
            batch_size=batch.x.shape[0],
        )
        return {"loss": loss}

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        _, metrics = self.step("val", batch)
        self.log_dict(
            metrics,
            prog_bar=True,
            batch_size=batch.x.shape[0],
        )

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        _, metrics = self.step("test", batch)
        self.log_dict(
            metrics,
            prog_bar=True,
            batch_size=batch.x.shape[0],
        )

    def predict_step(self, batch: Batch, batch_idx: int):
        logits = self.compute_logits(batch)
        return logits.argmax(dim=1)

    def configure_optimizers(self):  # TODO: define function output
        model_name = self.model.__class__.__name__

        if model_name in [
            "DiscreteDiagSheafDiffusion",
            "DiscreteBundleSheafDiffusion",
            "DiscreteGeneralSheafDiffusion",
        ]:
            sheaf_learner_params, other_params = self.model.grouped_parameters()
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": sheaf_learner_params,
                        "weight_decay": self.hparams.optimizer_config.sheaf_decay,
                    },
                    {
                        "params": other_params,
                        "weight_decay": self.hparams.optimizer_config.weight_decay,
                    },
                ],
                lr=self.hparams.optimizer_config.lr,
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                weight_decay=self.hparams.optimizer_config.weight_decay,
                lr=self.hparams.optimizer_config.lr,
            )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.hparams.optimizer_config.lr,
            max_lr=self.hparams.optimizer_config.lr * 10,
            step_size_up=1000,
            mode="exp_range",
            gamma=0.95,
            cycle_momentum=False,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
