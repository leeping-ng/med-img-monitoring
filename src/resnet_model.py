# Adapted from https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
from torchvision import transforms

from torchmetrics.classification import BinaryAUROC

class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        num_classes,
        resnet_version,
        optimizer="adam",
        lr=1e-4,
        batch_size=32,
        tune_fc_only=True,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        self.b_auroc = BinaryAUROC()
        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[resnet_version](weights="IMAGENET1K_V1")
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        # Added this
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        roc = self.b_auroc(preds, y)
        return loss, acc, roc


    def training_step(self, batch, batch_idx):
        loss, acc, roc = self._step(batch)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_roc-auc", roc, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc, roc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("val_roc-auc", roc, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)


    def test_step(self, batch, batch_idx):
        loss, acc, roc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("test_roc-auc", roc, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

    