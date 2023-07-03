# Adapted from https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torchmetrics import Accuracy, AUROC


class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()

        num_classes = config["model"]["num_classes"]
        resnet_version = config["model"]["resnet_version"]
        self.lr = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]

        self.optimizer = Adam
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(
            task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes
        )
        self.auroc = AUROC(task="multiclass", num_classes=num_classes)

        # Using a pretrained ResNet backbone
        backbone = self.resnets[resnet_version](
            weights="DEFAULT" if config["training"]["transfer_learn"] else None
        )
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, X):
        x = self.feature_extractor(X)
        # Flatten the tensor for linear layer
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch["image"], batch["target"]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        # Labels from logits - softmax prior?
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)

        # change this
        roc = self.auroc(logits, y)
        return loss, acc, roc

    def predict_step(self, batch, batch_idx):
        output = {}
        x, output["target"], output["filename"] = (
            batch["image"],
            batch["target"],
            batch["filename"],
        )
        # essentially forward(), but we want to extract embeddings for MMD later
        x = self.feature_extractor(x)
        embeddings = x.view(x.size(0), -1)
        logits = self.classifier(embeddings)

        softmax = nn.Softmax(dim=1)
        output["embeddings"] = embeddings
        output["softmax"] = softmax(logits)
        output["preds"] = torch.argmax(logits, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        loss, acc, roc = self._step(batch)
        # perform logging
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_roc-auc",
            roc,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, roc = self._step(batch)
        # perform logging
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_roc-auc",
            roc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return roc

    def test_step(self, batch, batch_idx):
        loss, acc, roc = self._step(batch)
        # perform logging
        self.log(
            "test_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_acc",
            acc,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_roc-auc",
            roc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return roc


class UntrainedAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            # TO-DO: avoid hard coding this
            nn.Linear(186624, config["inference"]["embedding_size"]),
        )

    def forward(self, X):
        x = self.feature_extractor(X)
        return x

    def predict_step(self, batch, batch_idx):
        return self.forward(batch["image"])
