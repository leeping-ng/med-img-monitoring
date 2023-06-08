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
        self.acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", 
                            num_classes=num_classes)
        self.auroc = AUROC(task="binary" if num_classes == 2 else "multiclass", 
                           num_classes=num_classes)
        
        # Using a pretrained ResNet backbone
        backbone = self.resnets[resnet_version](weights="IMAGENET1K_V1")
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
        roc = self.auroc(preds, y)
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

    