import pytorch_lightning as pl
from torchsummary import summary
from torchvision import transforms

from config import load_config
from model import ResNetClassifier
from rsna_dataloader import RSNAPneumoniaDataModule
from transforms import preprocess_transforms, train_transforms


CONFIG_PATH = "config.yml"


def setup(configs):
    model = ResNetClassifier(configs)
    model.cuda()
    summary(
        model,
        input_size=tuple(configs["model"]["input_size"]),
        batch_size=configs["training"]["batch_size"],
    )
    pl.seed_everything(33, workers=True)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_roc-auc", mode="max", patience=configs["training"]["patience"]
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_roc-auc",
        save_top_k=1,
        mode="max",
        filename="{epoch:02d}-{val_roc-auc:.3f}",
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=configs["training"]["logs_folder"])
    trainer = pl.Trainer(
        max_epochs=configs["training"]["max_epochs"],
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
    )

    return model, trainer


def prepare_data(configs):
    rsna = RSNAPneumoniaDataModule(
        configs,
        train_transforms=train_transforms,
        val_transforms=preprocess_transforms,
        test_transforms=preprocess_transforms,
    )
    return rsna


if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    model, trainer = setup(configs)
    rsna = prepare_data(configs)
    trainer.fit(model, rsna)
    trainer.test(dataloaders=rsna.test_dataloader(), ckpt_path="best")
