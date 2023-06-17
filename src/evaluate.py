import pytorch_lightning as pl
from torchsummary import summary

from config import load_config
from model import ResNetClassifier
from rsna_dataloader import RSNAPneumoniaDataModule
from transforms_select import PREPROCESS_TRANSFORMS


CONFIG_PATH = "config.yml"

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    model = ResNetClassifier.load_from_checkpoint(
        configs["inference"]["checkpoint_path"]
    )
    summary(
        model,
        input_size=tuple(configs["model"]["input_size"]),
        batch_size=configs["training"]["batch_size"],
    )
    model.eval()
    rsna = RSNAPneumoniaDataModule(configs, test_transforms=PREPROCESS_TRANSFORMS)
    pl.seed_everything(33, workers=True)
    trainer = pl.Trainer()
    trainer.test(model=model, dataloaders=rsna.test_dataloader())
