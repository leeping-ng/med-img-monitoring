import pytorch_lightning as pl
from torchsummary import summary

from config import load_config
from model import ResNetClassifier
from rsna_dataloader import RSNAPneumoniaDataModule


CONFIG_PATH = "config.yml"

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    model = ResNetClassifier.load_from_checkpoint(configs["evaluation"]["checkpoint_path"])
    summary(model, 
            input_size=tuple(configs["model"]["input_size"]), 
            batch_size=configs["training"]["batch_size"])
    model.eval()
    rsna = RSNAPneumoniaDataModule(configs)
    trainer = pl.Trainer()
    trainer.test(model=model, dataloaders=rsna.test_dataloader())
    #trainer.predict(model=model, dataloaders=rsna.predict_dataloader())