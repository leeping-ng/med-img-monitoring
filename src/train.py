
import pytorch_lightning as pl
from torchsummary import summary
from torchvision import transforms

from config import load_config
from model import ResNetClassifier
from rsna_dataloader import RSNAPneumoniaDataModule


CONFIG_PATH = "config.yml"

def setup(configs):
    model = ResNetClassifier(configs)
    model.cuda()
    summary(model, 
            input_size=tuple(configs["model"]["input_size"]), 
            batch_size=configs["training"]["batch_size"])
    pl.seed_everything(33, workers=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_roc-auc", 
                                                     mode="max", 
                                                     patience=configs["training"]["patience"])
    logger = pl.loggers.TensorBoardLogger(save_dir=configs["training"]["logs_folder"])
    trainer = pl.Trainer(max_epochs=configs["training"]["max_epochs"], 
                         callbacks=[early_stop_callback], 
                         logger=logger)

    return model, trainer

def prepare_data(configs):
    transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.Resize(256, antialias=True),
                                        transforms.RandomResizedCrop(224, (0.8,1), antialias=True)])
                                        # transforms.CenterCrop(224)])
    rsna = RSNAPneumoniaDataModule(configs, train_transforms=transform, val_transforms=transform)
    return rsna


if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    model, trainer = setup(configs)
    rsna = prepare_data(configs)
    trainer.fit(model, rsna)
    trainer.test(dataloaders=rsna.test_dataloader(), ckpt_path="best")