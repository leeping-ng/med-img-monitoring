import pytorch_lightning as pl
from torchsummary import summary

from config import load_config
from model import ResNetClassifier
from uae_model import UntrainedAutoEncoder
from rsna_dataloader import RSNAPneumoniaDataModule
from transforms_select import PREPROCESS_TF

from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.cd import MMDDrift
from functools import partial
import numpy as np
from torchvision import transforms

from transforms_custom import (
    ContrastTransform,
    GammaTransform,
    SaltPepperNoiseTransform,
    SpeckleNoiseTransform,
    BlurTransform,
    SharpenTransform,
)

CONFIG_PATH = "config.yml"

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    resnet_model = ResNetClassifier.load_from_checkpoint(
        configs["inference"]["checkpoint_path"]
    )
    uae_model = UntrainedAutoEncoder(configs, embedding_size=128)
    # summary(
    #     resnet_model,
    #     input_size=tuple(configs["model"]["input_size"]),
    #     batch_size=configs["training"]["batch_size"],
    # )
    uae_model.cuda()
    summary(
        uae_model,
        input_size=tuple(configs["model"]["input_size"]),
        batch_size=configs["training"]["batch_size"],
    )
    resnet_model.eval()
    uae_model.eval()
    rsna = RSNAPneumoniaDataModule(configs, test_transforms=PREPROCESS_TF)
    pl.seed_everything(33, workers=True)
    trainer = pl.Trainer()
    # trainer.test(model=resnet_model, dataloaders=rsna.test_dataloader())
    img_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    original_dataloader = rsna.predict_dataloader(img_indices, PREPROCESS_TF)
    # original_output = trainer.predict(model=uae_model, dataloaders=original_dataloader)
    # original_batch = next(iter(original_output)).numpy()
    original_output = trainer.predict(
        model=resnet_model, dataloaders=original_dataloader
    )
    original_batch = next(iter(original_output))["embeddings"].numpy()

    transform = transforms.Compose([PREPROCESS_TF, ContrastTransform(3)])
    shifted_dataloader = rsna.predict_dataloader(img_indices, transform)
    # shifted_output = trainer.predict(model=uae_model, dataloaders=shifted_dataloader)
    # shifted_batch = next(iter(shifted_output)).numpy()
    shifted_output = trainer.predict(model=resnet_model, dataloaders=shifted_dataloader)
    shifted_batch = next(iter(shifted_output))["embeddings"].numpy()

    cd = MMDDrift(original_batch, backend="pytorch")
    preds = cd.predict(shifted_batch)
    print(preds["data"])
