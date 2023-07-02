import numpy as np
import pandas as pd
import pytorch_lightning as pl

from config import load_config
from model import ResNetClassifier
from rsna_dataloader import RSNAPneumoniaDataModule
from transforms_select import PREPROCESS_TF


from torchvision import transforms

from transforms_custom import (
    ContrastTransform,
    GammaTransform,
    SaltPepperNoiseTransform,
    SpeckleNoiseTransform,
    BlurTransform,
    SharpenTransform,
)

# Change 1. INTERVAL
INTERVAL = np.arange(1, 1.6, step=0.04)


# Change 2. Transform in function
def transform(param):
    # only for magnify
    return transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * param), antialias=True),
            transforms.CenterCrop(224),
        ]
    )
    # return transforms.Compose([PREPROCESS_TF, GammaTransform(param)])


CONFIG_PATH = "config.yml"

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    model = ResNetClassifier.load_from_checkpoint(
        configs["inference"]["checkpoint_path"]
    )
    model.eval()
    pl.seed_everything(33, workers=True)
    trainer = pl.Trainer(enable_progress_bar=False)

    df = pd.DataFrame(
        columns=[
            "Param",
            "ROC-AUC",
        ]
    )
    for param in INTERVAL:
        rsna = RSNAPneumoniaDataModule(
            configs,
            test_transforms=transform(param),
            print_stats=False,
        )

        roc = trainer.test(model=model, dataloaders=rsna.test_dataloader())[0][
            "test_roc-auc"
        ]
        res = {"Param": param, "ROC-AUC": roc}
        print(res)

        df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)

    print(df)
