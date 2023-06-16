import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scipy import stats
from torchvision import transforms

from config import load_config
from model import ResNetClassifier
from rsna_dataloader import RSNAPneumoniaDataModule
from transforms import preprocess_transforms, ContrastTransform


CONFIG_PATH = "config.yml"

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    model = ResNetClassifier.load_from_checkpoint(
        configs["inference"]["checkpoint_path"]
    )

    model.eval()

    rsna = RSNAPneumoniaDataModule(configs, test_transforms=preprocess_transforms)
    pl.seed_everything(33, workers=True)
    trainer = pl.Trainer()

    df = pd.DataFrame(
        columns=[
            "K-S class 0 statistic",
            "K-S class 0 pvalue",
            "K-S class 1 statistic",
            "K-S class 1 pvalue",
            "Confidence orig",
            "Confidence shifted",
            "Chi-Squared statistic",
            "Chi-Squared pvalue",
        ]
    )

    output = trainer.predict(model=model, dataloaders=rsna.predict_dataloader())
    batch = next(iter(output))

    transforms = [
        transforms.Compose([transforms.ToTensor(), ContrastTransform(0.8)]),
        transforms.Compose([transforms.ToTensor(), ContrastTransform(0.6)]),
    ]

    for transform in transforms:
        # another loop here for multiple runs per transform
        dataloader = rsna.adj_predict_dataloader(transform)
        shifted_output = trainer.predict(model=model, dataloaders=dataloader)
        shifted_batch = next(iter(shifted_output))

        original_softmax = batch["softmax"].numpy()
        shifted_softmax = shifted_batch["softmax"].numpy()
        # print(original_softmax)
        # print(shifted_softmax)

        # K-S test
        # Need to make agnostic to number of classes
        original_class_0 = original_softmax[:, 0:1].squeeze()
        original_class_1 = original_softmax[:, 1:2].squeeze()
        shifted_class_0 = shifted_softmax[:, 0:1].squeeze()
        shifted_class_1 = shifted_softmax[:, 1:2].squeeze()
        test_class_0 = stats.ks_2samp(original_class_0, shifted_class_0)
        test_class_1 = stats.ks_2samp(original_class_1, shifted_class_1)

        # Confidence score
        original_confs = np.amax(original_softmax, axis=1)
        original_avg_conf = np.average(original_confs)
        shifted_confs = np.amax(shifted_softmax, axis=1)
        shifted_avg_conf = np.average(shifted_confs)

        # Chi-Squared test
        original_preds = batch["preds"].numpy()
        shifted_preds = shifted_batch["preds"].numpy()
        original_counts = np.bincount(original_preds)
        shifted_counts = np.bincount(shifted_preds)
        chisq, p = stats.chisquare(original_counts, shifted_counts)

        res = {
            "K-S class 0 statistic": test_class_0.statistic,
            "K-S class 0 pvalue": test_class_0.pvalue,
            "K-S class 1 statistic": test_class_1.statistic,
            "K-S class 1 pvalue": test_class_1.pvalue,
            "Confidence orig": original_avg_conf,
            "Confidence shifted": shifted_avg_conf,
            "Chi-Squared statistic": chisq,
            "Chi-Squared pvalue": p,
        }
        df = pd.concat([df, pd.DataFrame([res])])

    print(df)
