import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
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
    sample_size = configs["inference"]["sample_size"]
    num_classes = configs["model"]["num_classes"]
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

    transforms = [
        transforms.Compose([transforms.ToTensor(), ContrastTransform(0.8)]),
        transforms.Compose([transforms.ToTensor(), ContrastTransform(0.3)]),
    ]

    # loop over different transforms
    for transform in transforms:
        # fix sequence of randomness so that sequence of images for each transform are the same
        random.seed(0)

        # loop over experiment runs with randomly shuffled images
        for run in range(1):
            # these indices are for all the x images selected for a run (multiple batches to form x)
            img_indices = random.sample(
                range(0, sample_size),
                sample_size,
            )
            # print(img_indices)
            original_dataloader = rsna.predict_dataloader(
                img_indices, preprocess_transforms
            )
            shifted_dataloader = rsna.predict_dataloader(img_indices, transform)
            original_output = trainer.predict(
                model=model, dataloaders=original_dataloader
            )

            shifted_output = trainer.predict(
                model=model, dataloaders=shifted_dataloader
            )

            # initialise empty data structures to be extended over batches
            ks_original_softmaxes = {}
            ks_shifted_softmaxes = {}
            for c in range(num_classes):
                ks_original_softmaxes[c] = []
                ks_shifted_softmaxes[c] = []
            original_confs = []
            shifted_confs = []
            original_preds = []
            shifted_preds = []

            # loop over batches to aggregate softmax,
            for i, (original_batch, shifted_batch) in enumerate(
                zip(original_output, shifted_output)
            ):
                # print(
                #     "*************************** Batch: ", i, " ***********************"
                # )
                # print(original_batch["filename"])
                # print(shifted_batch["filename"])

                original_softmax = original_batch["softmax"].numpy()
                shifted_softmax = shifted_batch["softmax"].numpy()

                # K-S test
                for c in range(num_classes):
                    ks_original_softmaxes[c].extend(
                        list(original_softmax[:, c].squeeze())
                    )
                    ks_shifted_softmaxes[c].extend(
                        list(shifted_softmax[:, c].squeeze())
                    )

                # Confidence scores
                original_confs.extend(list(np.amax(original_softmax, axis=1)))
                shifted_confs.extend(list(np.amax(shifted_softmax, axis=1)))

                # Chi-Squared test labels
                original_preds.extend(list(original_batch["preds"].numpy()))
                shifted_preds.extend(list(shifted_batch["preds"].numpy()))

            # for c in range(num_classes):
            #     print(stats.ks_2samp(ks_original_softmaxes[c], ks_shifted_softmaxes[c]))

            test_class_0 = stats.ks_2samp(
                ks_original_softmaxes[0], ks_shifted_softmaxes[0]
            )
            test_class_1 = stats.ks_2samp(
                ks_original_softmaxes[1], ks_shifted_softmaxes[1]
            )

            original_avg_conf = sum(original_confs) / len(original_confs)
            shifted_avg_conf = sum(shifted_confs) / len(shifted_confs)

            original_counts = np.bincount(original_preds)
            shifted_counts = np.bincount(shifted_preds)
            try:
                chisq, p = stats.chisquare(original_counts, shifted_counts)
            except:
                chisq, p = None, None

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
