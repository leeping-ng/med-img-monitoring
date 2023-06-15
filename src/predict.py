import numpy as np
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

    output = trainer.predict(model=model, dataloaders=rsna.predict_dataloader())
    batch = next(iter(output))
    # print(batch["y"])
    # print(batch["preds"])
    # print(batch["softmax"])

    transform_c120 = transforms.Compose([transforms.ToTensor(), ContrastTransform(0.8)])
    dataloader_c120 = rsna.adj_predict_dataloader(transform_c120)
    shifted_output = trainer.predict(model=model, dataloaders=dataloader_c120)
    shifted_batch = next(iter(shifted_output))
    # print(shifted_batch["y"])
    # print(shifted_batch["preds"])
    # print(shifted_batch["softmax"])

    original_softmax = batch["softmax"].numpy()
    shifted_softmax = shifted_batch["softmax"].numpy()
    # print(original_softmax)
    # print(shifted_softmax)
    original_class_0 = original_softmax[:, 0:1].squeeze()
    original_class_1 = original_softmax[:, 1:2].squeeze()
    # print(original_class_0)
    # print(original_class_1)

    shifted_class_0 = shifted_softmax[:, 0:1].squeeze()
    shifted_class_1 = shifted_softmax[:, 1:2].squeeze()
    # print(shifted_class_0)
    # print(shifted_class_1)

    print("*************************************")
    print("K-S Test Results")
    print("No shift: ", stats.ks_2samp(original_class_0, original_class_0))
    # WHY ARE CLASS 0 AND 1 RESULTS THE SAME???
    print("Class 0: ", stats.ks_2samp(original_class_0, shifted_class_0))
    print("Class 1: ", stats.ks_2samp(original_class_1, shifted_class_1))

    original_confs = np.amax(original_softmax, axis=1)
    original_avg_conf = np.average(original_confs)
    shifted_confs = np.amax(shifted_softmax, axis=1)
    shifted_avg_conf = np.average(shifted_confs)

    print("*************************************")
    print("Confidence Score Results")
    print("Average max confidence for original: ", original_avg_conf)
    print("Average max confidence for shifted: ", shifted_avg_conf)

    original_preds = batch["preds"].numpy()
    shifted_preds = shifted_batch["preds"].numpy()
    original_counts = np.bincount(original_preds)
    shifted_counts = np.bincount(shifted_preds)
    # print(original_preds)
    # print(original_counts)
    # print(shifted_preds)
    # print(shifted_counts)

    print("*************************************")
    print("Chi-Squared Labels Results")
    print(stats.chisquare(original_counts, shifted_counts))
