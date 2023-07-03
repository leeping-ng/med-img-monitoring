from alibi_detect.cd import MMDDrift
import datetime
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import random
from scipy import stats

from config import load_config
from model import ResNetClassifier, UntrainedAutoEncoder
from rsna_dataloader import RSNAPneumoniaDataModule
from transforms_select import (
    PREPROCESS_TF,
    BLUR_TF,
    SHARPEN_TF,
    SALT_PEPPER_NOISE_TF,
    SPECKLE_NOISE_TF,
    CONTRAST_INC_TF,
    CONTRAST_DEC_TF,
    GAMMA_INC_TF,
    GAMMA_DEC_TF,
    MAGNIFY_TF,
)


CONFIG_PATH = "config.yml"
NUM_RUNS = 10

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    resnet_model = ResNetClassifier.load_from_checkpoint(
        configs["inference"]["checkpoint_path"]
    )
    resnet_model.eval()

    ue_model = UntrainedAutoEncoder(configs)
    ue_model.eval()

    rsna = RSNAPneumoniaDataModule(configs, test_transforms=PREPROCESS_TF)
    test_dataset_size = configs["inference"]["test_dataset_size"]
    sample_size = configs["inference"]["sample_size"]
    num_classes = configs["model"]["num_classes"]
    pl.seed_everything(33, workers=True)
    trainer = pl.Trainer(enable_progress_bar=False)

    df_tf = pd.DataFrame(
        columns=[
            "Transform",
            "K-S SM signal",
            "K-S SM pval",
            "K-S SM acc",
            "MMD UE signal",
            "MMD UE pval",
            "MMD UE acc",
            "MMD TE signal",
            "MMD TE pval",
            "MMD TE acc",
            "ROC-AUC",
        ]
    )

    all_tf = (
        BLUR_TF
        | SHARPEN_TF
        | SALT_PEPPER_NOISE_TF
        | SPECKLE_NOISE_TF
        | CONTRAST_INC_TF
        | CONTRAST_DEC_TF
        | GAMMA_INC_TF
        | GAMMA_DEC_TF
        | MAGNIFY_TF
    )

    # loop over different transforms
    for tf_name, transform in all_tf.items():
        print("Working on transform:", tf_name, "...")
        # fix sequence of randomness so that sequence of images for each transform are the same
        random.seed(0)

        ks_sm_signals = []
        ks_sm_pvals = []
        ks_sm_detected_shifts = 0
        mmd_ue_signals = []
        mmd_ue_pvals = []
        mmd_ue_detected_shifts = 0
        mmd_te_signals = []
        mmd_te_pvals = []
        mmd_te_detected_shifts = 0
        rocs = []

        # loop over experiment runs with randomly shuffled images
        for run in range(NUM_RUNS):
            # these indices are for all the x images selected for a run (multiple batches to form x)
            img_indices = random.sample(
                range(0, test_dataset_size),
                sample_size,
            )
            # print("Run", run, ":", img_indices)
            original_dataloader = rsna.predict_dataloader(img_indices, PREPROCESS_TF)
            shifted_dataloader = rsna.predict_dataloader(img_indices, transform)

            # for K-S and Chi-Sq test
            original_resnet_output = trainer.predict(
                model=resnet_model, dataloaders=original_dataloader
            )
            shifted_resnet_output = trainer.predict(
                model=resnet_model, dataloaders=shifted_dataloader
            )

            # for MMD test on untrained embeddings
            original_ue_output = trainer.predict(
                model=ue_model, dataloaders=original_dataloader
            )
            shifted_ue_output = trainer.predict(
                model=ue_model, dataloaders=shifted_dataloader
            )

            roc = trainer.test(model=resnet_model, dataloaders=shifted_dataloader)[0][
                "test_roc-auc"
            ]

            # initialise empty data structures to be extended over batches
            ks_sm_original_softmaxes = {}
            ks_sm_shifted_softmaxes = {}
            ks_sm_result = {}
            for c in range(num_classes):
                ks_sm_original_softmaxes[c] = []
                ks_sm_shifted_softmaxes[c] = []
            original_ue_embedding = []
            shifted_ue_embedding = []
            original_te_embedding = []
            shifted_te_embedding = []

            # loop over batches to aggregate softmax and embeddings
            for i, (original_batch, shifted_batch) in enumerate(
                zip(original_resnet_output, shifted_resnet_output)
            ):
                original_softmax = original_batch["softmax"].numpy()
                shifted_softmax = shifted_batch["softmax"].numpy()

                # K-S taking in softmax
                for c in range(num_classes):
                    ks_sm_original_softmaxes[c].extend(
                        list(original_softmax[:, c].squeeze())
                    )
                    ks_sm_shifted_softmaxes[c].extend(
                        list(shifted_softmax[:, c].squeeze())
                    )

                original_te_embedding.extend(original_batch["embeddings"].numpy())
                shifted_te_embedding.extend(shifted_batch["embeddings"].numpy())

            # loop over batches to aggregate untrained embeddings
            for i, (original_batch, shifted_batch) in enumerate(
                zip(original_ue_output, shifted_ue_output)
            ):
                original_ue_embedding.extend(original_batch.numpy())
                shifted_ue_embedding.extend(shifted_batch.numpy())

            # ROC-AUC
            rocs.append(roc)

            # K-S taking in softmax
            ks_sm_shift_detected = False
            ks_sm_signal = 0
            for c in range(num_classes):
                ks_sm_result[c] = stats.ks_2samp(
                    ks_sm_original_softmaxes[c], ks_sm_shifted_softmaxes[c]
                )
                # Reject null hypothesis if any p-value < Bonferroni corrected significance level
                if ks_sm_result[c].pvalue < configs["inference"]["alpha"] / num_classes:
                    ks_sm_shift_detected = True
                ks_sm_signal += ks_sm_result[c].statistic
            ks_sm_signal /= num_classes
            ks_sm_signals.append(ks_sm_signal)
            ks_sm_pvals.append((ks_sm_result[0].pvalue + ks_sm_result[1].pvalue) / 2)

            if ks_sm_shift_detected:
                ks_sm_detected_shifts += 1

            # MMD taking in untrained embeddings
            mmd_ue_model = MMDDrift(
                np.array(original_ue_embedding),
                backend="pytorch",
                p_val=configs["inference"]["alpha"],
                n_permutations=100,
            )
            mmd_ue_preds = mmd_ue_model.predict(np.array(shifted_ue_embedding))
            mmd_ue_signals.append(mmd_ue_preds["data"]["distance"])
            mmd_ue_pvals.append(mmd_ue_preds["data"]["p_val"])
            if mmd_ue_preds["data"]["is_drift"]:
                mmd_ue_detected_shifts += 1

            # MMD taking in trained embeddings from ResNet
            mmd_te_model = MMDDrift(
                np.array(original_te_embedding),
                backend="pytorch",
                p_val=configs["inference"]["alpha"],
                n_permutations=100,
            )
            mmd_te_preds = mmd_te_model.predict(np.array(shifted_te_embedding))
            mmd_te_signals.append(mmd_te_preds["data"]["distance"])
            mmd_te_pvals.append(mmd_te_preds["data"]["p_val"])
            if mmd_te_preds["data"]["is_drift"]:
                mmd_te_detected_shifts += 1

        res_tf = {
            "Transform": tf_name,
            "K-S SM signal": sum(ks_sm_signals) / NUM_RUNS,
            "K-S SM pval": sum(ks_sm_pvals) / NUM_RUNS,
            "K-S SM acc": ks_sm_detected_shifts / NUM_RUNS,
            "MMD UE signal": sum(mmd_ue_signals) / NUM_RUNS,
            "MMD UE pval": sum(mmd_ue_pvals) / NUM_RUNS,
            "MMD UE acc": mmd_ue_detected_shifts / NUM_RUNS,
            "MMD TE signal": sum(mmd_te_signals) / NUM_RUNS,
            "MMD TE pval": sum(mmd_te_pvals) / NUM_RUNS,
            "MMD TE acc": mmd_te_detected_shifts / NUM_RUNS,
            "ROC-AUC": sum(rocs) / NUM_RUNS,
        }

        df_tf = pd.concat([df_tf, pd.DataFrame([res_tf])], ignore_index=True)

    print(df_tf)
    os.makedirs(configs["inference"]["result_folder"], exist_ok=True)
    time_now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    df_tf.to_csv(
        configs["inference"]["result_folder"]
        + "/tf_signal_vs_rocauc_"
        + time_now
        + ".csv",
        index=False,
    )
