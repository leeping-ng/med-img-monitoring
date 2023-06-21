"""
TO-DO for Monday:
1. Statistical test for confidence
2. Accumulate metrics & calculate accuracy of shift detection over runs:
    - This should be in a different dataframe
    - Transform, K-S signal (avg), K-S accuracy, Conf orig (avg), Conf shifted (avg), Chi-Sq signal (avg), Chi-Sq accuracy
3. Test on 10 runs
4. Implement other transformations
"""
import datetime
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import random
from scipy import stats

from config import load_config
from model import ResNetClassifier
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
    # logging.getLogger("pl.accelerators.cuda").setLevel(logging.WARNING)
    configs = load_config(CONFIG_PATH)
    model = ResNetClassifier.load_from_checkpoint(
        configs["inference"]["checkpoint_path"]
    )

    model.eval()

    rsna = RSNAPneumoniaDataModule(configs, test_transforms=PREPROCESS_TF)
    sample_size = configs["inference"]["sample_size"]
    num_classes = configs["model"]["num_classes"]
    pl.seed_everything(33, workers=True)
    trainer = pl.Trainer(enable_progress_bar=False)

    # df = pd.DataFrame(
    #     columns=[
    #         "Transform",
    #         "K-S cls0 stat",
    #         "K-S cls0 pval",
    #         "K-S cls1 stat",
    #         "K-S cls1 pval",
    #         "K-S signal",
    #         "K-S shift",
    #         "Conf orig",
    #         "Conf shifted",
    #         "Chi-Sq stat",
    #         "Chi-Sq pval",
    #         "Chi-Sq shift",
    #     ]
    # )

    df_tf = pd.DataFrame(
        columns=[
            "Transform",
            "K-S signal",
            "K-S pval",
            "K-S acc",
            "Chi-Sq signal",
            "Chi-Sq pval",
            "Chi-Sq acc",
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

        ks_signals = []
        ks_pvals = []
        ks_detected_shifts = 0
        chisq_signals = []
        chisq_pvals = []
        chisq_detected_shifts = 0
        chisq_errors = 0

        # loop over experiment runs with randomly shuffled images
        for run in range(NUM_RUNS):
            # these indices are for all the x images selected for a run (multiple batches to form x)
            img_indices = random.sample(
                range(0, sample_size),
                sample_size,
            )
            # print(img_indices)
            original_dataloader = rsna.predict_dataloader(img_indices, PREPROCESS_TF)
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
            ks_result = {}
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
                # print("Original:", original_batch["filename"])
                # print("Shifted:", shifted_batch["filename"])

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

            # K-S test
            ks_shift_detected = False
            ks_signal = 0
            for c in range(num_classes):
                ks_result[c] = stats.ks_2samp(
                    ks_original_softmaxes[c], ks_shifted_softmaxes[c]
                )
                # Reject null hypothesis if any p-value < Bonferroni corrected significance level
                if ks_result[c].pvalue < configs["inference"]["alpha"] / num_classes:
                    ks_shift_detected = True
                ks_signal += ks_result[c].statistic
            ks_signal /= num_classes

            # Confidence scores
            original_avg_conf = sum(original_confs) / len(original_confs)
            shifted_avg_conf = sum(shifted_confs) / len(shifted_confs)

            # Chi-Squared test labels
            original_counts = np.bincount(original_preds)
            shifted_counts = np.bincount(shifted_preds)
            try:
                chisq_stat, chisq_p = stats.chisquare(original_counts, shifted_counts)
                if chisq_p < configs["inference"]["alpha"]:
                    chisq_shift_detected = True
                else:
                    chisq_shift_detected = False
            except:
                chisq_stat, chisq_p, chisq_shift_detected = None, None, None

            # res = {
            #     "Transform": tf_name,
            #     "K-S cls0 stat": ks_result[0].statistic,
            #     "K-S cls0 pval": ks_result[0].pvalue,
            #     "K-S cls1 stat": ks_result[1].statistic,
            #     "K-S cls1 pval": ks_result[1].pvalue,
            #     "K-S signal": ks_signal,
            #     "K-S shift": ks_shift_detected,
            #     "Conf orig": original_avg_conf,
            #     "Conf shifted": shifted_avg_conf,
            #     "Chi-Sq stat": chisq_stat,
            #     "Chi-Sq pval": chisq_p,
            #     "Chi-Sq shift": chisq_shift_detected,
            # }

            ks_signals.append(ks_signal)
            ks_pvals.append((ks_result[0].pvalue + ks_result[1].pvalue) / 2)
            if chisq_shift_detected is not None:
                chisq_signals.append(chisq_stat)
                chisq_pvals.append(chisq_p)
            else:
                chisq_errors += 1

            if ks_shift_detected:
                ks_detected_shifts += 1
            if chisq_shift_detected is True:
                chisq_detected_shifts += 1

            # df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)

        res_tf = {
            "Transform": tf_name,
            "K-S signal": sum(ks_signals) / NUM_RUNS,
            "K-S pval": sum(ks_pvals) / NUM_RUNS,
            "K-S acc": ks_detected_shifts / NUM_RUNS,
            "Chi-Sq signal": None
            if NUM_RUNS - chisq_errors == 0
            else sum(chisq_signals) / (NUM_RUNS - chisq_errors),
            "Chi-Sq pval": None
            if NUM_RUNS - chisq_errors == 0
            else sum(chisq_pvals) / (NUM_RUNS - chisq_errors),
            "Chi-Sq acc": None
            if NUM_RUNS - chisq_errors == 0
            else chisq_detected_shifts / (NUM_RUNS - chisq_errors),
        }

        df_tf = pd.concat([df_tf, pd.DataFrame([res_tf])], ignore_index=True)
        print(df_tf)

    # print(df)
    print(df_tf)
    os.makedirs(configs["inference"]["result_folder"], exist_ok=True)
    time_now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    df_tf.to_csv(
        configs["inference"]["result_folder"] + "/" + time_now + ".csv", index=False
    )
