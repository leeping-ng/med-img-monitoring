# Adapted from: https://github.com/melanibe/failure_detection_benchmark/blob/main/data_handling/rsna_pneumonia.py

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor


class RNSAPneumoniaDetectionDataset(VisionDataset):
    def __init__(self, root, dataframe, transform):
        """
        Torchvision dataset for loading RSNA dataset.
        Args:
            root: the data directory where the images can be found
            dataframe: the csv file mapping patient id, metadata, file names and label.
            transform: the transformation (i.e. preprocessing and / or augmentation) to apply to the image after loading them.

        This dataset returns a dictionary with the image data, label and metadata.
        """
        super().__init__(root=root, transform=transform)
        self.root = Path(self.root)
        self.dataset_dataframe = dataframe
        self.targets = self.dataset_dataframe.label_rsna_pneumonia.values.astype(
            np.int64
        )
        self.subject_ids = self.dataset_dataframe.patientId.values
        self.filenames = [
            self.root / f"{subject_id}.png" for subject_id in self.subject_ids
        ]
        self.genders = self.dataset_dataframe["Patient Gender"].values
        self.ages = self.dataset_dataframe["Patient Age"].values

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        scan_image = imread(filename).astype(np.float32)
        # Added to convert from 1 channel to 3 channels of grayscale
        scan_image = np.repeat(scan_image[..., np.newaxis], 3, -1)
        return {
            "image": self.transform(scan_image),
            "target": self.targets[index],
            "gender": self.genders[index],
            "gender_label": 1 if self.genders[index] == "M" else 0,
            "age": self.ages[index],
        }

    def __len__(self) -> int:
        return len(self.filenames)


class RSNAPneumoniaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        val_split=0.1,
        random_seed_for_splits=33,
    ):
        """
        Pytorch Lightning DataModule defining train / val / test splits for the RSNA dataset.
        """
        super().__init__()
        self.image_data = config["data"]["image_folder"]
        self.csv_data = config["data"]["metadata_path"]
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = (
            train_transforms if train_transforms is not None else ToTensor()
        )
        self.val_transforms = (
            val_transforms if val_transforms is not None else ToTensor()
        )
        self.test_transforms = (
            test_transforms if test_transforms is not None else ToTensor()
        )
        self.shuffle = shuffle
        self.val_split = val_split
        # if not DATA_DIR_RSNA_PROCESSED_IMAGES.exists():
        #     f"Data dir: {DATA_DIR_RSNA_PROCESSED_IMAGES} does not exist. Have you updated default_paths.py?"

        df_with_all_labels = pd.read_csv(self.csv_data)

        print("DISTRIBUTION LABEL BY GENDER")
        print(
            pd.crosstab(
                columns=df_with_all_labels["label_rsna_pneumonia"],
                index=df_with_all_labels["Patient Gender"],
                normalize="index",
            )
        )

        # Use 85% of dataset for train / val and 15% for test
        indices_train_val, indices_test = train_test_split(
            np.arange(len(df_with_all_labels)),
            test_size=0.15,
            random_state=random_seed_for_splits,
        )

        # ***self.df_to_use has not been initialised***
        # train_val_df = self.df_to_use.iloc[indices_train_val]
        # test_df = self.df_to_use.iloc[indices_test]
        train_val_df = df_with_all_labels.iloc[indices_train_val]
        self.test_df = df_with_all_labels.iloc[indices_test]

        # Further split train and val
        indices_train, indices_val = train_test_split(
            np.arange(len(train_val_df)),
            test_size=self.val_split,
            random_state=random_seed_for_splits,
        )
        train_df = train_val_df.iloc[indices_train]
        val_df = train_val_df.iloc[indices_val]

        self.dataset_train = RNSAPneumoniaDetectionDataset(
            str(self.image_data),
            dataframe=train_df,
            transform=self.train_transforms,
        )
        self.dataset_val = RNSAPneumoniaDetectionDataset(
            str(self.image_data),
            dataframe=val_df,
            transform=self.val_transforms,
        )
        self.dataset_test = RNSAPneumoniaDetectionDataset(
            str(self.image_data),
            dataframe=self.test_df,
            transform=self.test_transforms,
        )
        # inference also uses test dataset & transforms
        self.dataset_predict = RNSAPneumoniaDetectionDataset(
            str(self.image_data),
            dataframe=self.test_df.iloc[: config["inference"]["num_images"]],
            transform=self.test_transforms,
        )

        print("#train: ", len(self.dataset_train))
        print("#val:   ", len(self.dataset_val))
        print("#test:  ", len(self.dataset_test))

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_predict,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def adj_predict_dataloader(self, new_transforms):
        self.dataset_predict = RNSAPneumoniaDetectionDataset(
            str(self.image_data),
            dataframe=self.test_df.iloc[: self.config["inference"]["num_images"]],
            transform=new_transforms,
        )
        return DataLoader(
            self.dataset_predict,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
