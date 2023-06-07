from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from skimage.io import imread


# in my default_paths.py file:
# DATA_DIR_RSNA = "/vol/biodata/data/chest_xray/rsna-pneumonia-detection-challenge"
# DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"
# PATH_TO_PNEUMONIA_WITH_METADATA_CSV = "pneumonia_dataset_with_metadata.csv"
# The original dataset can be found at https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
# This dataset is originally a (relabelled) subset of the NIH dataset https://www.kaggle.com/datasets/nih-chest-xrays/data from
# which i took the metadata.


from default_paths import DATA_DIR_RSNA, DATA_DIR_RSNA_PROCESSED_IMAGES, PATH_TO_PNEUMONIA_WITH_METADATA_CSV


class RNSAPneumoniaDetectionDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        dataframe: pd.DataFrame,
        transform: Callable,
    ) -> None:
        """
        Torchvision dataset for loading RSNA dataset.
        Args:
            root: the data directory where the images can be found
            dataframe: the csv file mapping patient id, metadata, file names and label.
            transform: the transformation (i.e. preprocessing and / or augmentation) to apply to the image after loading them.

        This dataset returns a dictionary with the image data, label and metadata. 
        """
        super().__init__(root=root, transform=transform)
        self.root = Path(self.root)  # type: ignore
        self.dataset_dataframe = dataframe
        self.targets = self.dataset_dataframe.label_rsna_pneumonia.values.astype(np.int64)
        self.subject_ids = self.dataset_dataframe.patientId.values
        self.filenames = [self.root / f"{subject_id}.png" for subject_id in self.subject_ids]
        self.genders = self.dataset_dataframe["Patient Gender"].values
        self.ages = self.dataset_dataframe["Patient Age"].values

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        scan_image = imread(filename).astype(np.float32)
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
        train_transforms=None,
        val_transforms=None,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        val_split=0.1,
        random_seed_for_splits = 33
    ): 
        """
        Pytorch Lightning DataModule defining train / val / test splits for the RSNA dataset. 
        """
        super().__init__()
        self.root_dir = DATA_DIR_RSNA
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms if train_transforms is not None else ToTensor()
        self.val_transforms = val_transforms if val_transforms is not None else ToTensor()
        self.shuffle = shuffle
        self.val_split = val_split
        if not DATA_DIR_RSNA_PROCESSED_IMAGES.exists():
            f"Data dir: {DATA_DIR_RSNA_PROCESSED_IMAGES} does not exist. Have you updated default_paths.py?"

        df_with_all_labels = pd.read_csv(PATH_TO_PNEUMONIA_WITH_METADATA_CSV)

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
            np.arange(len(df_with_all_labels)), test_size=0.15, random_state=random_seed_for_splits
        )
        train_val_df = self.df_to_use.iloc[indices_train_val]
        test_df = self.df_to_use.iloc[indices_test]
        
        # Further split train and val
        indices_train, indices_val = train_test_split(
            np.arange(len(train_val_df)), test_size=self.val_split, random_state=random_seed_for_splits
        )
        train_df = train_val_df.iloc[indices_train]
        val_df = train_val_df.iloc[indices_val]

        self.dataset_train = RNSAPneumoniaDetectionDataset(
            str(DATA_DIR_RSNA_PROCESSED_IMAGES),
            dataframe=train_df,
            transform=self.train_transforms,
        )
        self.dataset_val = RNSAPneumoniaDetectionDataset(
            str(DATA_DIR_RSNA_PROCESSED_IMAGES),
            dataframe=val_df,
            transform=self.val_transforms,
        )
        self.dataset_test = RNSAPneumoniaDetectionDataset(
            str(DATA_DIR_RSNA_PROCESSED_IMAGES),
            dataframe=test_df,
            transform=self.val_transforms,
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


