from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def train_val_split(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42):
    """
    Shuffle the dataframe and split into train and validation sets.
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:].copy()
    return train_df, val_df


class BiomassImageDataset(Dataset):
    """
    PyTorch Dataset for plant biomass regression.
    Loads images from disk and returns (image, target) pairs.
    """

    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        img_path = self.images_dir / filename

        # target (regression)
        y = float(row["fresh_weight_total"])

        # load image
        img = Image.open(img_path).convert("RGB")

        # apply transforms if provided
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([y], dtype=torch.float32)
