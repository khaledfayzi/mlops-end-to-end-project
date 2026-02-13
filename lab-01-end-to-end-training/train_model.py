"""
Train a ResNet-based regression model to predict fresh_weight_total from images.

Data expected locally (NOT committed):
lab-01-end-to-end-training/data/
  - digital_biomass_labels.xlsx
  - images_med_res/
"""

from pathlib import Path
import argparse
import logging
import subprocess
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# Utils
# -----------------------------
def get_git_commit_hash() -> str:
    """Get current git commit hash (best effort)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# -----------------------------
# Dataset
# -----------------------------
class BiomassImageDataset(Dataset):
    def __init__(self, meta_path: Path, images_dir: Path, transform=None):
        self.df = pd.read_excel(meta_path)

        # Keep only rows with filename + target
        self.df = self.df.dropna(subset=["filename", "fresh_weight_total"]).copy()

        # Convert filename to string
        self.df["filename"] = self.df["filename"].astype(str)

        self.images_dir = images_dir
        self.transform = transform

        # filter out rows where the image file is missing
        self.df["img_path"] = self.df["filename"].apply(lambda f: images_dir / f)
        self.df = self.df[self.df["img_path"].apply(lambda p: p.exists())].copy()
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        y = float(row["fresh_weight_total"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Regression target as tensor shape [1]
        y = torch.tensor([y], dtype=torch.float32)
        return img, y


# -----------------------------
# Model
# -----------------------------
def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)  # regression head
        return m
    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
        return m
    else:
        raise ValueError("model_name must be 'resnet18' or 'resnet50'")


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def train_one_epoch(model, loader, device, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    images_dir = data_dir / "images_med_res"
    meta_path = data_dir / "digital_biomass_labels.xlsx"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Logging
    logger = setup_logger(base_dir / "training.log")
    logger.info("Starting training")
    logger.info(f"Git commit: {get_git_commit_hash()}")
    logger.info(f"Args: {vars(args)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Transform (ResNet expects normalized images)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset + split
    dataset = BiomassImageDataset(meta_path=meta_path, images_dir=images_dir, transform=tfm)
    logger.info(f"Dataset size (valid image + target): {len(dataset)}")

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_model(args.model_name)

    if args.freeze_backbone:
        # Freeze all layers except final fc
        for name, p in model.named_parameters():
            p.requires_grad = ("fc" in name)
        logger.info("Backbone frozen: training only final head (fc)")

    model.to(device)

    # Loss + optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate)

    # Training loop
    best_val = float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, device, loss_fn)

        logger.info(f"Epoch {epoch}/{args.epochs} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = results_dir / f"best_{args.model_name}.pt"
            torch.save({
                "model_name": args.model_name,
                "model_state_dict": model.state_dict(),
                "img_size": args.img_size,
                "val_mse": best_val,
                "args": vars(args),
            }, ckpt_path)
            logger.info(f"Saved best model to: {ckpt_path}")

    elapsed = time.time() - start
    logger.info(f"Done. Best val MSE: {best_val:.6f} | Time: {elapsed:.1f}s")

    # Save a tiny text summary
    (results_dir / "metrics.txt").write_text(f"best_val_mse={best_val}\n")


if __name__ == "__main__":
    main()
