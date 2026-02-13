"""
EDA (Exploratory Data Analysis) – Lab 1

Plot 1: Target distribution (fresh_weight_total)
Plot 2: Sample images grid (3x3) with biomass labels

Data expected at:
lab-01-end-to-end-training/data/
  - digital_biomass_labels.xlsx
  - images_med_res/
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# 1) Paths (where files are)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images_med_res"
META_PATH = DATA_DIR / "digital_biomass_labels.xlsx"

# Folder for plots (we save png files here)
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# -----------------------------
# 2) Load metadata (Excel)
# -----------------------------
df = pd.read_excel(META_PATH)

# Quick sanity checks (helps to understand the dataset quickly)
print("Head:")
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())

print("\nTarget stats (fresh_weight_total):")
print(df["fresh_weight_total"].describe())


# -----------------------------
# 3) Plot 1: Target distribution
# -----------------------------
# We remove NaNs (missing targets) before plotting
target = df["fresh_weight_total"].dropna()

plt.figure(figsize=(6, 4))
plt.hist(target, bins=30)
plt.title("Target Distribution: fresh_weight_total")
plt.xlabel("fresh_weight_total (grams)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "target_distribution.png")
plt.close()

print("\nSaved: figures/target_distribution.png")


# -----------------------------
# 4) Plot 2: Sample images grid
# -----------------------------
# Use only rows that have BOTH:
# - a filename (image exists)
# - a target value (fresh_weight_total not NaN)
df_valid = df.dropna(subset=["filename", "fresh_weight_total"]).copy()
print("Valid rows (filename + target):", len(df_valid))

# Sample 9 images reproducibly (always same result with same random_state)
samples = df_valid.sample(n=9, random_state=42)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.flatten()

for ax, (_, row) in zip(axes, samples.iterrows()):
    img_path = IMAGES_DIR / str(row["filename"])

    # Safety check: if an image is missing, show a placeholder text
    if not img_path.exists():
        ax.text(0.5, 0.5, "Missing image", ha="center", va="center")
        ax.set_title("N/A")
        ax.axis("off")
        continue

    # Load image and show it
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)

    # Title shows the biomass label (target)
    ax.set_title(f"{row['fresh_weight_total']:.3f} g")
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "sample_images.png")
plt.close()

print("Saved: figures/sample_images.png")
