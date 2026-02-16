"""
Dagster pipeline (Lab 2): data loading -> preprocessing -> training (MLflow) -> evaluation.

"""

from pathlib import Path
import math  
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mlflow

from torch.utils.data import DataLoader
from torchvision import transforms, models

from dagster import asset, AssetExecutionContext, Definitions
from dagster_mlflow import mlflow_tracking
from dataset import BiomassImageDataset, train_val_split

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "lab-01-end-to-end-training" / "data"
META_PATH = DATA_DIR / "digital_biomass_labels.xlsx"
IMAGES_DIR = DATA_DIR / "images_med_res"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@asset
def raw_dataset(context: AssetExecutionContext):
    context.log.info(f"Reading: {META_PATH}")
    df = pd.read_excel(META_PATH)

    df = df.dropna(subset=["filename", "fresh_weight_total"]).copy()
    df["filename"] = df["filename"].astype(str)

    exists_mask = df["filename"].apply(lambda f: (IMAGES_DIR / f).exists())
    missing = int((~exists_mask).sum())
    if missing > 0:
        context.log.warning(f"{missing} rows dropped because image file is missing.")
        df = df[exists_mask].copy()

    context.log.info(f"Rows after cleaning: {len(df)}")
    return {"metadata": df, "images_dir": str(IMAGES_DIR)}



@asset
def eda_plots(context: AssetExecutionContext, raw_dataset):
   
    df = raw_dataset["metadata"]

    plt.figure()
    df["fresh_weight_total"].hist(bins=30)
    plt.xlabel("fresh_weight_total")
    plt.ylabel("count")

    out = RESULTS_DIR / "eda_target_hist.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    context.log.info(f"Saved EDA plot: {out}")
    return {"eda_hist_path": str(out)}



@asset(config_schema={"batch_size": int})  
def preprocessed_data(context: AssetExecutionContext, raw_dataset):
    df = raw_dataset["metadata"]
    images_dir = Path(raw_dataset["images_dir"])

    train_df, val_df = train_val_split(df, train_frac=0.8, seed=42)
    context.log.info(f"Train size: {len(train_df)} | Val size: {len(val_df)}")


    batch_size = context.op_config["batch_size"]
    context.log.info(f"Using batch_size={batch_size}")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = BiomassImageDataset(train_df, images_dir, transform=tfm)
    val_ds   = BiomassImageDataset(val_df, images_dir, transform=tfm)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return {"train_loader": train_loader, "val_loader": val_loader}



@asset(required_resource_keys={"mlflow"}, config_schema={"epochs": int, "lr": float})  
def trained_model(context: AssetExecutionContext, preprocessed_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    context.log.info(f"Using device: {device}")

    train_loader = preprocessed_data["train_loader"]
    val_loader   = preprocessed_data["val_loader"]


    epochs = context.op_config["epochs"]
    lr = context.op_config["lr"]
    context.log.info(f"Using epochs={epochs}, lr={lr}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

    history = {
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],   
        "val_rmse": [],  
    }

    # MLflow: Parameter loggen
    mlflow.log_param("model_name", "resnet18")
    mlflow.log_param("batch_size", train_loader.batch_size)  
    mlflow.log_param("lr", lr)                               
    mlflow.log_param("epochs", epochs)                       

    
    for epoch in range(epochs):

        model.train()
        train_sum, n_train = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_sum += loss.item()
            n_train += 1

        train_mse = train_sum / max(n_train, 1)


        model.eval()


        val_mse_sum, val_mae_sum, n_val = 0.0, 0.0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)


                mse = nn.functional.mse_loss(pred, y).item()
                mae = nn.functional.l1_loss(pred, y).item()

                val_mse_sum += mse
                val_mae_sum += mae
                n_val += 1

        val_mse = val_mse_sum / max(n_val, 1)
        val_mae = val_mae_sum / max(n_val, 1)
        val_rmse = math.sqrt(val_mse)  

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)     
        history["val_rmse"].append(val_rmse)   


        context.log.info(
            f"Epoch {epoch+1}/{epochs}: train_mse={train_mse:.4f}, val_mse={val_mse:.4f}, "
            f"val_mae={val_mae:.4f}, val_rmse={val_rmse:.4f}"
        )

        # MLflow Metriken loggen
        mlflow.log_metric("train_mse", train_mse, step=epoch + 1)
        mlflow.log_metric("val_mse", val_mse, step=epoch + 1)
        mlflow.log_metric("val_mae", val_mae, step=epoch + 1)     
        mlflow.log_metric("val_rmse", val_rmse, step=epoch + 1)   

    model_path = RESULTS_DIR / "model.pt"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(str(model_path))
    context.log.info(f"Saved model: {model_path}")

    return {"model_path": str(model_path), "history": history}


@asset
def model_evaluation(context: AssetExecutionContext, trained_model, preprocessed_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_loader = preprocessed_data["val_loader"]

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(trained_model["model_path"], map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    val_sum, n_val = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_sum += criterion(pred, y).item()
            n_val += 1
    final_val_mse = val_sum / max(n_val, 1)

    # Plot curves aus history
    history = trained_model["history"]
    epochs = list(range(1, len(history["train_mse"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_mse"], label="train_mse")
    plt.plot(epochs, history["val_mse"], label="val_mse")
    plt.plot(epochs, history["val_mae"], label="val_mae")     
    plt.plot(epochs, history["val_rmse"], label="val_rmse")   
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()

    curves_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close()

    metrics_path = RESULTS_DIR / "metrics.txt"
    metrics_path.write_text(f"final_val_mse: {final_val_mse}\n")

    context.log.info(f"Saved plot: {curves_path}")
    context.log.info(f"Saved metrics: {metrics_path}")

    return {
        "final_val_mse": final_val_mse,
        "curves_path": str(curves_path),
        "metrics_path": str(metrics_path),
        "model_path": trained_model["model_path"],
    }


defs = Definitions(

    assets=[raw_dataset, eda_plots, preprocessed_data, trained_model, model_evaluation],
    resources={
        "mlflow": mlflow_tracking.configured({
            "experiment_name": "plant_biomass_pipeline",
            "mlflow_tracking_uri": f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}",
        })
    },
)
