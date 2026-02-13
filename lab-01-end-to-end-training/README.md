# Lab 1 – End-to-End Biomass Prediction (EDA + Training)

## Dataset
- Plant images (256x256) + Excel metadata
- Target: `fresh_weight_total` (Regression)

## EDA (Exploratory Data Analysis)
- Target distribution
- Sample images with labels
- Correlation heatmap (numeric features vs target)
- Age vs biomass
- Image pixel analysis

## Data Quality Issues
- Missing target values (viele NaNs in fresh_weight_total)
- Teilweise fehlende/inkonsistente Metadaten
- Potenzielles Data Leakage (mehrere Bilder pro Pflanze)

## Model
- Pretrained ResNet18/ResNet50
- Regression head (1 Output)
- Loss: MSE
- Optimizer: Adam

## Training
```bash
python train_model.py --epochs 5 --batch_size 16 --learning_rate 0.0001 --model_name resnet18
