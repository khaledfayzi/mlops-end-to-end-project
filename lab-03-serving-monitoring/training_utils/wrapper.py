import mlflow.pyfunc
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from io import BytesIO

class BiomassPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        #model laden
        device= "cuda" if torch.cuda.is_available() else "cpu"
        self.device=device

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)

        # trainiert model also gewicht holen
        state_dict = torch.load(context.artifacts["weights"], map_location=device)

        # die Gewichte in das model schriben
        model.load_state_dict(state_dict)

        #schiebt das model auf GPU oder CPU
        model.to (device)

        #test modus aktiviern
        model.eval()

        #Speichert das fertige Modell in der Klasse für predict()
        self.model = model

        # 2) Preprocessing wie im Training
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    
    def predict(self, context, model_input):

        if isinstance(model_input, pd.DataFrame):
            items = model_input.iloc[:, 0].tolist()
        else:
            items = list(model_input)

        preds = []
        with torch.no_grad():
            for item in items:
                # Fall 1: bytes (Gradio sendet bytes)
                if isinstance(item, (bytes, bytearray)):
                    img = Image.open(BytesIO(item)).convert("RGB")
                # Fall 2: Dateipfad (falls du später Pfade nutzt)
                else:
                    img = Image.open(item).convert("RGB")
    
                x = self.tfm(img).unsqueeze(0).to(self.device)
                y = self.model(x).squeeze().item()
                preds.append(float(y))
        return preds






