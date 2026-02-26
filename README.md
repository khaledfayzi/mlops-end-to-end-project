# 🚀 End-to-End MLOps Projekt: Training, Orchestrierung, Serving & Monitoring

Dieses Repository zeigt ein vollständiges End-to-End-MLOps-Projekt für eine bildbasierte Regressionsaufgabe (Vorhersage der Pflanzen-Biomasse).  
Es deckt den gesamten ML-Lebenszyklus ab – von explorativer Datenanalyse (EDA) und Modelltraining über Pipeline-Orchestrierung bis hin zu Modell-Serving und Monitoring mit Data-Drift-Erkennung.

Fokus: Produktionsnahe ML-Entwicklung & MLOps Best Practices  
Tech-Stack: PyTorch, Dagster, MLflow, Gradio, Evidently

---

## 📌 Projektstruktur

mlops-end-to-end-project/
├── data/                         # Zentraler Datenordner (Bilder + Labels)
├── lab-01-end-to-end-training/   # EDA + Modelltraining (PyTorch)
├── lab-02-ml-pipeline-dagster/   # Orchestrierte ML-Pipeline (Dagster + MLflow)
├── lab-03-serving-monitoring/    # Modell-Serving + Drift-Monitoring
├── requirements.txt
└── README.md

---

## 🧠 Problemstellung

Basierend auf Pflanzenbildern soll die frische Biomasse (Regression) vorhergesagt werden.  
Der Fokus des Projekts liegt nicht nur auf der Modellperformance, sondern auf dem Aufbau eines robusten, reproduzierbaren und produktionsnahen MLOps-Systems.

---

## 🏗️ Architektur (High-Level)

Daten (Bilder + Labels)  
→ EDA & Training (PyTorch)  
→ Orchestrierung (Dagster)  
→ Experiment-Tracking (MLflow)  
→ Serving (Gradio App)  
→ Monitoring & Drift-Erkennung (Evidently)

---

## 🧪 Lab 1 – End-to-End Training

Pfad: lab-01-end-to-end-training/

Inhalte:
- Explorative Datenanalyse (EDA)
- ResNet-basiertes Regressionsmodell (PyTorch)
- CLI-konfigurierbare Skripte (--data_dir)
- Reproduzierbare Trainingspipeline

Ausführen:
python lab-01-end-to-end-training/eda.py --data_dir data  
python lab-01-end-to-end-training/train_model.py --data_dir data

---

## 🧪 Lab 2 – ML-Pipeline Orchestrierung (Dagster + MLflow)

Pfad: lab-02-ml-pipeline-dagster/

Inhalte:
- Asset-basierte ML-Pipeline mit Dagster
- Parametrisierte Pipeline-Konfiguration
- Experiment-Tracking mit MLflow (Parameter, Metriken, Artefakte)
- Trainingskurven & Evaluationsartefakte

Ausführen:
dagster dev -f lab-02-ml-pipeline-dagster/dagster_pipeline.py

Dagster UI: http://localhost:3000  
MLflow UI:  
mlflow ui --backend-store-uri sqlite:///lab-02-ml-pipeline-dagster/mlflow.db

---

## 🧪 Lab 3 – Modell-Serving & Monitoring

Pfad: lab-03-serving-monitoring/

Inhalte:
- Modell-Serving über Gradio-Web-App
- Nutzung der MLflow Model Registry
- Produktionsnahe Inferenz-Pipeline
- Data-Drift-Monitoring mit Evidently
- Einfaches Champion/Challenger-Konzept

Serving starten:
python lab-03-serving-monitoring/app.py

Monitoring-Pipeline starten:
dagster dev -f lab-03-serving-monitoring/dagster_pipeline.py

---

## 🧰 Tech-Stack

Modellierung: PyTorch, torchvision  
Orchestrierung: Dagster  
Experiment-Tracking: MLflow  
Serving: Gradio  
Monitoring: Evidently  
Daten: pandas, numpy  
Visualisierung: matplotlib, seaborn  

---

## 🔁 Reproduzierbarkeit

- Zentraler data/ Ordner im Projekt-Root  
- Konfigurierbare Pfade (CLI / ENV, je nach Lab)  
- MLflow für Tracking von Parametern, Metriken und Artefakten  
- Deterministische Train/Validation-Splits  

---

## ⚙️ Setup

python -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  


---

## 📸 Screenshots (optional)

- Dagster Pipeline Run  
- MLflow Experimente  
- Gradio App  
- Evidently Drift Report  

---

## 🔮 Mögliche Erweiterungen

- CI/CD mit GitHub Actions  
- Dockerisiertes Serving (z. B. FastAPI)  
- Datenvalidierung (Great Expectations)  
- Automatisches Retraining bei Drift  

---

## 👤 Autor

Khaled Fayzi  
MLOps / ML Engineering Portfolio-Projekt  
Universitätsprojekt