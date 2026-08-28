# MediQ — Intelligent Symptom Diagnostics

A Flask web app that predicts likely diseases from selected symptoms using a
Random Forest model, and shows description, precautions, diets, medications,
and lifestyle recommendations for the predicted disease.

## Setup

```bash
pip install -r requirements.txt
python train_model.py   # builds artifacts/ (model, symptoms.json, knowledge_base.json)
python app.py            # runs the app at http://127.0.0.1:5001
```

## Project structure

- `app.py` — Flask web app (UI + `/predict` API)
- `train_model.py` — trains the model from `archive/` CSVs, writes `artifacts/`
- `archive/` — source dataset (symptoms, diseases, precautions, diets, medications, workouts)
- `artifacts/` — generated model + supporting JSON (created by `train_model.py`)

## Note

Model accuracy on this dataset is 100%, which likely reflects the dataset's
symptom-to-disease mapping being close to deterministic rather than real-world
diagnostic performance. This project is for educational purposes only and is
not a substitute for professional medical advice.
