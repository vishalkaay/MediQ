
import ast, json, pickle, re
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).resolve().parent
ARCHIVE = BASE / "archive"
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)
FILES = {
    "training": ARCHIVE / "Training.csv",
    "description": ARCHIVE / "description.csv",
    "diets": ARCHIVE / "diets.csv",
    "medications": ARCHIVE / "medications.csv",
    "precautions": ARCHIVE / "precautions_df.csv",
    "severity": ARCHIVE / "Symptom-severity.csv",
    "symptoms_df": ARCHIVE / "symtoms_df.csv",
    "workout": ARCHIVE / "workout_df.csv",
}

def clean(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+", " ", str(x).replace("\\", "").strip())

def dkey(x):
    x = clean(x).lower().replace("diseae", "disease").replace("paroymsal", "paroxysmal").replace("hemmorhoids", "hemorrhoids")
    return re.sub(r"\s+", " ", x).strip()

def skey(x):
    x = clean(x).lower().replace(" ", "_")
    x = re.sub(r"_+", "_", x)
    return x.strip("_")

def parse_list(x):
    if pd.isna(x) or str(x).strip() == "": return []
    try:
        y = ast.literal_eval(str(x))
        if isinstance(y, list): return [clean(i) for i in y if clean(i)]
    except Exception:
        pass
    return [clean(i) for i in re.split(r",|;", str(x)) if clean(i)]

def check_files():
    missing = [p.name for p in FILES.values() if not p.exists()]
    if missing: raise FileNotFoundError("Missing files: " + ", ".join(missing))

def dedupe_cols(df):
    if not df.columns.duplicated().any(): return df
    out = pd.DataFrame(index=df.index)
    for col in dict.fromkeys(df.columns):
        same = df.loc[:, df.columns == col]
        out[col] = same.iloc[:, 0] if same.shape[1] == 1 else same.apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)
    return out

def load_training():
    df = pd.read_csv(FILES["training"])
    df.columns = [clean(c) for c in df.columns]
    if "prognosis" not in df.columns: raise ValueError("Training.csv needs 'prognosis' column")
    df = df.rename(columns={c: skey(c) for c in df.columns if c != "prognosis"})
    df = dedupe_cols(df)
    features = [c for c in df.columns if c != "prognosis"]
    for c in features: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["prognosis"] = df["prognosis"].map(clean)
    df["disease_key"] = df["prognosis"].map(dkey)
    return df, [c for c in df.columns if c not in ["prognosis", "disease_key"]]

def display_names(df):
    return {r.disease_key: r.prognosis for r in df[["prognosis", "disease_key"]].drop_duplicates().itertuples()}

def add_desc(kb, names):
    df = pd.read_csv(FILES["description"])
    for _, r in df.iterrows():
        k = dkey(r.get("Disease", ""));
        if k: kb.setdefault(k, {"disease": names.get(k, clean(r.get("Disease", "")))})["description"] = clean(r.get("Description", ""))

def add_diets(kb, names):
    df = pd.read_csv(FILES["diets"])
    for _, r in df.iterrows():
        k = dkey(r.get("Disease", ""));
        if k: kb.setdefault(k, {"disease": names.get(k, clean(r.get("Disease", "")))})["diets"] = parse_list(r.get("Diet", ""))

def add_meds(kb, names):
    df = pd.read_csv(FILES["medications"])
    for _, r in df.iterrows():
        k = dkey(r.get("Disease", ""));
        if k: kb.setdefault(k, {"disease": names.get(k, clean(r.get("Disease", "")))})["medications"] = parse_list(r.get("Medication", ""))

def add_prec(kb, names):
    df = pd.read_csv(FILES["precautions"]); df.columns = [clean(c) for c in df.columns]
    for _, r in df.iterrows():
        k = dkey(r.get("Disease", ""))
        if k:
            vals = [clean(r.get(c, "")) for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
            kb.setdefault(k, {"disease": names.get(k, clean(r.get("Disease", "")))})["precautions"] = [v for v in vals if v]

def add_workout(kb, names):
    df = pd.read_csv(FILES["workout"]); df.columns = [clean(c) for c in df.columns]
    for _, r in df.iterrows():
        k = dkey(r.get("disease", r.get("Disease", "")))
        if k:
            item = clean(r.get("workout", ""))
            arr = kb.setdefault(k, {"disease": names.get(k, clean(r.get("disease", ""))) }).setdefault("lifestyle_recommendations", [])
            if item and item not in arr: arr.append(item)

def add_symptoms_df(kb, names):
    df = pd.read_csv(FILES["symptoms_df"]); df.columns = [clean(c) for c in df.columns]
    cols = [c for c in df.columns if c.lower().startswith("symptom")]
    for _, r in df.iterrows():
        k = dkey(r.get("Disease", ""))
        if k:
            arr = kb.setdefault(k, {"disease": names.get(k, clean(r.get("Disease", ""))) }).setdefault("common_symptoms", [])
            for c in cols:
                s = skey(r.get(c, ""))
                if s and s not in arr: arr.append(s)

def load_severity():
    df = pd.read_csv(FILES["severity"])
    sev = {}
    for _, r in df.iterrows():
        s = skey(r.get("Symptom", ""))
        if s: sev[s] = max(sev.get(s, 0), int(r.get("weight", 0)))
    return sev

def build_kb(train_df):
    names = display_names(train_df)
    kb = {k: {"disease": v} for k, v in names.items()}
    add_desc(kb, names); add_diets(kb, names); add_meds(kb, names); add_prec(kb, names); add_workout(kb, names); add_symptoms_df(kb, names)
    for v in kb.values():
        for field, default in {"description":"", "diets":[], "medications":[], "precautions":[], "lifestyle_recommendations":[], "common_symptoms":[]}.items():
            v.setdefault(field, default)
    return kb

def main():
    check_files()
    df, features = load_training()
    le = LabelEncoder(); y = le.fit_transform(df["disease_key"]); X = df[features]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", max_features="sqrt", n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte); acc = accuracy_score(yte, pred)
    sev = load_severity(); kb = build_kb(df)
    pickle.dump({"model": model, "label_encoder": le, "feature_cols": features, "severity": sev}, open(ART/"disease_model.pkl", "wb"))
    json.dump([{"id": s, "label": s.replace("_", " "), "severity": sev.get(s, 0)} for s in sorted(features)], open(ART/"symptoms.json", "w", encoding="utf-8"), indent=2)
    json.dump(kb, open(ART/"knowledge_base.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(ART/"model_report.txt", "w", encoding="utf-8").write(f"Accuracy: {acc:.4f}\n\n" + classification_report(yte, pred, zero_division=0))
    print(f"Training complete. Accuracy: {acc:.4f}. Saved to {ART}")

if __name__ == "__main__": main()
