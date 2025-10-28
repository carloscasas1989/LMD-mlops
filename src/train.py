#!/usr/bin/env python
import argparse, yaml, pandas as pd, json, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(args):
    params = load_params(args.params)
    data_path = Path(params["data"]["processed_path"])
    target = params["data"]["target"]
    artifacts_dir = Path(params["artifacts"]["model_path"]).parent
    model_path = Path(params["artifacts"]["model_path"])
    metrics_path = Path(params["artifacts"]["metrics_path"])

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset procesado en {data_path}. Ejecuta 'dvc repro prep'.")

    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna target '{target}' en el dataset.")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Separar columnas por tipo
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Modelo base: Logistic Regression
    model_cfg = params["model"]
    clf = LogisticRegression(
        C=model_cfg.get("C", 1.0),
        max_iter=model_cfg.get("max_iter", 500),
        random_state=model_cfg.get("random_state", 42)
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg.get("test_size", 0.2),
        random_state=model_cfg.get("random_state", 42),
        stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Modelo guardado en:", model_path)
    print("[OK] Métricas:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()
    main(args)
