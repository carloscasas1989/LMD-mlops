#!/usr/bin/env python
import argparse, yaml, pandas as pd, json, os, sys
from pathlib import Path

def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(args):
    params = load_params(args.params)
    raw_path = Path(params["data"]["raw_path"])
    out_path = Path(params["data"]["processed_path"])
    target = params["data"]["target"]
    drop_cols = params["features"].get("drop_cols", [])
    numeric_impute = params["features"].get("numeric_impute", "median")

    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset crudo en {raw_path}.")
    df = pd.read_csv(raw_path)

    # Limpieza mínima y robusta
    # 1) Tipos y espacios
    df.columns = [c.strip() for c in df.columns]
    # 2) Quitar columnas no útiles
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 3) Imputación simple numérica
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_impute == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif numeric_impute == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # 4) Eliminar filas con target faltante (si existiera)
    if target in df.columns:
        df = df[~df[target].isna()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Guardado limpio: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()
    main(args)
