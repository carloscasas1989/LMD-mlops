import os
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def main():
    # rutas (usamos las mismas que en dvc.yaml)
    data_path = "data/processed/churn_clean.csv"
    model_path = "artifacts/model.pkl"

    # por si queremos dejar todo junto
    os.makedirs("artifacts", exist_ok=True)

    # ===== 1. Cargar datos y modelo =====
    df = pd.read_csv(data_path)

    # ojo: asumimos que la columna target se llama "churn"
    y = df["churn"]
    X = df.drop(columns=["churn"])

    # cargamos el modelo entrenado
    model = joblib.load(model_path)

    # ===== 2. Predicciones =====
    # probas para ROC
    y_proba = model.predict_proba(X)[:, 1]
    # etiqueta final
    y_pred = model.predict(X)

    # ===== 3. Métricas numéricas =====
    roc_auc = roc_auc_score(y, y_proba)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    metrics = {
        "roc_auc": float(roc_auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = "artifacts/metrics_eval.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"ROC AUC: {roc_auc:.4f}")
    print("Métricas guardadas en", metrics_path)

    # ===== 4. Gráfico ROC =====
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="azar")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC - churn")
    plt.legend()
    roc_path = "artifacts/roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print("Curva ROC guardada en", roc_path)

    # ===== 5. Matriz de confusión =====
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de confusión - churn")
    cm_path = "artifacts/conf_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print("Matriz de confusión guardada en", cm_path)


if __name__ == "__main__":
    main()
