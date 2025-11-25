from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="API churn LMD-MLOps",
    description="Endpoint simple para predecir churn con el modelo entrenado.",
)

# Ruta al modelo entrenado
MODEL_PATH = "artifacts/model.pkl"
model = joblib.load(MODEL_PATH)


# =========
#  Esquema de entrada (TODAS las features que usa el modelo)
# =========
class ChurnRequest(BaseModel):
    age: int
    gender: str
    region: str
    contract_type: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    internet_service: str
    phone_service: str
    multiple_lines: str
    payment_method: str


@app.get("/")
def root():
    return {"message": "API de churn funcionando. Ir a /docs para probar."}


@app.post("/predict")
def predict(req: ChurnRequest):
    """
    Recibe un JSON con las mismas columnas que usó el modelo para entrenar
    y devuelve la probabilidad de churn.
    """
    try:
        # Convertimos el JSON a DataFrame con UNA fila
        data = pd.DataFrame([req.dict()])

        # Probabilidad de churn (clase 1)
        proba = model.predict_proba(data)[0][1]

        return {
            "churn_probability": float(proba),
            "churn_pred": int(proba >= 0.5),
        }
    except Exception as e:
        # Si algo falla, devolvemos el error como texto
        return {"error": f"Error al predecir: {e!s}"}

