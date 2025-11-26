# Proyecto LMD-MLOps – Predicción de Churn

Este proyecto busca predecir si un cliente dejará de usar un servicio (churn) usando un modelo de machine learning.  
Forma parte del trabajo práctico del curso 
**Laboratorio de Minería de Datos II (ISTEA)**.

---

## Funcionalidad del proyecto

1. **Limpieza y preparación del dataset (data_prep.py)
2. **Entrenamiento del modelo de regresión logística (train.py)
3. **Evaluación del modelo (evaluate.py)
- AUC-ROC
- Matriz de confusión
- Reporte de clasificación
- Curva ROC guardada como .png
4. Pipeline reproducible con DVC
5. Experimentación con variación de hiperparámetros
6. CI/CD que ejecuta tests y pipeline automáticamente
7. Servicio en producción mediante FastAPI
8. Predicción online vía endpoint /predict


>## Estructura principal

LMD-mlops/

│── data/

│   ├── raw/                  # Datos crudos

│   └── processed/            # Datos limpios

│── src/

│   ├── data_prep.py          # Limpieza y preparación de datos

│   ├── train.py              # Entrenamiento del modelo

│   ├── evaluate.py           # Métricas y gráficos

│   └── api.py                # API REST con FastAPI

│── artifacts/
│   ├── model.pkl             # Modelo entrenado

│   ├── metrics.json          # Métricas base

│   ├── metrics_eval.json     # Métricas avanzadas

│   ├── roc_curve.png         # Curva ROC

│   └── conf_matrix.png       # Matriz de confusión

│
│── params.yaml               # Hiperparámetros

│── dvc.yaml                  # Pipeline de DVC

│── dvc.lock                  # Estado del pipeline ejecutado

│── requirements.txt

│── README.md


## Tecnologías usadas

- Python 3.11
- Pandas, Scikit-learn
- DVC, GitHub Actions, FastAPI
- DagsHub
- Matplotlib
- FastAPI + Uvicorn

## Etapas del proyecto

1.Configuración inicial del entorno y repositorio	
**-Completada**

2️. Preparación y limpieza de datos
**-Completada**

3️. Entrenamiento del modelo base	
**-Completada**

4️. Experimentación y ajuste de hiperparámetros	 
**-Completada**

5. CI/CD con GitHub Actions
**Workflow: .github/workflows/ci.yaml**
**-Completada**
   
7. Iteración colaborativa con PRs
   
Incluye:

Script evaluate.py con métricas avanzadas

Curva ROC y matriz de confusión

API REST:
Endpoint de predicción:
POST /predict?age=30&income=5000

Ejecución del servidor:
uvicorn src.api:app --reload

**Autor**

Carlos Casas

https://github.com/carloscasas1989/LMD-mlops
