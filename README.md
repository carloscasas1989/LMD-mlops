# Proyecto LMD-MLOps – Predicción de Churn

Este proyecto busca predecir si un cliente dejará de usar un servicio (churn) usando un modelo de machine learning.  
Forma parte del trabajo práctico del curso 
**Laboratorio de Minería de Datos II (ISTEA)**.

---

## Funcionalidad del proyecto

1. **Limpia los datos** del archivo `telco_churn.csv`
2. **Entrena un modelo** de regresión logística
3. **Guarda y versiona** los datos, el modelo y las métricas con DVC
4. **Registra experimentos** y resultados en DagsHub

>## Estructura principal

LMD-mlops/

├── data/ # Datos crudos y procesados

├── src/ # Scripts de preprocesamiento y entrenamiento

├── artifacts/ # Modelo entrenado y métricas

├── params.yaml # Parámetros del modelo

├── dvc.yaml # Definición del pipeline

└── README.md


## Tecnologías usadas

- Python 3.11

- Pandas, Scikit-learn

- DVC

- DagsHub

- Git

## Etapas del proyecto

1.	Configuración inicial del entorno y repositorio	**-Completada**
2️. Preparación y limpieza de datos	**-Completada**
3️. Entrenamiento del modelo base	**-Completada**
4️. Experimentación y ajuste de hiperparámetros	 **En progreso**

**Autor**

Carlos Casas