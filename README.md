# Proyecto MLOps — Predicción de Churn (Etapas 1 a 3)

Este repo está listo para que completes **hasta la Etapa 3** del proyecto del cuaderno *ISTEA — Lab Minería II*.

## Estructura
```
churn_mlops_project/
├─ data/
│  ├─ raw/            # coloca aquí tu archivo CSV crudo (p.ej. churn.csv)
│  └─ processed/      # salidas limpias
├─ src/
│  ├─ data_prep.py    # limpieza + features (Etapa 2)
│  └─ train.py        # entrenamiento (Etapa 3)
├─ params.yaml        # hiperparámetros y rutas
├─ dvc.yaml           # pipeline DVC (prep -> train)
├─ requirements.txt
└─ README.md
```

> **Dataset**: coloca tu CSV crudo en `data/raw/churn.csv` (o ajusta `params.yaml`).

---

# Paso a paso (con tiempos estimados)

> **Nota**: los tiempos son orientativos para una persona que ya tiene git, Python y conda instalados. Si te falta algo, agrega +20–40 min de setup de entorno.

## Etapa 1 — Setup inicial (∼ 60–90 min)
1. **Crear repo local y GitHub/DagsHub**
   ```bash
   git init
   git add .
   git commit -m "init: estructura base"
   # GitHub
   git remote add origin <URL-de-tu-repo-GitHub>
   git push -u origin main
   # (opcional) DagsHub como mirror o remote adicional
   git remote add dagshub <URL-de-tu-repo-DagsHub>
   git push -u dagshub main
   ```
2. **Crear y activar entorno**
   ```bash
   conda create -n churn-mlops python=3.11 -y
   conda activate churn-mlops
   pip install -r requirements.txt
   ```
3. **Inicializar DVC y enlazar remoto (DagsHub o local)**
   ```bash
   dvc init
   # Remoto en DagsHub (ejemplo)
   dvc remote add -d origin https://dagshub.com/<user>/<repo>.dvc
   git add .dvc/config
   git commit -m "chore: init dvc + remote"
   git push
   ```

**Entregable**: repo con estructura base y DVC inicializado.

---

## Etapa 2 — Limpieza y features (∼ 90–120 min)
1. **Coloca el dataset crudo** en `data/raw/churn.csv`.
2. **Ajusta rutas/columnas en `params.yaml`** si tu CSV tiene otros nombres.
3. **Ejecuta la stage de preparación** (crea dataset limpio y lo versiona con DVC):
   ```bash
   dvc repro prep
   dvc push
   git add dvc.lock data/.gitignore
   git commit -m "feat: dataset limpio versionado"
   git push
   ```

**Entregable**: pipeline reproducible con dataset crudo y limpio versionados.

---

## Etapa 3 — Entrenamiento de modelo (∼ 60–90 min)
1. **Entrenar con los hiperparámetros de `params.yaml`**:
   ```bash
   dvc repro train
   dvc push
   git add dvc.lock
   git commit -m "feat: modelo base + métricas"
   git push
   ```
2. **Revisar métricas** en `artifacts/metrics.json` y el modelo en `artifacts/model.pkl`.

**Entregable**: modelo entrenado, métricas versionadas, pipeline actualizado.

---

## Comandos útiles
- Reproducir todo el pipeline:
  ```bash
  dvc repro
  ```
- Ejecutar una etapa específica:
  ```bash
  dvc repro prep     # solo limpieza
  dvc repro train    # solo entrenamiento
  ```

---

## Siguientes pasos sugeridos (Etapa 4+)
- Correr **experimentos** variando hiperparámetros:
  ```bash
  dvc exp run -S model.C=0.5
  dvc exp run -S model.C=1.5
  dvc exp show
  ```
- Registrar en **MLflow/DagsHub** si lo prefieres.
