# Hydraulic Sparse-Sensing — corrida de revisión

Experimentos de refuerzo para el paper *"Sparse Sensing for Building IoT Predictive
Maintenance: A Multi-Model Benchmark on Hydraulic Condition Monitoring"* (MIA-103,
UV). Responde a la revisión de 6 jueces (apunta a ~6.5/7) sobre el dataset público
**UCI Condition Monitoring of Hydraulic Systems** (descarga automática).

## Cómo correr (Colab)

Abrí `notebooks/Hydraulic_Revision_Colab.ipynb` en Google Colab (Runtime **con GPU**
para el experimento KAN) y *Run all*. El notebook clona este repo, instala
dependencias, descarga el dataset y corre todo. Es **resumible**: el cache vive en
`MyDrive/hydra_cache/`, así que si Colab se corta, re-ejecutás *Run all* y continúa.

## Qué corre

| # | Experimento | Responde a |
|---|---|---|
| **1** | Forward-selection **solo-físico** (excluye sensores virtuales SE/CE/CP) vs todos | el kill-shot del revisor adversarial: ¿la reducción de sensores es real o pasa por el sensor virtual SE? |
| **2** | Multiseed **30 seeds** + tamaños de efecto (Cliff's delta) | potencia estadística (antes 10 seeds) |
| **3** | Tuning **uniforme** (mismo presupuesto random-search para las 8 familias) | comparación cross-tier justa (antes boosting tenía 977 trials vs GridSearch) |
| **5** | Expresión **simbólica KAN** (2 sensores, GPU) | mostrar la recuperación simbólica que el título reclama |
| **6** | **Friedman + Nemenyi** (diagrama Critical Difference) + medianas/IQR | reporte estadístico completo |

## Estructura

- `hydra/io.py`, `hydra/features.py` — carga del dataset y extracción de 255 features (17 sensores × 15 descriptores).
- `hydra/experiments.py` — toda la lógica de #1–#6 (cacheable, paralela CPU/GPU).
- `notebooks/Hydraulic_Revision_Colab.ipynb` — orquestación con `rich`/`tqdm`.

## Reproducibilidad

Seeds fijas (42…), cache firmado por parámetros, presupuesto de tuning uniforme.
Dataset: UCI ML Repository, *Condition monitoring of hydraulic systems* (ID 447).
