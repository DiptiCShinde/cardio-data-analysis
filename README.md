# Cardiovascular Disease Risk Prediction (Medallion Lakehouse & MLOps)

An end-to-end data engineering and machine learning pipeline built on Databricks to predict cardiovascular disease risk, prioritizing patient safety by minimizing false negatives in clinical screening.

---

## Project Overview

* **Goal:** Build an automated clinical screening pipeline on Databricks to predict cardiovascular risk from patient records.
* **Tech Stack:** PySpark, Delta Lake, Scikit-Learn, XGBoost, MLflow.
* **Dataset:** 68,500+ patient records covering physiological examinations, subjective habits, and medical biomarkers.
* **Core Impact:** Achieved **0.80 ROC-AUC** and **78% recall** by tuning the clinical decision threshold to maximize sensitivity.

---

## Pipeline Architecture

* **Bronze (`cardio_br`):** Raw batch ingestion of semicolon-delimited CSV clinical records into Delta Lake with schema validation.
* **Silver (`cardio_sil`):** Outlier removal based on physiological blood pressure bounds, record deduplication, age normalization, Body Mass Index (BMI) calculation, and blood pressure categorization.
* **Gold / ML:** Hemodynamic feature engineering (Pulse Pressure, Mean Arterial Pressure), preprocessing pipelines, model benchmarking, threshold optimization, and registration to MLflow Model Registry.

---

## Model Benchmark & Results

Evaluated on a holdout test set (20% split, $N = 13,719$):

* **Champion Model:** Tuned `XGBClassifier` (outperformed Logistic Regression and Random Forest).
* **Decision Threshold:** Set to **0.40** (shifted from 0.50 to reduce false negatives in screening).
* **Test Performance:**
  * **ROC-AUC:** 0.80
  * **Recall:** 78%
  * **Precision:** 71%
  * **Accuracy:** 73%
  * **F1-Score:** 0.74

---

## Key Risk Drivers (Feature Importance)

* **Primary:** Systolic blood pressure (~62% relative importance).
* **Secondary:** Cholesterol levels, patient age, and Mean Arterial Pressure (MAP).
* **Lifestyle & Baseline:** Physical activity, smoking status, glucose levels, and BMI.

---

## Repository Structure

```text
├── notebooks/
│   ├── 01_bronze_ingestion.py        # Raw CSV batch load to Delta Lake (cardio_br)
│   ├── 02_silver_transformation.py   # Cleansing, validations, and baseline features (cardio_sil)
│   └── 03_gold_ml_modeling.py        # Hemodynamics, model training, threshold tuning & MLflow
└── README.md