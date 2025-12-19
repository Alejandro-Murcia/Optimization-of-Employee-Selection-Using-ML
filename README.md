# The Better Hire — Optimization of Employee Selection Using Machine Learning (Ternium Case Study)

End-to-end **People Analytics / HR Analytics** project that builds and deploys supervised Machine Learning models to support **high-quality, faster, and more consistent hiring decisions**. The solution benchmarks multiple classifiers, performs rigorous evaluation (confusion-matrix metrics + ROC-AUC), tunes hyperparameters, and operationalizes the final model through a **Flask web application**.

> Academic context: Tecnológico de Monterrey (Campus Monterrey), course **TC2004B — Data Science Analysis**, in collaboration with **Ternium**.

![showcase](WebApp/static/company.png)

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Business Problem](#2-business-problem)
- [3. ML Formulation](#3-ml-formulation)
- [4. Dataset and Feature Space](#4-dataset-and-feature-space)
- [5. Data Preparation](#5-data-preparation)
- [6. Modeling Approach](#6-modeling-approach)
- [7. Evaluation Protocol and Metrics](#7-evaluation-protocol-and-metrics)
- [8. Results (Representative)](#8-results-representative)
- [9. Hyperparameter Optimization](#9-hyperparameter-optimization)
- [10. Deployment: Flask WebApp](#10-deployment-flask-webapp)
- [11. Reproducibility](#11-reproducibility)
- [12. Responsible Use (Bias, Governance)](#12-responsible-use-bias-governance)
- [13. Team, Roles, and Credits](#13-team-roles-and-credits)
- [14. References](#14-references)

---

## 1. Executive Summary

Hiring pipelines frequently depend on manual screening and rule-based scoring across many fields per candidate. This introduces:
- operational **latency** (slow screening),
- **inconsistency** across reviewers,
- and a higher risk of **missing strong candidates**.

This project builds a structured, tabular ML pipeline to predict the target variable:

**`Ingresados Sí/No`** → whether a candidate was admitted/hired.

We:
1) engineered a compact, operational feature set from candidate evaluations,
2) trained and compared multiple supervised classifiers,
3) evaluated models using confusion-matrix decomposition + ROC-AUC,
4) tuned hyperparameters via GridSearchCV for top candidates,
5) deployed the best-performing model into a Flask WebApp for practical usage by HR stakeholders.

![showcase](WebApp/static/showcase.png)

---

## 2. Business Problem

**Stakeholder:** Talent acquisition / HR teams.

**Objective:** Build a decision-support system that:
- reduces the effort of screening large candidate pools,
- increases consistency of pre-screen decisions,
- supports selection of “high potential” candidates with fewer input fields,
- and provides an accessible interface (web application) for real use.

**Business constraint:** The tool must be usable in real workflows. Therefore, we prioritize:
- features that are available early in screening,
- minimal manual input for prediction,
- and transparent evaluation metrics.

---

## 3. ML Formulation

**Task:** Binary classification  
- Class `1`: Candidate admitted/hired  
- Class `0`: Candidate not admitted

**Why supervised classification works here:** The dataset includes historical outcomes (“Ingresados Sí/No”), enabling learning of a decision boundary from prior labeled cases.

---

## 4. Dataset and Feature Space

The modeling phase focuses on a reduced set of operational variables derived from candidate assessments (e.g., psychometrics, recommendation flags, functional area signals, language proficiency, and group activity evaluations). Example variables used in the project:

### Predictors (examples)
- **Perfil Pymetrics** (categorical profile index; encoded as integer)
- **Altamente Recomendado** (binary)
- Functional area ratings (ordinal/categorical; encoded):
  - Operaciones–Calidad
  - MTTO–DIMA
  - Comercial–Planeamiento
  - DIGI–SC
  - Resto–Soft
- **Actividad Grupal.1** (categorical; encoded)
- **Apto AG** (binary)
- **Inglés** (ordinal: A1 → C2 encoded as 1..6)
- Screening summaries:
  - **Apto** (binary)
  - **Destacado** (binary)
  - **Destacado Pym** (binary)

### Target
- **Ingresados Sí/No** (binary)

> Note: The original dataset contained many additional attributes with significant missingness. Part of the project is identifying which features are both informative and realistic to operationalize.

---

## 5. Data Preparation

### 5.1 Cleaning and standardization
- UTF-8 compatible reading, with column name normalization for accents.
- Removal of irrelevant or deprecated variables (e.g., legacy scoring fields not used by stakeholders).
- Handling of duplicated columns (e.g., repeated “Inglés” / “Actividad Grupal” versions), keeping the informative/non-empty one.

### 5.2 Missing values strategy
Missingness was handled according to variable semantics:
- For targets: empty outcomes were mapped to `No` where consistent with stakeholder interpretation.
- For categorical inputs: missing values were mapped to explicit `Unknown` / `No` / `0`, depending on the field’s meaning.

### 5.3 Encoding
Because most variables are categorical, we convert them into machine-consumable numeric representations using domain-aligned mappings, for example:
- English level: A1..C2 → 1..6 (with special cases mapped to 0 where necessary).
- Recommendation tiers: Do Not Recommend / Recommend / Highly Recommend → 0/1/2.
- Binary flags: Yes/No → 1/0.

> This encoding approach keeps the pipeline simple and deployable while preserving ordinal meaning where applicable.

---

## 6. Modeling Approach

We benchmarked multiple supervised classifiers that are commonly strong on structured/tabular data:

- **K-Nearest Neighbors (KNN)**
- **Gaussian Naive Bayes (GNB)**
- **Support Vector Machine (SVM)**
- **Logistic Regression (LR)**
- **Decision Tree (DT)**
- **Gradient Boosting (GB)**
- **Random Forest (RF)**
- **Neural Network (MLP)**

The model shortlist is driven by:
- predictive performance,
- stability across metrics,
- robustness under class imbalance and practical deployment constraints,
- and the ability to tune/validate reliably.

---

## 7. Evaluation Protocol and Metrics

### 7.1 Train/Test split
We used a hold-out evaluation strategy with a fixed `random_state` for reproducibility:
- `train_test_split(X, y, random_state=0)` (typical split: 75/25 unless otherwise specified in code)

### 7.2 Confusion matrix decomposition
For each model we compute:
- **TP** (true positives), **TN** (true negatives), **FP** (false positives), **FN** (false negatives)

### 7.3 Metrics reported
- **Accuracy**: overall correctness.
- **Precision**: fraction of predicted positives that are correct.
- **Recall / Sensitivity**: fraction of actual positives recovered (important to avoid missing high-quality candidates).
- **Specificity**: fraction of actual negatives correctly rejected.
- **F1-score**: balance between precision and recall.
- **ROC-AUC**: ranking quality across thresholds.

> Hiring is a high-stakes decision process: accuracy alone is not enough. We emphasize recall/false negatives because failing to identify strong candidates is often more costly than reviewing an additional shortlist entry.

---

## 8. Results (Representative)

From the representative run shown in the report (example confusion matrices and derived metrics):

### Support Vector Machine (SVM)
- Confusion matrix:
  - TN = 237, FP = 5
  - FN = 2,  TP = 19
- **Accuracy:** 0.973
- **Precision:** ~0.792
- **Recall (Sensitivity):** ~0.905
- **Specificity:** ~0.979
- **F1-score:** ~0.844
- **ROC-AUC:** ~0.942

### Logistic Regression (LR)
- **Accuracy:** ~0.970
- **Precision:** ~0.783
- **Recall:** ~0.857
- **ROC-AUC:** ~0.918

### Random Forest (RF)
- Similar to SVM in the representative run:
- **Accuracy:** 0.973
- **Recall:** ~0.905
- **ROC-AUC:** ~0.942

Overall, the strongest candidates were **SVM**, **Logistic Regression**, and **Random Forest**, with SVM commonly selected for deployment due to strong aggregate performance and stable error trade-offs.

---

## 9. Hyperparameter Optimization

We applied **GridSearchCV** to the strongest candidates:

### SVM (example tuned configuration)
- `kernel`: `poly`
- `degree`: `2`
- `max_iter`: `-1` (no iteration cap)

### Logistic Regression (example tuned configuration)
- `C`: `2`
- `penalty`: `l2`

### Random Forest (example tuned configuration)
- `n_estimators`: explored values (e.g., 50..160 in grid)
- `criterion`: `gini` or `entropy`

After tuning, performance remained strong and similar across top models; final selection favored **SVM** due to its consistent performance and competitive precision/recall balance.

---

## 10. Deployment: Flask WebApp

The project includes a Flask web application that operationalizes the pipeline for real usage.

Typical functionality included:
1) **Candidate scoring**: HR enters a compact set of candidate attributes → the model outputs a recommendation (admit / do not admit).
2) **Dataset utilities (optional)**: upload and clean candidate datasets following the same rules used for training.
3) **Visualization panel (optional)**: plot summary distributions to provide context to stakeholders.

This transforms the work from “notebook-only” to an actual decision-support tool.

---

## 11. Reproducibility

To reproduce the full workflow end-to-end:
1) Obtain the dataset (often private in HR contexts; do not commit if restricted).
2) Run preprocessing steps (cleaning + encoding).
3) Train models and compute evaluation metrics.
4) Run GridSearchCV on shortlisted models.
5) Export the best model (`.pkl`) and load it in the Flask app.

---

## 12. Responsible Use (Bias, Governance)

This repository provides a **decision-support** approach, not an autonomous hiring system.

Before production use, any HR model should include:
- bias and fairness audits (gender, university, socioeconomic proxies),
- drift monitoring (changing applicant pools over time),
- governance on human-in-the-loop review,
- explainability artifacts (e.g., feature importance baselines, SHAP on tree models),
- and clear documentation of permissible use cases.

---

## 13. Team, Roles, and Credits

**Institution:** Tecnológico de Monterrey — Campus Monterrey  
**Course:** TC2004B — Data Science Analysis  
**Industry partner:** Ternium  
**Team:** Equipo 3 “Dinamita” — *The Better Hire*

**Roles**
- César Guillermo Vázquez Álvarez — Project Manager  
- Ana Daniela López Dávila — Data Scientist, Web/BI Designer  
- Alejandro José Murcia Alfaro — Chief Data Officer (CDO)  
- Camila Navarro Llaven — UX/UI Designer  
- Paola Guadalupe Machorro Ortiz — Data Engineer  

---

## 14. References

- Pessach, D., Singer, G., Avrahami, D., Chalutz Ben-Gal, H., Shmueli, E., & Ben-Gal, I. (2020). *Employees recruitment: A prescriptive analytics approach via machine learning and mathematical programming*. Decision Support Systems, 134, 113290. https://doi.org/10.1016/j.dss.2020.113290  
- Diez, F., Bussin, M., & Lee, V. (2019). *Fundamentals of HR Analytics: A Manual on Becoming HR Analytical*. Emerald Publishing.  
- Hunter, J. (2021). *New Standards: How Screening Data Is Evolving Hiring Process: Leveraging data to create new talent acquisition standards*. Talent Acquisition Excellence, 9(11), 20–21.  
- Chandler, S. (2017). *The AI chatbot will hire you now*. Wired.

---

## License
Add an appropriate license (e.g., MIT) if you want reusability. If the dataset is proprietary, document access constraints explicitly.
