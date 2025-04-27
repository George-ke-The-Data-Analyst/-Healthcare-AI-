# -Healthcare-AI-
Diabetes Risk Assessment Dashboard • Streamlit • Dask • MLflow • SHAP
# Diabetes Risk Assessment Dashboard

## Objective  
Build a lightweight, local **Diabetes Risk Assessment Dashboard** that predicts the onset of diabetes from patient clinical measurements, demonstrates model performance, and visualizes global feature importance—all in a Streamlit app.

---

## 1. Topic Selection   
**Domain:** Healthcare AI  
**Scope:** Use the Pima Indians Diabetes dataset (`diabetes.csv`) to predict diabetes with classical and ensemble models, then deploy as an interactive dashboard.

---

## 2. Problem Definition  
**Goal:** Given eight features—Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age—classify patients as **Diabetes** or **No Diabetes**, and provide risk probabilities.  
- **Benefit:** Enables small clinics to triage patients quickly without cloud infrastructure.

---

## 3. Background Study   
- **Dataset origin:** UCI Machine Learning Repository / Kaggle’s Pima Indians Diabetes database (768 samples, 9 columns).  
- **Classic methods:**  
  - **Logistic Regression** for interpretable decision boundaries (Hosmer & Lemeshow, 2000).  
  - **Random Forests** for robust, non-linear classification (Breiman, 2001).  
- **Deployment context:** Many FDA-approved decision-support tools begin as desktop dashboards before moving to enterprise platforms.

---

## 4. Methodology or Approach 
1. **Parallel data cleaning** with **Dask**: Replace physiologically impossible zeros with median values in parallel.  
2. **Train/test split**: 80/20 stratified, `random_state=42`.  
3. **Scaling**: Standardize features with `StandardScaler`.  
4. **Model selection**:  
   - **LogisticRegression** vs. **RandomForestClassifier**  
   - Hyperparameter tuning via 5-fold `GridSearchCV` on ROC-AUC  
5. **Experiment tracking** with **MLflow**: log parameters, metrics, and register best model.  
6. **Explainability** with **SHAP**: generate a global summary plot of feature importance.  
7. **Deployment**: Bundle best model & scaler in `diabetes_pipeline.joblib`, build a **Streamlit** dashboard with prediction, performance plots, and SHAP visualization.

---

## 5. Implementation or Analysis  
- **Training & logging script** (`train.py`):  
  ```python
  # …[data loading with Dask]…
  # …[GridSearchCV over LogisticRegression & RandomForest]…
  # …[Select best model, log with MLflow]…
  # …[Compute SHAP values & save summary plot]…
