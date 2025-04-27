import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics          import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing     import StandardScaler

# --- Page configuration ---
st.set_page_config(
    page_title="Diabetes Risk Dashboard",
    layout="wide"
)

# --- Load pipeline ---
pipe = joblib.load("diabetes_pipeline.joblib")
model, scaler = pipe["model"], pipe["scaler"]

# --- Sidebar inputs ---
st.sidebar.header("Enter Patient Data")
df_orig = pd.read_csv("diabetes.csv")

inputs = {}
for feat in ["Pregnancies","Glucose","BloodPressure","SkinThickness",
             "Insulin","BMI","DiabetesPedigreeFunction","Age"]:
    low  = float(df_orig[feat].min())
    high = float(df_orig[feat].max())
    default = float(df_orig[feat].median())
    if feat in ["Pregnancies","Age"]:
        # integer sliders
        inputs[feat] = st.sidebar.slider(feat, int(low), int(high), int(default))
    else:
        inputs[feat] = st.sidebar.slider(feat, low, high, default)

# --- Prediction ---
st.header("Prediction")
input_df = pd.DataFrame([inputs])
X_scaled = scaler.transform(input_df)
prob     = model.predict_proba(X_scaled)[0,1]
label    = "Diabetes" if prob > 0.5 else "No Diabetes"
st.markdown(f"### **{label}**  (Risk = {prob:.1%})")

# --- Test‐set performance ---
with st.expander("Show Test‐Set Performance Metrics"):
    # Recreate test split & scaling
    df = df_orig.copy()
    for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        df[col].replace(0, np.nan, inplace=True)
        df[col].fillna(df[col].median(), inplace=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    X_te_s = scaler.transform(X_te)
    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:,1]

    acc     = accuracy_score(y_te, y_pred)
    auc     = roc_auc_score(y_te, y_prob)
    report  = classification_report(y_te, y_pred, target_names=["No Diabetes","Diabetes"])
    cm      = confusion_matrix(y_te, y_pred)

    st.write(f"**Accuracy:** {acc:.3f}  **ROC‐AUC:** {auc:.3f}")
    st.subheader("Classification Report")
    st.text(report)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Diabetes","Diabetes"])
    ax.set_yticklabels(["No Diabetes","Diabetes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>max(cm.flatten())/2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --- SHAP summary plot ---
with st.expander("Show Global Feature Importance (SHAP)"):
    st.image("shap_summary.png", use_column_width=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "📊 Built with Streamlit • Pipeline trained with Dask, MLflow & SHAP"
)