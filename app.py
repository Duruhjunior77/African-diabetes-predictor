import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="African Diabetes Predictor", page_icon="🩺")
st.image("futurizestudio_logo.jpeg", width=150)
# -------------------------------------------
# Team Section
# -------------------------------------------

st.sidebar.title("Project Team")

st.sidebar.markdown("""
**Team Name:** Futurize Academy ⚡ Zenith-Trident Team

**Developers:**
- **Joseph Duruh** – Lead Developer (AI/ML)
- **Nasisira Seezibella** – IT Infrastructure & Systems
- **Chimyzerem Janet Uche-Ukah** – Software Developer
""")




@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['preg', 'glucose', 'bp', 'skin', 'insulin','bmi', 'pedigree', 'age', 'class']
    df = pd.read_csv(url, names=cols)

    zero_cols = ['glucose','bp','skin','insulin','bmi']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['glucose'] = df['glucose'].fillna(df['glucose'].median())
    df.fillna(df.mean(), inplace=True)

    df['bmi_age'] = df['bmi'] * df['age']
    df['glucose_bmi'] = df['glucose'] * df['bmi']
    df['high_bp'] = (df['bp'] > 130).astype(int)

    return df

@st.cache_resource
def train_models(df):
    X = df.drop("class", axis=1)
    y = df["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    }

    results = {}

    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        results[name] = {
            "model": m,
            "accuracy": accuracy,
            "auc": auc,
            "scaler": scaler,
            "features": X.columns
        }

    return results

df = load_data()
models = train_models(df)

st.title("🩺 African Diabetes Risk Predictor")
st.markdown("Prototype model using multiple ML algorithms.")

# model selection
selected = st.selectbox("Choose model", list(models.keys()))
model_info = models[selected]
model = model_info["model"]
scaler = model_info["scaler"]
features = model_info["features"]

# display performance
st.subheader("Model Performance")
st.write("Accuracy:", model_info["accuracy"])
st.write("ROC-AUC:", model_info["auc"])

# patient input
st.sidebar.header("Patient Data")
def user_input():
    d = {}
    for feature in features:
        if feature == "high_bp":
            continue
        d[feature] = st.sidebar.number_input(feature, value=1.0)
    d["high_bp"] = 1 if d["bp"] > 130 else 0
    d["bmi_age"] = d["bmi"] * d["age"]
    d["glucose_bmi"] = d["glucose"] * d["bmi"]
    return pd.DataFrame([d])

input_df = user_input()
st.subheader("Patient Input")
st.write(input_df)

if st.button("Predict"):
    X = scaler.transform(input_df[features])
    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    st.write("Risk Probability:", f"{proba*100:.2f}%")
    st.write("Prediction:", "High Risk" if pred==1 else "Low Risk")
