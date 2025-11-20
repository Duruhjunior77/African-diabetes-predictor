import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def predict(preg, glucose, bp, skin, insulin, bmi, pedigree, age):
    df = pd.DataFrame([{
        "preg": preg,
        "glucose": glucose,
        "bp": bp,
        "skin": skin,
        "insulin": insulin,
        "bmi": bmi,
        "pedigree": pedigree,
        "age": age,
        "bmi_age": bmi*age,
        "glucose_bmi": glucose*bmi,
        "high_bp": 1 if bp>130 else 0
    }])

    scaler = StandardScaler()
    model = LogisticRegression(solver="liblinear")

    # placeholder trained on Pima dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['preg', 'glucose', 'bp', 'skin','insulin','bmi','pedigree','age','class']
    data = pd.read_csv(url, names=cols)
    data['bmi_age'] = data['bmi'] * data['age']
    data['glucose_bmi'] = data['glucose'] * data['bmi']
    data['high_bp'] = (data['bp']>130).astype(int)

    X = data.drop("class", axis=1)
    y = data["class"]
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)

    X_new = scaler.transform(df)
    proba = model.predict_proba(X_new)[0][1]
    pred = model.predict(X_new)[0]

    return f"Risk: {proba*100:.2f}%", "High Risk" if pred else "Low Risk"

iface = gr.Interface(
    fn=predict,
    inputs=["number"]*8,
    outputs=["text","text"],
    title="African Diabetes Predictor (Prototype)"
)

iface.launch()
