import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoder.pkl")

def preprocess_input(data):
    df = pd.DataFrame([data])

    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    df = scaler.transform(df)
    return df

def predict_survival(data):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0]
    return prediction, probability