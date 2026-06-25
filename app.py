import streamlit as st
import numpy as np
import joblib

# --------------------------
# Page Configuration
# --------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# --------------------------
# Load Model
# --------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------
# Sidebar
# --------------------------

st.sidebar.title("🚢 Titanic Survival Predictor")

st.sidebar.info(
    """
    This Machine Learning application predicts whether a passenger would survive the Titanic disaster.

    **Model:** Logistic Regression
    """
)

# --------------------------
# Title
# --------------------------

st.title("🚢 Titanic Survival Prediction System")

st.markdown("---")

# --------------------------
# User Inputs
# --------------------------

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

    sex = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    fare = st.number_input(
        "Fare",
        0.0,
        600.0,
        50.0
    )

with col2:

    sibsp = st.number_input(
        "Siblings/Spouses",
        0,
        10,
        0
    )

    parch = st.number_input(
        "Parents/Children",
        0,
        10,
        0
    )

    embarked = st.selectbox(
        "Embarked",
        ["C","Q","S"]
    )

# --------------------------
# Encoding
# --------------------------

sex = 1 if sex=="Female" else 0

embarked_q = 1 if embarked=="Q" else 0
embarked_s = 1 if embarked=="S" else 0

# --------------------------
# Scaling
# --------------------------

scaled = scaler.transform(
    [[
        pclass,
        age,
        sibsp,
        parch,
        fare
    ]]
)

pclass = scaled[0][0]
age = scaled[0][1]
sibsp = scaled[0][2]
parch = scaled[0][3]
fare = scaled[0][4]

# --------------------------
# Prediction
# --------------------------

if st.button("Predict Survival"):

    features = np.array([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked_q,
        embarked_s
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][prediction]

    st.markdown("---")

    st.subheader("Prediction")

    if prediction==1:

        st.success("✅ Passenger is likely to Survive")

    else:

        st.error("❌ Passenger is unlikely to Survive")

    st.metric(
        "Prediction Confidence",
        f"{probability*100:.2f}%"
    )

    st.markdown("---")

    st.subheader("Passenger Details")

    st.write(f"**Passenger Class:** {1 if pclass else pclass}")
    st.write(f"**Gender:** {'Female' if sex else 'Male'}")
    st.write(f"**Age:** {age:.1f}")
    st.write(f"**Fare:** {fare:.2f}")
    st.write(f"**Siblings/Spouses:** {sibsp:.0f}")
    st.write(f"**Parents/Children:** {parch:.0f}")
    st.write(f"**Embarked:** {embarked}")