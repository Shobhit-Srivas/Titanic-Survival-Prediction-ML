import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Performance Dashboard")

st.markdown("---")

# =========================
# Load Dataset
# =========================

df = pd.read_csv("titanic.csv")

df.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ],
    inplace=True
)

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

df["Sex"] = df["Sex"].map({
    "male":0,
    "female":1
})

df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True
)
df = df.dropna()
X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()

cols = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

X[cols] = scaler.fit_transform(X[cols])

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = joblib.load("model.pkl")

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

# =========================
# KPI Cards
# =========================

col1,col2,col3=st.columns(3)

with col1:

    st.metric(
        "Model",
        "Logistic Regression"
    )

with col2:

    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

with col3:

    st.metric(
        "Test Samples",
        len(X_test)
    )

st.markdown("---")

# =========================
# Confusion Matrix
# =========================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    prediction
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No","Actual Yes"],
    columns=["Predicted No","Predicted Yes"]
)

fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# Classification Report
# =========================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    prediction,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df
)

# =========================
# Feature Importance
# =========================

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Coefficient":model.coef_[0]
})

importance = importance.sort_values(
    "Coefficient",
    ascending=False
)

fig = px.bar(
    importance,
    x="Coefficient",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.success("Model evaluated successfully.")