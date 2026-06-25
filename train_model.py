import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("titanic.csv")

# ===========================
# Drop Unnecessary Columns
# ===========================

df.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ],
    inplace=True
)

# ===========================
# Fill Missing Values
# ===========================

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# ===========================
# Encode Categorical Columns
# ===========================

df["Sex"] = df["Sex"].map(
    {
        "male":0,
        "female":1
    }
)

df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True
)

print("\nMissing Values Before Training:")
print(df.isnull().sum())

# Remove any remaining missing values
df = df.dropna()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ===========================
# Features & Target
# ===========================

X = df.drop(
    "Survived",
    axis=1
)

y = df["Survived"]

print("Training Features:\n")
print(X.columns.tolist())

# ===========================
# Feature Scaling
# ===========================

scaler = StandardScaler()

numeric = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

X[numeric] = scaler.fit_transform(
    X[numeric]
)

# ===========================
# Train Test Split
# ===========================

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ===========================
# Train Model
# ===========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

# ===========================
# Accuracy
# ===========================

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nAccuracy :",accuracy)

# ===========================
# Save Model
# ===========================

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nModel Saved Successfully")