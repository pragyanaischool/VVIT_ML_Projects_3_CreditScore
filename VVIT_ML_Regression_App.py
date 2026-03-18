import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------------------------
# Title
# ---------------------------
st.title("🏦 Credit Score Prediction Dashboard")

# ---------------------------
# Load Dataset (FROM URL)
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/pragyanaischool/VVIT_ML_Projects_3_CreditScore/refs/heads/main/CreditScoring.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ---------------------------
# Basic Cleaning
# ---------------------------
df.columns = df.columns.str.strip()

# ---------------------------
# Sidebar Menu
# ---------------------------
menu = st.sidebar.selectbox("Menu", [
    "EDA Dashboard",
    "Model Training",
    "Hyperparameter Tuning",
    "Prediction"
])

# ---------------------------
# EDA
# ---------------------------
if menu == "EDA Dashboard":

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.subheader("📈 Data Types")
    st.write(df.dtypes)

    st.subheader("📊 Missing Values")
    st.write(df.isnull().sum())

    # Convert numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess(df):

    df = df.copy()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Target column (auto detect last column)
    target_col = df.columns[-1]

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns

X, y, scaler, feature_names = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Model Training
# ---------------------------
if menu == "Model Training":

    st.subheader("🤖 Model Selection")

    model_name = st.selectbox("Choose Model", [
        "Logistic Regression",
        "Random Forest",
        "SVM"
    ])

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Random Forest":
        model = RandomForestClassifier()

    else:
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

# ---------------------------
# Hyperparameter Tuning
# ---------------------------
if menu == "Hyperparameter Tuning":

    st.subheader("⚙️ Hyperparameter Tuning")

    model_name = st.selectbox("Select Model", [
        "Random Forest",
        "SVM"
    ])

    if model_name == "Random Forest":

        params = {
            'n_estimators':[100,200],
            'max_depth':[5,10]
        }

        grid = GridSearchCV(RandomForestClassifier(), params, cv=3)

    else:

        params = {
            'C':[0.1,1,10],
            'kernel':['linear','rbf']
        }

        grid = GridSearchCV(SVC(), params, cv=3)

    grid.fit(X_train, y_train)

    st.write("Best Params:", grid.best_params_)
    st.write("Best Score:", grid.best_score_)

# ---------------------------
# Prediction UI
# ---------------------------
if menu == "Prediction":

    st.subheader("🔮 Predict Credit Score")

    user_inputs = []

    for col in feature_names:
        val = st.number_input(f"{col}", value=0.0)
        user_inputs.append(val)

    # Train model once
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    if st.button("Predict"):
        data = scaler.transform([user_inputs])
        pred = model.predict(data)

        st.success(f"Predicted Credit Score: {pred[0]}")
