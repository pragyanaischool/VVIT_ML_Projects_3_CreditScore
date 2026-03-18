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
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CreditScoring.csv")
    return df

df = load_data()

# ---------------------------
# Sidebar Navigation
# ---------------------------
menu = st.sidebar.selectbox("Menu", [
    "EDA Dashboard",
    "Model Training",
    "Hyperparameter Tuning",
    "Prediction"
])

# ---------------------------
# EDA Dashboard
# ---------------------------
if menu == "EDA Dashboard":

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    st.subheader("📈 Class Distribution")
    st.bar_chart(df['Credit_Score'].value_counts())

    st.subheader("📊 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(df):

    df = df.dropna()

    le = LabelEncoder()

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Credit_Score', axis=1)
    y = df['Credit_Score']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler

X, y, scaler = preprocess(df)

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

    st.subheader("📊 Results")

    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

# ---------------------------
# Hyperparameter Tuning
# ---------------------------
if menu == "Hyperparameter Tuning":

    st.subheader("⚙️ GridSearchCV")

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

    inputs = []

    for col in df.drop('Credit_Score', axis=1).columns:
        val = st.number_input(col)
        inputs.append(val)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    if st.button("Predict"):
        data = scaler.transform([inputs])
        pred = model.predict(data)

        st.success(f"Predicted Credit Score: {pred[0]}")
