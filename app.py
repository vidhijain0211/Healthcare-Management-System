import streamlit as st
import sqlite3
import hashlib
import pickle
import os
import pandas as pd
import numpy as np
from feature_matadata import feature_metadata

# ----------------- User Auth DB -----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user and user[0] == hash_password(password)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://c8.alamy.com/comp/R5M2TB/medical-diagnosis-black-icon-vector-sign-on-isolated-background-medical-diagnosis-concept-symbol-illustration-R5M2TB.jpgg');
        background-size: cover;
    }
    .css-1d391kg { display: none }  /* Hide sidebar */
    </style>
""", unsafe_allow_html=True)

st.title("Healthcare Prediction Using ML")

# ----------------- Session Control -----------------
if 'menu' not in st.session_state:
    st.session_state['menu'] = "Login"

# ----------------- Paths -----------------
feature_files = {
    "Diabetes": "Datasets/diabetes_data.csv",
    "Heart Disease": "Datasets/heart_disease_data.csv",
    "Parkinson's": "Datasets/parkinson_data.csv",
    "Lung Cancer": "Datasets/preprocessed_lungs_data.csv",
    "Thyroid": "Datasets/preprocessed_hypothyroid.csv"
}

model_files = {
    "Diabetes": "models/best_diabetes_model.sav",
    "Heart Disease": "models/best_heart_model.sav",
    "Parkinson's": "models/best_parkinsons_model.sav",
    "Lung Cancer": "models/best_lung_cancer_model.sav",
    "Thyroid": "models/best_thyroid_model.sav"
}

# ----------------- Utilities -----------------
def get_features(disease):
    file_path = feature_files.get(disease)
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.columns.tolist()[:-1]
    return []

def load_model(disease):
    model_path = model_files.get(disease)
    if model_path and os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, tuple) and len(obj) == 2:
                return obj[0], obj[1]  # scaler, model
            else:
                return None, obj  # older model only
    return None, None

# ----------------- Pages -----------------
if st.session_state['menu'] == "Sign Up":
    st.subheader("Create New Account")
    new_user = st.text_input("Username", key="signup_username")
    new_pass = st.text_input("Password", type="password", key="signup_password")
    if st.button("Sign Up"):
        if register_user(new_user, new_pass):
            st.success("Account created successfully! Go to Login.")
            st.session_state['menu'] = "Login"
            st.rerun()
        else:
            st.error("Username already exists.")
    if st.button("Back to Login"):
        st.session_state['menu'] = "Login"
        st.rerun()

elif st.session_state['menu'] == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if login_user(username, password):
            st.success(f"Welcome {username}!")
            st.session_state['menu'] = "Disease Prediction Page"
            st.rerun()
        else:
            st.error("Invalid credentials.")
    if st.button("Create an Account"):
        st.session_state["menu"] = "Sign Up"
        st.rerun()

elif st.session_state['menu'] == "Disease Prediction Page":
    st.subheader("Select Disease for Prediction")
    disease = st.selectbox("Choose a Disease", list(feature_files.keys()))

    if disease:
        st.subheader(f"Enter Patient Details for {disease}")
        features = get_features(disease)
        user_inputs = {}

        for feature in features:
            if feature == "sex":
                selected_gender = st.selectbox("Sex", options=["Male", "Female"])
                user_inputs['sex'] = 1 if selected_gender == "Male" else 0
            else:
                min_val, max_val, default_val = 0, 100, 0
                if disease in feature_metadata and feature in feature_metadata[disease]:
                    min_val, max_val, default_val = feature_metadata[disease][feature]

                if isinstance(default_val, int):
                    user_inputs[feature] = st.number_input(feature, min_value=min_val, max_value=max_val, value=default_val, step=1)
                else:
                    user_inputs[feature] = st.number_input(feature, min_value=min_val, max_value=150.0, value=min(default_val, 150.0), step=0.1, format="%.2f")

        if st.button("Predict Disease"):
            scaler, model = load_model(disease)
            if model:
                input_data = np.array([[user_inputs[feature] for feature in features]])
                if scaler:
                    input_data = scaler.transform(input_data)
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.error(f"The patient has {disease}.")
                else:
                    st.success(f"The patient does NOT have {disease}.")
            else:
                st.error("Model not found.")

        if st.button("Logout"):
            st.session_state['menu'] = "Login"
            st.rerun()
