import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

st.title("Alpha-Thalassemia Prediction")

# --- Load CSV
csv_path = os.path.join(os.path.dirname(__file__), "alphanorm.csv")
df = pd.read_csv(csv_path)

# --- Preprocessing
df = df.dropna().reset_index(drop=True)
df['phenotype'] = df['phenotype'].str.strip().str.lower().map({
    'normal': 0, 'alpha carrier': 1, 'alpha_carrier': 1, 'alpha-carrier': 1, 'carrier': 1
}).astype(int)

if 'sex' in df.columns:
    df['sex'] = df['sex'].str.lower().map({'male':1,'female':0}).fillna(0).astype(int)

X = df.select_dtypes(include=[np.number]).drop(columns=['phenotype'])
y = df['phenotype']

# --- Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SMOTE balancing
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_scaled, y)

# --- Train models
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

rf.fit(X_bal, y_bal)
xgb.fit(X_bal, y_bal)

# --- Sidebar for user input
st.sidebar.header("Enter patient details:")

user_input = {}
for col in X.columns:
    val = st.sidebar.number_input(f"{col}", value=float(X[col].mean()))
    user_input[col] = val

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# --- Prediction buttons
if st.sidebar.button("Predict with RandomForest"):
    pred = rf.predict(input_scaled)[0]
    st.write(f"Prediction (RandomForest): {'Normal' if pred==0 else 'Alpha Carrier'}")

if st.sidebar.button("Predict with XGBoost"):
    pred = xgb.predict(input_scaled)[0]
    st.write(f"Prediction (XGBoost): {'Normal' if pred==0 else 'Alpha Carrier'}")

st.write("✅ Models trained and ready to predict!")
