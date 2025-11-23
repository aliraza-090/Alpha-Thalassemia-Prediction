import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os

# ---------------------- SESSION STATE ----------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
/* Dark Gradient Background */
.stApp {background: linear-gradient(135deg, #37474f, #455a64); min-height:100vh;}

/* Animated Title */
.title-style {font-size:72px; font-weight:900; color:#ffcc80; text-align:center;
    animation: slideIn 1.2s ease-out forwards; text-shadow:4px 4px 15px rgba(0,0,0,0.5);}
@keyframes slideIn {0% {transform: translateY(-80px); opacity:0;} 100% {transform: translateY(0); opacity:1;}}

/* Result Box */
.result-box {padding:25px; border-radius:20px; font-size:22px; font-weight:700;
    box-shadow:0px 4px 20px rgba(0,0,0,0.25); text-align:center; transition:0.3s;}
.result-box:hover {transform:scale(1.05); box-shadow:0px 8px 30px rgba(0,0,0,0.35);}

/* Sidebar */
.sidebar .sidebar-content {background-color:#607d8b; border-radius:15px; padding:20px; color:#fff;}

/* AI Chat Button */
.ai-chat-btn {background-color:#ff9800; color:white; font-size:20px; font-weight:bold;
    padding:15px 25px; border:none; border-radius:15px; cursor:pointer; text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3); transition:0.3s; margin-top:20px;}
.ai-chat-btn:hover {transform:scale(1.05); box-shadow:0px 8px 25px rgba(0,0,0,0.4); background-color:#f57c00;}

/* Chat Message Boxes */
.user-msg {background-color:#ffcc80; padding:10px; border-radius:10px; margin-bottom:5px;}
.ai-msg {background-color:#90caf9; padding:10px; border-radius:10px; margin-bottom:5px;}
</style>
""", unsafe_allow_html=True)

# ---------------------- HELPER FUNCTION ----------------------
def display_result(pred, model_name):
    if pred == 0:
        st.markdown(f"""
            <div class="result-box" style="background-color:#388e3c; color:#fff;">
                🟢 <b>Result ({model_name}): You are healthy (Normal)</b><br>
                Your blood parameters are within normal range.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box" style="background-color:#d32f2f; color:#fff;">
                🔴 <b>Result ({model_name}): Alpha-Thalassemia Carrier</b><br>
                You may carry the Alpha-Thalassemia trait. Please consult your doctor for confirmation.
            </div>
        """, unsafe_allow_html=True)

# ---------------------- HOME PAGE ----------------------
if st.session_state.page == 'home':
    st.markdown('<p class="title-style">Alpha-Thalassemia Prediction</p>', unsafe_allow_html=True)

    # Load CSV
    csv_path = os.path.join(os.path.dirname(__file__), "alphanorm.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)
    df['phenotype'] = df['phenotype'].str.strip().str.lower().map({
        'normal':0, 'alpha carrier':1,'alpha_carrier':1,'alpha-carrier':1,'carrier':1}).astype(int)
    if 'sex' in df.columns: df['sex'] = df['sex'].str.lower().map({'male':1,'female':0}).fillna(0).astype(int)
    X = df.select_dtypes(include=[np.number]).drop(columns=['phenotype'])
    y = df['phenotype']

    # Scale + balance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, y)

    # Train models
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf.fit(X_bal, y_bal)
    xgb.fit(X_bal, y_bal)

    # Sidebar Inputs
    st.sidebar.header("Enter Patient Details")
    st.sidebar.write("### 🔹 Gender Input Guide")
    st.sidebar.write("✔ Enter 1 = Male, 0 = Female")
    user_input = {}
    for col in X.columns:
        val = st.sidebar.number_input(f"{col}", value=float(X[col].mean()))
        user_input[col] = val
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    # Prediction Buttons
    if st.sidebar.button("Predict with RandomForest"):
        pred = rf.predict(input_scaled)[0]
        display_result(pred, "RandomForest")
    if st.sidebar.button("Predict with XGBoost"):
        pred = xgb.predict(input_scaled)[0]
        display_result(pred, "XGBoost")

    st.write("✅ Models trained and ready to predict!")

    # AI Chat Button
    if st.button("💬 Chat with AI Consultant", key="ai_chat"):
        st.session_state.page = 'chat'

# ---------------------- CHAT PAGE ----------------------
if st.session_state.page == 'chat':
    st.markdown('<p class="title-style">AI Health Consultant</p>', unsafe_allow_html=True)
    st.info("Ask any question regarding symptoms, blood results, or health concerns. The AI will provide advice based on your input.")

    # Chat Input
    chat_input = st.text_input("Type your question here...", key="chat_input")
    send = st.button("Send 🡆")

    if send and chat_input.strip():
        symptom_keywords = ['pain','fever','blood','hb','rbc','mcv','mch','mchc','thalasemia','carrier','fatigue','weakness']
        if any(word in chat_input.lower() for word in symptom_keywords):
            ai_response = f"Based on your input: '{chat_input}', the AI suggests monitoring your symptoms and consulting a doctor. Regular checkups are recommended."
        else:
            ai_response = "Sorry, I can only answer health-related questions. Please ask about symptoms or blood results."

        st.session_state.chat_history.append({"role":"user","message":chat_input})
        st.session_state.chat_history.append({"role":"ai","message":ai_response})

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"<div class='user-msg'><b>You:</b> {chat['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'><b>AI:</b> {chat['message']}</div>", unsafe_allow_html=True)

    # Back button
    if st.button("⬅ Back to Prediction"):
        st.session_state.page = 'home'
