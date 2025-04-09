import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time

# Page Config
st.set_page_config(page_title="Stress Level Prediction", layout="wide")

# Title
st.title("🚀 Stress Level Prediction Web App")

# Load Dataset
data = pd.read_csv(r'C:\Users\B.Suneel\OneDrive\Desktop\mtech sem2\own project\stress_dataset.csv')

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Model Comparison", "Make Prediction"])

# Home Section
if option == "Home":
    st.write("👋 Welcome to the Stress Level Prediction App!")
    st.write("""
        📊 **Dataset**: Stress Dataset with features like **temperature, steps, humidity, heart rate, sleep count**.
        
        🔥 **Models Used**:
        - Random Forest
        - Naive Bayes
        - SVM
        
        ✅ Feature Scaling & SMOTE applied.
    """)

# Model Comparison Section
elif option == "Model Comparison":
    st.subheader("📊 Model Accuracy Comparison")

    # Model names and accuracies
    model_names = ["Random Forest", "Naive Bayes", "SVM"]
    accuracies = [0.8318, 0.7105, 0.7417]

    # Create a bar plot for model comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=model_names, y=[acc * 100 for acc in accuracies], palette='Set2', ax=ax)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_title("Model Accuracy Comparison")

    # Add accuracy percentages on top of bars
    for i, acc in enumerate(accuracies):
        ax.text(i, acc * 100 + 1, f"{acc * 100:.2f}%", ha='center', fontsize=10)

    st.pyplot(fig)

# Make Prediction Section
elif option == "Make Prediction":
    st.subheader("🔍 Predict Stress Level")

    # Load Random Forest Model and Scaler
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure 'random_forest_model.pkl' and 'scaler.pkl' are in the correct directory.")
        st.stop()

    # User Inputs
    st.markdown("### Enter your physiological data:")
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", value=36.5, step=0.1, help="Body temperature in Celsius.")
    with col2:
        steps = st.number_input("👣 Steps", value=5000, step=100, help="Number of steps taken in a day.")
    with col3:
        humidity = st.number_input("💧 Humidity (%)", value=60, step=1, help="Humidity level in percentage.")

    col4, col5 = st.columns(2)
    with col4:
        heart_rate = st.number_input("❤️ Heart Rate (bpm)", value=80, step=1, help="Heart rate in beats per minute.")
    with col5:
        sleep_count = st.number_input("😴 Sleep Count (hours)", value=6.0, step=0.1, help="Hours of sleep.")

    # Initialize prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []

    # Reset history button
    if st.button("🗑️ Reset Prediction History"):
        st.session_state['prediction_history'] = []
        st.success("Prediction history has been reset!")

    # Prediction Button
    if st.button("🚀 Predict"):
        try:
            # Prepare input data
            input_data = pd.DataFrame([[temperature, steps, humidity, heart_rate, sleep_count]], 
                                      columns=['temperature', 'steps', 'humidity', 'heart_rate', 'sleep_count'])
            input_scaled = scaler.transform(input_data)

            # Progress Bar Animation
            with st.sidebar:
                st.info("🔄 Model is processing...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(100):
                    time.sleep(0.01)  # Smooth animation
                    progress_bar.progress(i + 1)
                    status_text.text(f"🚀 Progress: {i+1}%")
                time.sleep(0.3)  # Smooth ending

            # Make prediction using Random Forest
            prediction = rf_model.predict(input_scaled)[0]
            prediction_label = {0: "No Stress", 1: "Avg Stress", 2: "Stressed"}[prediction]

            # Display prediction result
            st.success(f"🎯 Predicted Stress Level: **{prediction_label}**")
            st.balloons()

            # Save prediction to history
            st.session_state['prediction_history'].append({
                'Temperature': temperature,
                'Steps': steps,
                'Humidity': humidity,
                'Heart Rate': heart_rate,
                'Sleep Count': sleep_count,
                'Prediction': prediction_label
            })

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Display Prediction History
    if st.session_state['prediction_history']:
        st.subheader("📜 Prediction History")
        history_df = pd.DataFrame(st.session_state['prediction_history'])
        st.dataframe(history_df)

        # Line Graph for Prediction Progress Over Time
        st.subheader("📈 Stress Prediction Progress Over Time")
        history_df['Prediction Code'] = history_df['Prediction'].map({'No Stress': 0, 'Avg Stress': 1, 'Stressed': 2})
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history_df.index, history_df['Prediction Code'], marker='o', linestyle='-', color='teal')
        ax.set_xlabel("Prediction Instance")
        ax.set_ylabel("Stress Level")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['No Stress', 'Avg Stress', 'Stressed'])
        ax.set_title("Prediction Progress Over Time")
        st.pyplot(fig)