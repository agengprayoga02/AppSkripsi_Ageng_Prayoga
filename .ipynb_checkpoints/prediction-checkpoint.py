# prediction.py
import streamlit as st
import os
import pandas as pd
import numpy as np
import sqlite3
from utils import load_dataset, scale_data, check_stationarity, split_data_arima, evaluate_predictions, plot_predictions, generate_recommendation, compare_actual_vs_predicted, transformer_prediction_final, load_transformer_model_final, evaluate_arima_model
from db import insert_history, insert_evaluation_history

DATASETS_DIR = "C:/Users/ASUS/Aplikasi_Skripsi/datasets"

def prediction_page(conn, cursor):
    st.header("Prediksi & Evaluasi")

    # Fetch Models from DB
    cursor.execute("SELECT model_id, model_name, algorithm FROM models")
    models = cursor.fetchall()
    model_options = {f"{model[1]} ({model[2]})": model[0] for model in models} # Dictionary of "Model Name (Algorithm)": model_id

    model_name = st.selectbox("Choose Model", options=model_options.keys())
    model_id = model_options[model_name]
    algorithm = model_name.split('(')[-1][:-1] # Extract algorithm from model name

    # Load Dataset
    file_name = st.selectbox("Select a dataset", os.listdir(DATASETS_DIR))
    df = load_dataset(file_name) if file_name else None

    if df is None:
        st.stop()

    # Validasi NaN di y_true
    y_true = df["jumlah_kasus"].tolist()
    if any(pd.isna(y_true)):
        st.error("Dataset contains NaN in target column 'jumlah_kasus'. Please clean the data.")
        st.stop()

    # Preprocessing Data
    st.write("Dataset Loaded. Performing Exploratory Data Analysis:")
    st.write(df.head())
    plot_predictions(df["jumlah_kasus"].values[:12], df["jumlah_kasus"].values[:12], title="Actual Data")
    metrics_data_frame = df.info()
    st.write(metrics_data_frame)

    forecast_months = st.number_input("Enter the number of months to forecast", min_value=1, step=1, value=12)
    if forecast_months > 12:
        st.write(f"You chose to predict for {forecast_months} months, this is {forecast_months - 12} month(s) more than standard 12 months forecast.")
        
    if st.button("Preprocess and Run Prediction & Evaluation"):
        if algorithm == "ARIMA":
            #model_dict = utils.load_arima_model() # remove this
            #if model_dict: # Remove this also
            
            metrics, y_pred_original_scale = evaluate_arima_model(df, forecast_months)
            if metrics is not None and y_pred_original_scale is not None:
                st.success("ARIMA prediction and evaluation completed.")
                
                # Validasi NaN di y_true
                if any(pd.isna(y_pred_original_scale)):
                  st.error("ARIMA Model prediction contains NaN. Check model input or preprocessing steps.")
                  st.stop()

                y_actual = df["jumlah_kasus"].values[-len(y_pred_original_scale):]
                plot_predictions(y_actual, y_pred_original_scale, algorithm)
                metrics = evaluate_predictions(y_actual, y_pred_original_scale)

                recommendation = generate_recommendation(metrics, y_pred_original_scale, y_actual)
                st.write("Recommendation:", recommendation)
                comparison_metrics = compare_actual_vs_predicted(y_actual, y_pred_original_scale)
                st.write("Comparison Metrics:", comparison_metrics)
                
                try:
                  # Insert history and evaluation
                    cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = ? ORDER BY dataset_id DESC LIMIT 1", (file_name,))
                    result = cursor.fetchone()
                    dataset_id = result[0] if result else None
                    
                    if dataset_id is not None:
                        # Insert history and evaluation
                        if insert_history(cursor, model_id, dataset_id, y_pred_original_scale.tolist()):  # Convert NumPy array to list
                            conn.commit()
                            st.success(f"Prediction added to database.")
                        else:
                            st.error(f"Failed to add prediction to database.")

                        if insert_evaluation_history(cursor, model_id, dataset_id, metrics):
                            conn.commit()
                            st.success(f"Evaluation added to database.")
                        else:
                            st.error(f"Failed to add evaluation to database.")
                    else:
                        st.error("dataset_id not found.")
                except sqlite3.Error as e:
                    st.error(f"Failed to add prediction or evaluation to database: {e}")
            else:
              st.error("Failed to load ARIMA model.")
              st.stop()
              
                
        elif algorithm == "Transformer":
            y_pred = transformer_prediction_final(df, forecast_months)  # Use the new function
            if y_pred is not None:
                y_true = df["jumlah_kasus"].tolist()
                st.success("Transformer prediction completed.")
                st.write("Predictions:", y_pred[:forecast_months])  # Display forecast months
                y_actual = y_true[len(y_true) - len(y_pred):]
                plot_predictions(y_actual, y_pred, title="Transformer Actual vs Predicted")

                metrics = evaluate_predictions(y_actual, y_pred)
                recommendation = generate_recommendation(metrics, y_pred, y_actual)
                st.write("Recommendation:", recommendation)
                comparison_metrics = compare_actual_vs_predicted(y_actual, y_pred)
                st.write("Comparison Metrics:", comparison_metrics)

                try:
                    # Fetch model_id for the final model (assuming you have a record for it in the models table)
                    #cursor.execute("SELECT model_id FROM models WHERE model_name = 'final_model' ORDER BY model_id DESC LIMIT 1")  # Adjust query as needed
                    #result = cursor.fetchone()
                    #model_id = result[0] if result else None

                    if model_id is not None:
                        cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = ? ORDER BY dataset_id DESC LIMIT 1", (file_name,))
                        result = cursor.fetchone()
                        dataset_id = result[0] if result else None
                        if dataset_id is not None:
                            if insert_history(cursor, model_id, dataset_id, y_pred.tolist()):
                                conn.commit()
                                st.success(f"Prediction added to database.")
                            else:
                                st.error(f"Failed to add prediction to database.")
                            if insert_evaluation_history(cursor, model_id, dataset_id, metrics):
                                conn.commit()
                                st.success(f"Evaluation added to database.")
                            else:
                                st.error(f"Failed to add evaluation to database.")
                        else:
                            st.error("dataset_id not found.")
                    else:
                        st.error("model_id not found.")
                except sqlite3.Error as e:
                    st.error(f"Failed to add prediction or evaluation to database: {e}")
    else:
      st.write("Select an algorithm or upload the model.")