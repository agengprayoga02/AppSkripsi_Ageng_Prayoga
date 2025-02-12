import streamlit as st
import os
import pandas as pd
from utils import load_dataset, load_arima_model, evaluate_arima_model, load_transformer_model, evaluate_transformer, plot_predictions, load_transformer_model_final, transformer_prediction_final, evaluate_predictions, generate_recommendation  # Import generate_recommendation
from db import insert_history, insert_evaluation_history  # Import fungsi dari db.py

MODELS_DIR = "saved_models"  # Sesuaikan path
DATASETS_DIR = "datasets"  # Sesuaikan path

def compare_models_page(conn, cursor):
    st.header("Compare Models")
    file_name = st.selectbox("Select a dataset for comparison", os.listdir(DATASETS_DIR))
    df = load_dataset(file_name) if file_name else None

    if df is not None:
        if 'comparison_model' not in st.session_state:
            st.session_state.comparison_model = []

        # Model Selection for Comparison
        model_options = [model.split(".")[0] for model in os.listdir(MODELS_DIR)]
        selected_models = st.multiselect("Select models for comparison", model_options)

        if st.button("Compare Models"):
            if selected_models:
                # Ensure that the selected models are loaded
                models = {} # Simpan model yang sudah di load
                for model_name in selected_models:
                    if model_name == "sarimax_model":
                        model = load_arima_model()
                    elif "model_fold" in model_name:
                        fold = model_name.split("_")[-1]
                        model = load_transformer_model(int(fold))
                    elif model_name == "final_model": # Tambahkan ini
                        model = load_transformer_model_final() # Tambahkan ini
                    else :
                        st.error(f"Model {model_name} is not available.")
                        st.stop()

                    if model:
                        models[model_name] = model # Simpan model yang berhasil di load

                # Perform Evaluation for Each Selected Model
                results = {}
                dataset_name = file_name

                # Fetch dataset_id
                cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = ? ORDER BY dataset_id DESC LIMIT 1", (dataset_name,))
                result = cursor.fetchone()
                dataset_id = result[0] if result else None

                for model_name, model in models.items(): # Iterasi melalui model yang sudah di load
                    if "sarimax_model" in model_name:
                        # Fetch model_id
                        cursor.execute("SELECT model_id FROM models WHERE algorithm = 'ARIMA' ORDER BY model_id DESC LIMIT 1")
                        result = cursor.fetchone()
                        model_id = result[0] if result else None

                        metrics, y_pred_original = evaluate_arima_model(df, forecast_months = 12)
                        if metrics is not None and y_pred_original is not None:
                            results[model_name] = {
                                "metrics": metrics,
                                "y_pred" : y_pred_original,
                                "y_actual" : df["jumlah_kasus"].values[-len(y_pred_original):]
                            }

                            # Insert history and evaluation data
                            if model_id is not None and dataset_id is not None:
                                if insert_history(cursor, model_id, dataset_id, y_pred_original.tolist()):  # Convert NumPy array to list
                                    conn.commit()
                                    st.success(f"Prediction {model_name} added to database.")
                                else:
                                    st.error(f"Failed to add prediction {model_name} to database.")

                                if insert_evaluation_history(cursor, model_id, dataset_id, metrics):
                                    conn.commit()
                                    st.success(f"Evaluation {model_name} added to database.")
                                else:
                                    st.error(f"Failed to add evaluation {model_name} to database.")
                            else:
                                st.error("model_id or dataset_id not found.")

                    elif "model_fold" in model_name:
                        # Fetch model_id
                        cursor.execute("SELECT model_id FROM models WHERE model_name = ? ORDER BY model_id DESC LIMIT 1", (model_name,))
                        result = cursor.fetchone()
                        model_id = result[0] if result else None
                        fold = model_name.split("_")[-1]
                        metrics, y_pred = evaluate_transformer(df, int(fold), forecast_months = 12)
                        if metrics is not None and y_pred is not None:
                            results[model_name] = {
                                "metrics": metrics,
                                "y_pred" : y_pred,
                                "y_actual" : df["jumlah_kasus"].tolist()[len(df["jumlah_kasus"])-len(y_pred):]
                            }

                            # Insert history and evaluation data
                            if model_id is not None and dataset_id is not None:
                                if insert_history(cursor, model_id, dataset_id, y_pred.tolist()):  # Convert NumPy array to list
                                    conn.commit()
                                    st.success(f"Prediction {model_name} added to database.")
                                else:
                                    st.error(f"Failed to add prediction {model_name} to database.")

                                if insert_evaluation_history(cursor, model_id, dataset_id, metrics):
                                    conn.commit()
                                    st.success(f"Evaluation {model_name} added to database.")
                                else:
                                    st.error(f"Failed to add evaluation {model_name} to database.")
                            else:
                                st.error("model_id or dataset_id not found.")

                    elif "final_model" in model_name:
                        # Fetch model_id
                        cursor.execute("SELECT model_id FROM models WHERE model_name = ? ORDER BY model_id DESC LIMIT 1", (model_name,))
                        result = cursor.fetchone()
                        model_id = result[0] if result else None

                        y_pred = transformer_prediction_final(df, forecast_months = 12) # Menggunakan fungsi prediksi baru
                        y_actual = df["jumlah_kasus"].tolist()[-len(y_pred):]

                        if y_pred is not None and len(y_pred) > 0: # Memastikan y_pred tidak kosong dan memiliki elemen
                            metrics = evaluate_predictions(y_actual, y_pred) # Evaluasi hasil prediksi
                            results[model_name] = {
                                "metrics": metrics,
                                "y_pred": y_pred,
                                "y_actual": y_actual
                            }

                            # Insert history and evaluation data
                            if model_id is not None and dataset_id is not None:
                                if insert_history(cursor, model_id, dataset_id, y_pred):  # Convert NumPy array to list
                                    conn.commit()
                                    st.success(f"Prediction {model_name} added to database.")
                                else:
                                    st.error(f"Failed to add prediction {model_name} to database.")

                                if insert_evaluation_history(cursor, model_id, dataset_id, metrics):
                                    conn.commit()
                                    st.success(f"Evaluation {model_name} added to database.")
                                else:
                                    st.error(f"Failed to add evaluation {model_name} to database.")
                            else:
                                st.error("model_id or dataset_id not found.")

                # Display Comparison Results
                if results:
                    st.subheader("Model Comparison Results")

                    # Create a DataFrame for the comparison table
                    comparison_data = []
                    model_names = list(results.keys())
                    for model_name, model_result in results.items():
                        recommendation = generate_recommendation(model_result["metrics"], model_result["y_pred"], model_result["y_actual"])
                        comparison_data.append({
                            "Model": model_name,
                            "RMSE": model_result["metrics"].get("RMSE"),
                            "MAE": model_result["metrics"].get("MAE"),
                            "R2": model_result["metrics"].get("R2"),
                            "MAPE": model_result["metrics"].get("MAPE"),
                            "Recommendation": recommendation
                        })

                    # Display table
                    comparison_df = pd.DataFrame(comparison_data)

                    # Reorder columns to put 'Model' first
                    column_order = ["Model", "RMSE", "MAE", "R2", "MAPE", "Recommendation"]
                    comparison_df = comparison_df[column_order]

                    # Round numeric values
                    numeric_cols = ["RMSE", "MAE", "R2", "MAPE"]
                    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(2)

                    st.dataframe(comparison_df)

                    # Model recommendation
                    best_model = comparison_df.sort_values(by="RMSE").iloc[0]
                    st.subheader("Recommendation")

                    # Detail explanation
                    st.write(f"Based on the results, the model with the lowest RMSE is **{best_model['Model']}**.")
                    if "final_model" in best_model['Model']:
                        st.write(f"The **final_model** refers to the **Transformer model** which often excels in capturing complex temporal dependencies.")
                    elif "sarimax_model" in best_model['Model']:
                        st.write(f"The **sarimax_model** refers to the **ARIMA model** which is known for its simplicity and effectiveness in modeling linear time series data.")

                    st.write(f"Therefore, we recommend using **{best_model['Model']}** for prediction.")

                    # Plot Predictions for selected model
                    selected_model_for_plot = st.selectbox("Select a model to plot", model_names)
                    st.write(f"Showing plot for **{selected_model_for_plot}**")
                    model_result = results[selected_model_for_plot]

                    plot_predictions(model_result["y_actual"], model_result["y_pred"], title=f"Actual vs Predicted for {selected_model_for_plot}")

                    # Display Evaluation Metrics for each model
                    st.subheader("Evaluation Metrics for each Model:")
                    for model_name, model_result in results.items():
                        st.write(f"**{model_name}**: {model_result['metrics']}")

                else:
                    st.error("Please select at least one model to compare.")
