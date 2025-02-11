import streamlit as st
from db import get_all_history

def history_page(conn, cursor):
    st.header("Prediction and Evaluation History")
    history = get_all_history(cursor)
    if history:
        st.write("All History:")
        for (prediction_id, prediction_date, results, model_name, algorithm, dataset_name, metrics, evaluated_at) in history:
            st.write("---")
            st.write(f"**Prediction ID:** {prediction_id}")
            st.write(f"**Prediction Date:** {prediction_date}")
            st.write(f"**Model:** {model_name} ({algorithm})")
            st.write(f"**Dataset:** {dataset_name}")
            st.write(f"**Prediction Results:** {results}")
            if metrics:
              st.write(f"**Evaluation Metrics:** {metrics}")
              st.write(f"**Evaluated At:** {evaluated_at}")
    else:
        st.write("No Prediction or Evaluation history found")