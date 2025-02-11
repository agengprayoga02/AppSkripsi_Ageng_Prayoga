# models.py
import streamlit as st
import os
import pandas as pd
import sqlite3
from db import hash_password, check_password, is_valid_email, get_all_models, get_model_by_id, update_model, delete_model
from utils import load_dataset

MODELS_DIR = "saved_models"  # Sesuaikan path

def models_page(conn, cursor, username):
    st.header("Models")
    st.subheader(f"Hello, {username}")

    st.subheader("Available Models")
    all_models = get_all_models(cursor)
    registered_models = {model[1]: model[0] for model in all_models} # {model_name: model_id}

    # Display available models from the MODELS_DIR
    files = os.listdir(MODELS_DIR)
    if files:
        st.write("Models in directory:")
        for file_name in files:
            model_name = file_name.split(".")[0]  # Remove .h5 or .pkl extension
            if model_name in registered_models:
                model_id = registered_models[model_name]
                st.write(f"- ID: {model_id}, Name: {model_name} (Registered)")
            else:
                st.write(f"- Name: {model_name} (Not Registered)")

    else:
        st.write("No models found in the directory.")

    st.subheader("Register New Model")
    new_model_name = st.text_input("New Model Name")
    new_algorithm = st.selectbox("Algorithm", ["ARIMA", "Transformer"], key="new_algorithm_selectbox") # Tambahkan key
    new_parameters = st.text_input("Parameters (e.g., p,d,q values for ARIMA)")
    # new_file_path = st.file_uploader("Upload File", type=["h5", "pkl"])
    if st.button("Add Model"):
        if new_model_name and new_algorithm:
            user_id = st.session_state.user_id
            try:
                # Get the smallest available model_id
                cursor.execute("SELECT COALESCE(MAX(model_id), 0) + 1 FROM models")
                model_id = cursor.fetchone()[0]

                cursor.execute(
                    "INSERT INTO models (model_id, model_name, algorithm, parameters, created_by) VALUES (?, ?, ?, ?, ?)",
                    (model_id, new_model_name, new_algorithm, new_parameters, user_id),
                )
                conn.commit()
                st.success(f"Model '{new_model_name}' added to database.")
            except sqlite3.Error as e:
                st.error(f"Failed to add model to database: {e}")

    # Model Management
    st.subheader("Manage Models")
    all_models = get_all_models(cursor)  # Refresh models after adding new one
    if all_models:
        st.write("All Registered Models:")
        for model_id, model_name, algorithm, created_at in all_models:
            st.write(f"- ID: {model_id}, Name: {model_name}, Algorithm: {algorithm}, Created At: {created_at}")

        # Pastikan ada model sebelum meminta ID
        if all_models:
            default_model_id = all_models[0][0]  # Gunakan model ID pertama sebagai default
        else:
            default_model_id = 1  # Jika tidak ada model, gunakan default 1 atau nilai yang sesuai

        model_id_to_update = st.number_input("Select model ID to update", min_value=1, step=1, value=default_model_id, key="update_model", format="%d")

        # Validasi bahwa model ID ada
        model_ids = [model[0] for model in all_models]
        if int(model_id_to_update) in model_ids:
            model_details = get_model_by_id(cursor, model_id_to_update)

            if model_details:
                st.write(f"Details of model {model_id_to_update}")
                st.write(f"model_id: {model_details[0]}, model_name: {model_details[1]}, algorithm: {model_details[2]}")

                update_model_name = st.text_input("New Model Name", value=model_details[1])
                update_algorithm = st.selectbox("Algorithm", ["ARIMA", "Transformer"], index=["ARIMA", "Transformer"].index(model_details[2]), key="update_algorithm_selectbox") # Tambahkan key
                update_parameters = st.text_input("New Parameters", value=model_details[3] if model_details[3] else "")
                if st.button("Update Model", key="update_model_button"):
                    user_id = st.session_state.user_id
                    if update_model(
                        conn, cursor, model_id_to_update, update_model_name, update_algorithm, update_parameters, user_id
                    ):
                        #conn.commit()
                        st.success("Model updated successfully")
                    else:
                        st.error("Failed to update the model")

                model_id_to_delete = st.number_input("Select model ID to delete", min_value=1, step=1, value=1, key="delete_model_id", format="%d")
                if st.button("Delete Model", key="delete_model_button"):
                    user_id = st.session_state.user_id
                    if delete_model(conn, cursor, model_id_to_delete, user_id):
                        #conn.commit()
                        st.success("Model deleted successfully")
                    else:
                        st.error("Failed to delete model")

            else:
                st.error("Model Not Found")
        else:
            st.error(f"Model with ID {model_id_to_update} not found.")

    else:
        st.write("No models found in the database.")

# CRUD functions for models
def get_all_models(cursor):
    cursor.execute("SELECT model_id, model_name, algorithm, created_at FROM models")
    return cursor.fetchall()

def get_model_by_id(cursor, model_id):
    cursor.execute("SELECT model_id, model_name, algorithm, parameters, created_at FROM models WHERE model_id = ?", (model_id,))
    return cursor.fetchone()

def update_model(conn, cursor, model_id, model_name, algorithm, parameters, user_id):
    try:
        cursor.execute("UPDATE models SET model_name = ?, algorithm = ?, parameters = ? WHERE model_id = ? AND created_by = ?", (model_name, algorithm, parameters, model_id, user_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Failed to update model: {e}")
        return False
def delete_model(conn, cursor, model_id, user_id):
     try:
         cursor.execute("DELETE FROM models WHERE model_id = ? AND created_by = ?", (model_id, user_id))
         conn.commit()
         return True
     except sqlite3.Error as e:
         st.error(f"Failed to delete model: {e}")
         return False
