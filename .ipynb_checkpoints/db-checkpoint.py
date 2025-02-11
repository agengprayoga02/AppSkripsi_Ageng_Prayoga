import sqlite3
import bcrypt
import re
import os
import streamlit as st
import datetime

DATABASE_PATH = os.path.join("C:/Users/ASUS/Aplikasi_Skripsi", "dbd_prediction.db")  # Path ke database SQLite

# SQLite Database Setup
def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Create Tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL UNIQUE,
                file_path TEXT NOT NULL,
                description TEXT,
                uploaded_by INTEGER NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (uploaded_by) REFERENCES users (user_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                model_name TEXT NOT NULL,
                algorithm TEXT CHECK(algorithm IN ('ARIMA', 'Transformer')) NOT NULL,
                parameters TEXT,
                created_by INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users (user_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                dataset_id INTEGER NOT NULL,
                prediction_date DATE NOT NULL,
                results TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (model_id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                dataset_id INTEGER NOT NULL,
                metrics TEXT NOT NULL,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (model_id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        """)
        conn.commit()
        print("Database initialized or already exists.")
        return conn, cursor
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.close()
        return None, None

# Fungsi untuk hashing password dengan bcrypt
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8'), salt.decode('utf-8')

# Fungsi untuk verifikasi password dengan bcrypt
def check_password(password, hashed_password, salt):
    try:
        salted_password = bcrypt.hashpw(password.encode('utf-8'), salt.encode('utf-8'))
        return salted_password.decode('utf-8') == hashed_password
    except ValueError:
        return False  # Return False if hash format is invalid

# Fungsi untuk validasi email
def is_valid_email(email):
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(email_regex, email)

# CRUD functions for datasets
def get_all_datasets(cursor):
    cursor.execute("SELECT dataset_id, dataset_name, file_path, uploaded_at FROM datasets")
    return cursor.fetchall()

def get_dataset_by_id(cursor, dataset_id):
    cursor.execute("SELECT dataset_id, dataset_name, file_path, description, uploaded_at FROM datasets WHERE dataset_id = ?", (dataset_id,))
    return cursor.fetchone()

def get_dataset_by_name(cursor, dataset_name):
    cursor.execute("SELECT dataset_id, dataset_name, file_path, description, uploaded_at FROM datasets WHERE dataset_name = ?", (dataset_name,))
    return cursor.fetchone()

def update_dataset(cursor, dataset_id, dataset_name, file_path, description, user_id):
    try:
        cursor.execute("UPDATE datasets SET dataset_name = ?, file_path = ?, description = ? WHERE dataset_id = ? AND uploaded_by = ?", (dataset_name, file_path, description, dataset_id, user_id))
        return True
    except sqlite3.Error as e:
        print(f"Failed to update dataset: {e}")
        return False

def delete_dataset(cursor, dataset_id, user_id):
    try:
        cursor.execute("DELETE FROM datasets WHERE dataset_id = ? AND uploaded_by = ?", (dataset_id, user_id))
        return True
    except sqlite3.Error as e:
        print(f"Failed to delete dataset: {e}")
        return False

# CRUD functions for models
def get_all_models(cursor):
    cursor.execute("SELECT model_id, model_name, algorithm, created_at FROM models")
    return cursor.fetchall()

def get_model_by_id(cursor, model_id):
    cursor.execute("SELECT model_id, model_name, algorithm, parameters, created_at FROM models WHERE model_id = ?", (model_id,))
    return cursor.fetchone()

def update_model(cursor, model_id, model_name, algorithm, parameters, user_id):
    try:
        cursor.execute("UPDATE models SET model_name = ?, algorithm = ?, parameters = ? WHERE model_id = ? AND created_by = ?", (model_name, algorithm, parameters, model_id, user_id))
        return True
    except sqlite3.Error as e:
        print(f"Failed to update model: {e}")
        return False
def delete_model(cursor, model_id, user_id):
     try:
         cursor.execute("DELETE FROM models WHERE model_id = ? AND created_by = ?", (model_id, user_id))
         return True
     except sqlite3.Error as e:
         print(f"Failed to delete model: {e}")
         return False

# History functions
def insert_history(cursor, model_id, dataset_id, results):
    """
    Menambahkan history prediksi ke database.
    :param cursor: Cursor database
    :param model_id: ID model yang digunakan
    :param dataset_id: ID dataset yang digunakan
    :param results: Hasil prediksi
    """
    import datetime
    prediction_date = datetime.date.today().strftime('%Y-%m-%d')
    try:
        cursor.execute("INSERT INTO predictions (model_id, dataset_id, prediction_date, results) VALUES (?, ?, ?, ?)", (model_id, dataset_id, prediction_date, str(results)))
        return True
    except sqlite3.Error as e:
       print(f"Failed to add prediction to database: {e}")
       return False
        
def insert_evaluation_history(cursor, model_id, dataset_id, metrics):
    """
    Menambahkan history evaluasi ke database.
    :param cursor: Cursor database
    :param model_id: ID model yang digunakan
    :param dataset_id: ID dataset yang digunakan
    :param metrics: Hasil evaluasi
    """
    try:
        cursor.execute("INSERT INTO evaluations (model_id, dataset_id, metrics) VALUES (?, ?, ?)", (model_id, dataset_id, str(metrics)))
        return True
    except sqlite3.Error as e:
       print(f"Failed to add evaluation to database: {e}")
       return False

def get_all_history(cursor):
    """
    Mengambil semua history prediksi dan evaluasi dari database.
    :param cursor: Cursor database
    :return: List of tuples, setiap tuple berisi info prediksi dan evaluasi.
    """
    cursor.execute("""
        SELECT 
            p.prediction_id,
            p.prediction_date,
            p.results,
            m.model_name,
            m.algorithm,
            d.dataset_name,
            e.metrics,
            e.evaluated_at
        FROM predictions p
        INNER JOIN models m ON p.model_id = m.model_id
        INNER JOIN datasets d ON p.dataset_id = d.dataset_id
        LEFT JOIN evaluations e ON p.prediction_id = e.model_id AND p.dataset_id = e.dataset_id  -- Menggunakan p.prediction_id sebagai foreign key
        ORDER BY p.prediction_date DESC
    """)
    return cursor.fetchall()

def get_smallest_available_model_id(cursor):
    """
    Mencari ID model terkecil yang tersedia (tidak digunakan).
    :param cursor: Cursor database
    :return: ID model terkecil yang tersedia, atau 1 jika tidak ada model yang terdaftar.
    """
    cursor.execute("SELECT model_id FROM models ORDER BY model_id ASC")
    existing_ids = [row[0] for row in cursor.fetchall()]

    if not existing_ids:
        return 1  # Jika tidak ada model, mulai dari ID 1

    smallest_available_id = 1
    while smallest_available_id in existing_ids:
        smallest_available_id += 1

    return smallest_available_id