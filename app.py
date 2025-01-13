import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import hashlib
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from itertools import product

# Paths
BASE_DIR = "C:/Users/ASUS/Aplikasi_Skripsi"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATABASE_PATH = os.path.join(BASE_DIR, "dbd_prediction.db")  # Path ke database SQLite

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                description TEXT,
                uploaded_by INTEGER NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (uploaded_by) REFERENCES users (user_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_by INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users (user_id)
            )
        """)
        conn.commit()
        print("Database initialized or already exists.")
        return conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        if conn:
             conn.close()
        return None

# Function to hash the password
def hash_password(password):
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

# Helper function to clear model state
def clear_model_state():
    for key in st.session_state.keys():
      if key.startswith("transformer_model_fold_"):
        del st.session_state[key]
    if 'y_pred' in st.session_state:
        del st.session_state['y_pred']

# --- Load Data ---
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['periode'] = pd.to_datetime(df['periode'], format="%Y-%m")
    df.set_index('periode', inplace=True)
    return df

# --- Data Normalization ---
def scale_data(df, feature_column='jumlah_kasus'):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature_column]])
    return scaled_data, scaler

# --- Stationarity Check (For ARIMA) ---
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    if result[1] <= 0.05:
        print("Data sudah stasioner.")
        return timeseries
    else:
        print("Data tidak stasioner, melakukan differencing.")
        return pd.DataFrame(timeseries).diff().dropna().values # Return differenced data
# --- Split Data (ARIMA) ---
def split_data_arima(scaled_data, train_ratio=0.8):
    train_size = int(len(scaled_data) * train_ratio)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    return train_data, test_data

# --- Create Sequences (For Informer) ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# --- Split Data for Cross-Validation (Informer) ---
from sklearn.model_selection import KFold
def split_data_informer(X, y, n_splits=5, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return kf.split(X)
    
# Load Dataset
def load_dataset(file_name):
    file_path = os.path.join(DATASETS_DIR, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.write(f"Missing values in dataset before preprocessing: {df.isna().sum().sum()}")
        if df["jumlah_kasus"].isna().any():
            st.error("Kolom 'jumlah_kasus' memiliki NaN. Mohon perbaiki dataset terlebih dahulu.")
            return None
        return df
    else:
        st.error(f"File {file_name} not found in {DATASETS_DIR}")
        return None

def preprocess_dataset_informer(df, target_col="jumlah_kasus", seq_length=12):
        """
        Preprocess dataset for Informer model, by performing only numeric conversion and
        handle NaNs, scaling and sequence creation done separately.
        """
        x_data = df.copy()
        # Convert all columns to numeric
        x_data = x_data.apply(pd.to_numeric, errors="coerce")

        # Handle missing values
        if x_data.isna().any().any():
            st.warning("Dataset contains NaN values before processing.")

            # Drop columns with all NaN values
            cols_to_drop = [col for col in x_data.columns if x_data[col].isna().all()]
            if cols_to_drop:
                st.write(f"Dropping columns with all NaN values: {cols_to_drop}")
                x_data = x_data.drop(cols_to_drop, axis=1)
                if 'periode' in cols_to_drop:
                    st.warning("Kolom 'periode' dihapus karena hanya mengandung nilai NaN.")

            # Fill remaining NaNs with column means
            x_data.fillna(x_data.mean(), inplace=True)

        if x_data.empty:
            st.error("No valid columns after preprocessing.")
            return None

        st.write(f"NaN values after filling: {x_data.isna().sum().sum()}")
        y_data_true = df[target_col].values.astype(np.float32)
        return x_data, y_data_true

# Load Models
def load_arima_model():
    model_path = os.path.join(MODELS_DIR, "sarimax_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        return model_dict
    else:
        st.error("ARIMA model not found.")
        return None

def load_transformer_model(fold):
    model_path = os.path.join(MODELS_DIR, f"model_fold_{fold}.h5")
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"Transformer model fold {fold} not found.")
        return None

# Evaluate Model
def evaluate_predictions(y_true, y_pred):
    # Validasi NaN
    if any(pd.isna(y_true)) or any(pd.isna(y_pred)):
        st.write("y_true:", y_true)
        st.write("y_pred:", y_pred)
        raise ValueError("y_true or y_pred contains NaN. Ensure all inputs are valid.")
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if np.any(y_true == 0):
        mape = np.nan  # Set MAPE to NaN if y_true contains zero
    else:
       mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# Plot Visualization
def plot_predictions(y_true, y_pred, title="Actual vs Predicted", xlabel="Periode", ylabel="Jumlah Kasus"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", marker="o", color="blue")
    plt.plot(y_pred, label="Predicted", marker="x", color="orange")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    st.pyplot(plt)

def evaluate_arima_model(df):
    # Load Model
    model_dict = load_arima_model()
    if not model_dict:
        return None, None
    model = model_dict["model"]
    scaler = model_dict["scaler"]
   
    y_actual = df["jumlah_kasus"].values[-12:].astype(np.float32)  # Data aktual (12 bulan terakhir)
    
    # Preprocessing data for ARIMA 
    scaled_data, _ = scale_data(df)
    scaled_data = check_stationarity(scaled_data)

    if scaled_data is not None:
        # split the train data
        train_data, test_data = split_data_arima(scaled_data)
        y_pred = model.forecast(len(y_actual))  # Prediksi ARIMA
        y_pred_original = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()

        # Evaluasi
        metrics = evaluate_predictions(y_actual, y_pred_original)

        return metrics, y_pred_original
    else:
        st.error("Data preprocessing failed.")
        return None, None

def transformer_prediction(df, fold):
    model = load_transformer_model(fold)
    if not model:
         return None
    
    # Preprocessing Data
    x_data, _ = preprocess_dataset_informer(df)
    scaled_data, scaler_informer = scale_data(pd.DataFrame(x_data))
    X, _ = create_sequences(scaled_data, seq_length=12)

    if X is None:
        st.error("Error during data preprocessing for Transformer model")
        return None
    
    y_pred = model.predict(X).flatten()
    if np.isnan(y_pred).any():
         st.error("Output model contains NaN, please check the model.")
         return None

    y_pred = scaler_informer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    if isinstance(y_pred, np.ndarray):
        return y_pred
    else:
        return np.array([y_pred])  # Pastikan selalu array

def evaluate_transformer(df, fold):
    model = load_transformer_model(fold)
    
    if not model:
        return None, None
    
    # Preprocessing data
    x_data, y_true = preprocess_dataset_informer(df)
    scaled_data, scaler_informer = scale_data(pd.DataFrame(x_data))
    X, y = create_sequences(scaled_data, seq_length=12)
    y_true = y_true[12:]
    if X is None:
        st.error("Error during data preprocessing for Transformer model")
        return None, None
    
    # Check for NaNs in input data
    if np.isnan(X).any() or np.isnan(y_true).any():
        st.error("Input data contains NaN values after preprocessing.")
        return None, None
    
    # Reshape x_data to fit the model's expected input shape
    y_pred = model.predict(X).flatten()
    
    # Validate predictions
    if np.isnan(y_pred).any():
        st.error("Output model contains NaN. Please check the model.")
        return None, None

    y_pred = scaler_informer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    metrics = evaluate_predictions(y_true[len(y_true) - len(y_pred):], y_pred)  # adjust y_true
    
    return metrics, y_pred

# Function for EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")

    # 1. Memahami Struktur Dataset
    st.markdown("### 1. Struktur Dataset")
    st.write("Beberapa baris pertama dataset:")
    st.write(df.head())
    st.write(f"Dimensi dataset: {df.shape}")
    st.write("Informasi dataset:")
    buffer = df.info(memory_usage=False)
    st.text(buffer)
    st.write("Deskripsi statistik:")
    st.write(df.describe())

    # 2. Analisis Data Kategorikal
    st.markdown("### 2. Analisis Data Kategorikal")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.write(f"Distribusi nilai di kolom {col}:")
            st.write(df[col].value_counts())
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, y=col)
            st.pyplot(plt)
    else:
        st.write("Tidak ada kolom kategorikal dalam dataset.")

    # 3. Analisis Data Numerik
    st.markdown("### 3. Analisis Data Numerik")
    numerical_cols = df.select_dtypes(include=np.number).columns
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            st.write(f"Distribusi kolom numerik {col}:")
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            st.pyplot(plt)

            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            st.pyplot(plt)
            
            st.write(f"Statistik deskriptif kolom {col}:")
            st.write(df[col].describe())
    else:
        st.write("Tidak ada kolom numerik dalam dataset.")
    
    # 4. Analisis Korelasi
    st.markdown("### 4. Analisis Korelasi")
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt)

        st.write("Scatter plot untuk melihat hubungan antar dua variabel:")
        col1 = st.selectbox("Pilih kolom 1 untuk scatter plot", numerical_cols)
        col2 = st.selectbox("Pilih kolom 2 untuk scatter plot", numerical_cols, index = 1 if len(numerical_cols) > 1 else 0)
        if col1 and col2 and col1 != col2:
           plt.figure(figsize=(8,6))
           sns.scatterplot(x=df[col1],y=df[col2])
           st.pyplot(plt)
        elif col1 == col2:
           st.error("Pilih dua kolom yang berbeda")

        st.write("Pairplot untuk melihat hubungan antar beberapa variabel numerik:")
        sns.pairplot(df[numerical_cols])
        st.pyplot(plt)
    else:
        st.write("Tidak cukup kolom numerik untuk analisis korelasi.")


    # 5. Analisis Missing Values
    st.markdown("### 5. Analisis Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.write("Persentase data yang hilang untuk setiap kolom:")
        st.write((missing_values / len(df) * 100).sort_values(ascending=False))

        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
        st.pyplot(plt)
    else:
        st.write("Tidak ada missing values dalam dataset.")
        

    # 6. Analisis Outlier
    st.markdown("### 6. Analisis Outlier")
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            st.write(f"Outlier pada kolom {col}:")
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            st.pyplot(plt)
    else:
      st.write("Tidak ada kolom numerik untuk analisis outlier.")
    
# CRUD functions for reports
def get_all_reports(cursor):
    cursor.execute("SELECT report_id, title, content, created_at FROM reports")
    return cursor.fetchall()

def get_report_by_id(cursor, report_id):
    cursor.execute("SELECT report_id, title, content, created_at FROM reports WHERE report_id = ?", (report_id,))
    return cursor.fetchone()

def update_report(cursor, report_id, title, content, user_id):
    try:
        cursor.execute("UPDATE reports SET title = ?, content = ? WHERE report_id = ? AND created_by = ?", (title, content, report_id, user_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Failed to update report: {e}")
        return False

def delete_report(cursor, report_id, user_id):
    try:
        cursor.execute("DELETE FROM reports WHERE report_id = ? AND created_by = ?", (report_id, user_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Failed to delete report: {e}")
        return False

# CRUD functions for datasets
def get_all_datasets(cursor):
    cursor.execute("SELECT dataset_id, dataset_name, file_path, uploaded_at FROM datasets")
    return cursor.fetchall()

def get_dataset_by_id(cursor, dataset_id):
    cursor.execute("SELECT dataset_id, dataset_name, file_path, description, uploaded_at FROM datasets WHERE dataset_id = ?", (dataset_id,))
    return cursor.fetchone()

def update_dataset(cursor, dataset_id, dataset_name, file_path, description, user_id):
    try:
        cursor.execute("UPDATE datasets SET dataset_name = ?, file_path = ?, description = ? WHERE dataset_id = ? AND uploaded_by = ?", (dataset_name, file_path, description, dataset_id, user_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Failed to update dataset: {e}")
        return False

def delete_dataset(cursor, dataset_id, user_id):
     try:
         cursor.execute("DELETE FROM datasets WHERE dataset_id = ? AND uploaded_by = ?", (dataset_id, user_id))
         conn.commit()
         return True
     except sqlite3.Error as e:
         st.error(f"Failed to delete dataset: {e}")
         return False


def main():
    # Initialize database connection
    conn = init_db()
    if conn is None:
      st.error("Failed to connect to the database.")
      return
    cursor = conn.cursor()
    
    st.title("DBD Prediction Application")

    # Initialize session state for authentication
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
         st.session_state.username = None
    
    # Admin Profile
    if st.session_state.logged_in:
        if st.sidebar.button(f"Profile: {st.session_state.username}"):
             st.sidebar.write(f"User ID: {st.session_state.user_id}")
             if st.sidebar.button("Logout"):
               st.session_state.logged_in = False
               st.session_state.user_id = None
               st.session_state.username = None
               st.experimental_rerun()
    
    # Authentication Logic
    if not st.session_state.logged_in:
        auth_page = st.radio("Select", ["Register", "Login"])
        
        if auth_page == "Register":
            st.header("Register")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_email = st.text_input("Email")

            if st.button("Register"):
                if new_username and new_password and new_email:
                    try:
                        hashed_password = hash_password(new_password)
                        cursor.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (new_username, hashed_password, new_email))
                        conn.commit()
                        st.success("Registration successful. Please log in.")
                        st.write("Already registered? Go to the Login tab.")
                    except sqlite3.IntegrityError:
                        st.error("Username or email already exists.")
                else:
                    st.error("Please fill all the fields.")
        elif auth_page == "Login":
            st.header("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username and password:
                    hashed_password = hash_password(password)
                    cursor.execute("SELECT user_id, username, password_hash FROM users WHERE username = ?", (username,))
                    user = cursor.fetchone()
                    if user and user[2] == hashed_password:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user[0]
                        st.session_state.username = user[1]
                        st.success("Login successful")
                        st.experimental_rerun()
                    else:
                         st.error("Invalid username or password")
                else:
                    st.error("Please enter username and password")
        return # Exit main if not logged in

    # Navigation if logged in
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Datasets", "Models", "Prediction", "Evaluation", "Reports"])
    
    if page == "Datasets":
        clear_model_state()
        st.header("Available Datasets")
        datasets = os.listdir(DATASETS_DIR)
        if datasets:
            for dataset in datasets:
                st.write(f"- {dataset}")
            file_name = st.selectbox("Select a dataset to preview", datasets)
            if st.button("Load Dataset"):
                df = load_dataset(file_name)
                if df is not None:
                   st.write(df.head())
                   perform_eda(df)
                   # Insert into database
                   try:
                        # Assume a hardcoded admin ID (you may need a better solution for admin selection)
                        user_id = st.session_state.user_id
                        cursor.execute("INSERT INTO datasets (dataset_name, file_path, uploaded_by) VALUES (?, ?, ?)", (file_name, os.path.join(DATASETS_DIR, file_name), user_id))
                        conn.commit()
                        st.success(f"Dataset {file_name} metadata added to database.")
                   except sqlite3.Error as e:
                        st.error(f"Failed to add dataset metadata to database: {e}")
            
            # Dataset Management
            st.subheader("Manage Datasets")
            all_datasets = get_all_datasets(cursor)
            if all_datasets:
                st.write("All Datasets:")
                for dataset_id, dataset_name, file_path, uploaded_at in all_datasets:
                     st.write(f"- ID: {dataset_id}, Name: {dataset_name}, Path: {file_path}, Uploaded At: {uploaded_at}")
                dataset_id_to_update = st.number_input("Select dataset ID to Update", min_value = 1, step = 1, value = 1, key = "update_dataset")
                dataset_details = get_dataset_by_id(cursor, dataset_id_to_update)

                if dataset_details:
                   st.write(f"Details of dataset {dataset_id_to_update}")
                   st.write(f"dataset_id: {dataset_details[0]}, dataset_name: {dataset_details[1]}, file_path: {dataset_details[2]}")
                   
                   update_dataset_name = st.text_input("New Dataset Name", value = dataset_details[1])
                   update_file_path = st.text_input("New file_path", value = dataset_details[2])
                   update_description = st.text_input("New Description", value = dataset_details[3] if dataset_details[3] else "")

                   if st.button("Update Dataset", key = "update_dataset_button"):
                         user_id = st.session_state.user_id
                         if update_dataset(cursor, dataset_id_to_update, update_dataset_name, update_file_path, update_description, user_id):
                             st.success("Dataset updated successfully")
                         else:
                            st.error("Failed to update the Dataset")
                   
                   dataset_id_to_delete = st.number_input("Select Dataset ID to delete", min_value=1, step=1, value = 1, key = "delete_dataset_id")
                   if st.button("Delete Dataset", key = "delete_dataset_button"):
                     user_id = st.session_state.user_id
                     if delete_dataset(cursor, dataset_id_to_delete, user_id):
                         st.success("Dataset deleted successfully")
                     else:
                         st.error("Failed to delete dataset")
                else:
                   st.error("Dataset Not Found")
        else:
            st.write("No datasets found in the directory.")

    elif page == "Models":
        clear_model_state()
        st.header("Available Models")
        models = os.listdir(MODELS_DIR)
        if models:
            for model in models:
                st.write(f"- {model}")
        else:
            st.write("No models found in the directory.")
        model_name = st.text_input("Model Name")
        algorithm = st.selectbox("Choose Algorithm", ["ARIMA", "Transformer"])
        if st.button("Register Model"):
           if model_name and algorithm:
              try:
                   # Assume a hardcoded admin ID (you may need a better solution for admin selection)
                  user_id = st.session_state.user_id
                  cursor.execute("INSERT INTO models (model_name, algorithm, created_by) VALUES (?, ?, ?)", (model_name, algorithm, user_id))
                  conn.commit()
                  st.success(f"Model {model_name} metadata added to database.")
              except sqlite3.Error as e:
                  st.error(f"Failed to register model: {e}")
           else:
                st.error("Please fill in all required fields.")
    
    elif page == "Prediction":
        st.header("Prediction")
        algo = st.selectbox("Choose Algorithm", ["ARIMA", "Transformer"])
        file_name = st.selectbox("Select a dataset for prediction", os.listdir(DATASETS_DIR))
        df = load_dataset(file_name) if file_name else None
        
        if df is not None:
            # Validasi NaN di y_true
            y_true = df["jumlah_kasus"].tolist()
            if any(pd.isna(y_true)):
                st.error("Dataset contains NaN in target column 'jumlah_kasus'. Please clean the data.")
                st.stop()
            
        if df is not None:
          if algo == "ARIMA":
                if st.button("Predict", key = "predict_arima"):
                    model_dict = load_arima_model()
                    if model_dict:
                        model = model_dict["model"]  # Ambil model dari dictionary
                        
                        # Preprocess data for ARIMA
                        scaled_data, scaler_arima = scale_data(df)
                        scaled_data = check_stationarity(scaled_data)

                        if scaled_data is not None:
                          train_data, test_data = split_data_arima(scaled_data)
                          y_pred = model.forecast(len(y_true))  # ARIMA prediksi

                          # Inverse transform menggunakan scaler
                          y_pred_original_scale = scaler_arima.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()

                          st.success("ARIMA prediction completed.")

                          # Validasi NaN pada y_pred
                          if any(pd.isna(y_pred_original_scale)):
                              st.error("ARIMA Model prediction contains NaN. Check model input or preprocessing steps.")
                              st.stop()

                          st.write("Predictions:", y_pred_original_scale[:12])  # Menampilkan 12 bulan prediksi
                          plot_predictions(y_true, y_pred_original_scale)
                        
                         # Insert into database
                          try:
                              # Fetch the corresponding model_id
                              cursor.execute("SELECT model_id FROM models WHERE algorithm = 'ARIMA' ORDER BY model_id DESC LIMIT 1")
                              result = cursor.fetchone()
                              model_id = result[0] if result else None
                              
                              if model_id is not None:
                                  # Fetch the corresponding dataset_id
                                  cursor.execute("SELECT dataset_id FROM datasets WHERE file_path = ? ORDER BY dataset_id DESC LIMIT 1", (os.path.join(DATASETS_DIR,file_name),))
                                  result = cursor.fetchone()
                                  dataset_id = result[0] if result else None
                                  
                                  if dataset_id is not None:
                                      prediction_date = datetime.today().strftime('%Y-%m-%d')
                                      results_text = str(y_pred_original_scale[:12])  # convert list to string
                                      cursor.execute("INSERT INTO predictions (model_id, dataset_id, prediction_date, results) VALUES (?, ?, ?, ?)", (model_id, dataset_id, prediction_date, results_text))
                                      conn.commit()
                                      st.success(f"Prediction added to database.")
                              else:
                                  st.error("model_id not found.")
                          except sqlite3.Error as e:
                              st.error(f"Failed to add prediction to database: {e}")

                        else:
                          st.error("Data preprocessing failed.")

          elif algo == "Transformer":
                if 'fold' not in st.session_state:
                   st.session_state.fold = 1
                fold = st.selectbox("Choose Model Fold", [1, 2, 3, 4, 5], key = "fold")
                if st.button("Predict", key=f"predict_transformer_{fold}"):
                  y_pred = transformer_prediction(df, fold)
                  if y_pred is not None:
                    y_true = df["jumlah_kasus"].tolist()
                    st.success("Transformer prediction completed.")
                    st.write("Predictions:", y_pred[:12])  # Menampilkan 12 bulan prediksi
                    plot_predictions(y_true[len(y_true)-len(y_pred):], y_pred)
                     # Insert into database
                    try:
                        # Fetch the corresponding model_id
                        cursor.execute("SELECT model_id FROM models WHERE algorithm = 'Transformer' ORDER BY model_id DESC LIMIT 1")
                        result = cursor.fetchone()
                        model_id = result[0] if result else None
                        
                        if model_id is not None:
                            # Fetch the corresponding dataset_id
                            cursor.execute("SELECT dataset_id FROM datasets WHERE file_path = ? ORDER BY dataset_id DESC LIMIT 1", (os.path.join(DATASETS_DIR,file_name),))
                            result = cursor.fetchone()
                            dataset_id = result[0] if result else None
                            
                            if dataset_id is not None:
                                prediction_date = datetime.today().strftime('%Y-%m-%d')
                                results_text = str(y_pred[:12])  # convert list to string
                                cursor.execute("INSERT INTO predictions (model_id, dataset_id, prediction_date, results) VALUES (?, ?, ?, ?)", (model_id, dataset_id, prediction_date, results_text))
                                conn.commit()
                                st.success(f"Prediction added to database.")
                            else:
                                st.error("dataset_id not found.")
                        else:
                            st.error("model_id not found.")
                    except sqlite3.Error as e:
                         st.error(f"Failed to add prediction to database: {e}")

    elif page == "Evaluation":
        st.header("Evaluation")
        algo = st.selectbox("Choose Algorithm for Evaluation", ["ARIMA", "Transformer"])
        file_name = st.selectbox("Select a dataset for evaluation", os.listdir(DATASETS_DIR))
        df = load_dataset(file_name) if file_name else None

        if df is not None:
             # Validasi NaN di y_true
            y_true = df["jumlah_kasus"].tolist()
            if any(pd.isna(y_true)):
                st.error("Dataset contains NaN in target column 'jumlah_kasus'. Please clean the data.")                
                st.stop()

        if df is not None:
            if algo == "ARIMA":
                if st.button("Evaluate", key="evaluate_arima"):                    
                    metrics, y_pred_original = evaluate_arima_model(df)
                    if metrics is not None and y_pred_original is not None:
                        y_actual = df["jumlah_kasus"].values[-12:]
                        st.write("**Evaluasi Model ARIMA:**")
                        st.json(metrics)
                        st.line_chart({"Actual": y_actual, "Predicted": y_pred_original})
                        # Insert evaluation data
                        try:
                            # Fetch the corresponding model_id
                            cursor.execute("SELECT model_id FROM models WHERE algorithm = 'ARIMA' ORDER BY model_id DESC LIMIT 1")
                            result = cursor.fetchone()
                            model_id = result[0] if result else None
                            
                            if model_id is not None:
                                # Fetch the corresponding dataset_id
                                cursor.execute("SELECT dataset_id FROM datasets WHERE file_path = ? ORDER BY dataset_id DESC LIMIT 1", (os.path.join(DATASETS_DIR,file_name),))
                                result = cursor.fetchone()
                                dataset_id = result[0] if result else None
                                
                                if dataset_id is not None:
                                    metrics_text = str(metrics)
                                    cursor.execute("INSERT INTO evaluations (model_id, dataset_id, metrics) VALUES (?, ?, ?)", (model_id, dataset_id, metrics_text))
                                    conn.commit()
                                    st.success(f"Evaluation added to database.")
                            else:
                                st.error("model_id not found.")
                        except sqlite3.Error as e:
                            st.error(f"Failed to add evaluation to database: {e}")

            elif algo == "Transformer":
                 if 'fold' not in st.session_state:
                    st.session_state.fold = 1
                 fold = st.selectbox("Choose Model Fold", [1, 2, 3, 4, 5], key= "fold")
                 if st.button("Evaluate", key=f"evaluate_transformer_{fold}"):
                    metrics, y_pred = evaluate_transformer(df, fold)
                    if metrics is not None and y_pred is not None:
                        y_true = df["jumlah_kasus"].tolist()
                        st.write(metrics)
                        plot_predictions(y_true[len(y_true)-len(y_pred):], y_pred, title="Transformer Actual vs Predicted")
                         # Insert evaluation data
                        try:
                            # Fetch the corresponding model_id
                            cursor.execute("SELECT model_id FROM models WHERE algorithm = 'Transformer' ORDER BY model_id DESC LIMIT 1")
                            result = cursor.fetchone()
                            model_id = result[0] if result else None
                            
                            if model_id is not None:
                                # Fetch the corresponding dataset_id
                                cursor.execute("SELECT dataset_id FROM datasets WHERE file_path = ? ORDER BY dataset_id DESC LIMIT 1", (os.path.join(DATASETS_DIR,file_name),))
                                result = cursor.fetchone()
                                dataset_id = result[0] if result else None
                                
                                if dataset_id is not None:
                                    metrics_text = str(metrics)
                                    cursor.execute("INSERT INTO evaluations (model_id, dataset_id, metrics) VALUES (?, ?, ?)", (model_id, dataset_id, metrics_text))
                                    conn.commit()
                                    st.success(f"Evaluation added to database.")
                            else:
                                st.error("model_id not found.")
                        except sqlite3.Error as e:
                            st.error(f"Failed to add evaluation to database: {e}")
    
    elif page == "Reports":
        clear_model_state()
        st.header("Report Management")
        report_title = st.text_input("Report Title")
        report_content = st.text_area("Report Content")

        if st.button("Create Report"):
            if report_title and report_content:
                try:
                    # Assume a hardcoded admin ID (you may need a better solution for admin selection)
                    user_id = st.session_state.user_id
                    cursor.execute("INSERT INTO reports (title, content, created_by) VALUES (?, ?, ?)", (report_title, report_content, user_id))
                    conn.commit()
                    st.success("Report created successfully.")
                except sqlite3.Error as e:
                    st.error(f"Failed to add report to database: {e}")
            else:
                 st.error("Please fill all required fields.")
        
        # Report Management
        st.subheader("Manage Reports")
        all_reports = get_all_reports(cursor)
        if all_reports:
            st.write("All Reports:")
            for report_id, title, content, created_at in all_reports:
               st.write(f"- ID: {report_id}, Title: {title}, Created At: {created_at}")
            
            report_id_to_update = st.number_input("Select report ID to Update or Delete", min_value=1, step=1, value=1, key = "report_id_input")
            report_details = get_report_by_id(cursor, report_id_to_update)
            if report_details:
                st.write(f"Details of Report {report_id_to_update}:")
                st.write(f"report_id: {report_details[0]}, title: {report_details[1]}, content: {report_details[2]}")

                update_title = st.text_input("New Title", value = report_details[1])
                update_content = st.text_area("New Content", value = report_details[2])
                
                if st.button("Update Report", key = "update_report_button"):
                    user_id = st.session_state.user_id
                    if update_report(cursor, report_id_to_update, update_title, update_content, user_id):
                         st.success("Report updated successfully")
                    else:
                        st.error("Failed to update the Report")
                
                if st.button("Delete Report", key = "delete_report_button"):
                   user_id = st.session_state.user_id
                   if delete_report(cursor, report_id_to_update, user_id):
                      st.success("Report deleted Successfully")
                   else:
                       st.error("Failed to delete report")
            else:
                 st.error("Report Not Found")
    # Close connection after main func
    if conn:
        conn.close()

if __name__ == "__main__":
    main()