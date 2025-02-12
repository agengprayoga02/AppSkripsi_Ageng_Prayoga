import re
import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import KFold
import streamlit as st
import logging  # Import logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Pastikan ini sesuai dengan struktur direktori Anda)
DATASETS_DIR = ("datasets")
MODELS_DIR = ("saved_models")


# --- Load Data ---
def load_and_preprocess_data(file_path):
    """Loads data, converts 'periode' to datetime, sets as index."""
    df = pd.read_csv(file_path)
    df['periode'] = pd.to_datetime(df['periode'], format="%Y-%m")
    df.set_index('periode', inplace=True)
    return df

# --- Data Normalization ---
def scale_data(df, feature_column='jumlah_kasus'):
    """Scales data using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature_column]])
    return scaled_data, scaler

# --- Stationarity Check (For ARIMA) ---
def check_stationarity(timeseries):
    """Checks stationarity using ADFuller test, applies differencing if needed."""
    result = adfuller(timeseries)
    if result[1] <= 0.05:
        logging.info("Data sudah stasioner.")
        return timeseries
    else:
        logging.info("Data tidak stasioner, melakukan differencing.")
        return pd.DataFrame(timeseries).diff().dropna().values  # Return differenced data

# --- Split Data (ARIMA) ---
def split_data_arima(scaled_data, train_ratio=0.8):
    """Splits data into training and testing sets for ARIMA."""
    train_size = int(len(scaled_data) * train_ratio)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    return train_data, test_data

# --- Create Sequences (For Informer) ---
def create_sequences(data, seq_length):
    """Creates sequences for LSTM-based models."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# --- Load Dataset ---
def load_dataset(file_name):
    """Loads dataset from the specified file."""
    file_path = os.path.join(DATASETS_DIR, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        st.error(f"File not found: {file_path}")
        return None

def preprocess_dataset_informer(df, target_col="jumlah_kasus", seq_length=12):
    """
    Preprocess dataset for Informer model. Converts to numeric, handles NaNs.
    """
    x_data = df.copy()
    # Convert all columns to numeric
    x_data = x_data.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    if x_data.isna().any().any():
        # Drop columns with all NaN values
        cols_to_drop = [col for col in x_data.columns if x_data[col].isna().all()]
        if cols_to_drop:
            x_data = x_data.drop(cols_to_drop, axis=1)
            if 'periode' in cols_to_drop:
                print("Kolom 'periode' dihapus karena hanya mengandung nilai NaN.")

        # Fill remaining NaNs with column means
        x_data.fillna(x_data.mean(), inplace=True)

    if x_data.empty:
        st.error("Data is empty after preprocessing.")
        return None, None

    y_data_true = df[target_col].values.astype(np.float32)
    return x_data, y_data_true

# --- Load Models ---
def load_arima_model():
    """Loads the ARIMA model."""
    model_path = os.path.join(MODELS_DIR, "sarimax_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        return model_dict
    else:
        st.error(f"ARIMA model not found at {model_path}")
        return None

def load_transformer_model_final():
    """Loads the final Transformer model."""
    model_path = os.path.join(MODELS_DIR, "final_model.h5")
    try: # Tambahkan try-except
        logging.info(f"Mencoba memuat model dari: {model_path}")  # Tambahkan ini
        if os.path.exists(model_path):
            model = load_model(model_path)
            logging.info(f"Final Transformer model loaded from: {model_path}")  # Tambahkan ini
            return model
        else:
            st.error(f"Final Transformer model not found at {model_path}")
            logging.warning(f"Final Transformer model not found at: {model_path}") # Tambahkan ini
            return None
    except Exception as e:
        st.error(f"Error loading Transformer model: {e}")
        logging.exception(f"Error loading Transformer model: {e}")
        return None

def load_informer_scaler(): # Fungsi Baru untuk memuat scaler
    """Loads the scaler for Informer model."""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")  # Sesuaikan jika nama file scaler berbeda
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logging.info(f"Informer scaler loaded from: {scaler_path}")  # Tambahkan ini
        return scaler
    except Exception as e:
        st.error(f"Error loading Informer scaler: {e}")
        logging.exception(f"Error loading Informer scaler: {e}") # Tambahkan ini
        return None

# --- Evaluate Model ---
def evaluate_predictions(y_true, y_pred):
    """Evaluates predictions using common metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Handle division by zero for MAPE
    if np.any(y_true == 0):
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# --- Plot Visualization ---
def plot_predictions(y_true, y_pred, title="Actual vs Predicted", xlabel="Periode", ylabel="Jumlah Kasus", algo=None):
    """Plots actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", marker="o", color="blue")
    plt.plot(y_pred, label="Predicted", marker="x", color="orange")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    st.pyplot(plt)

# --- Generate Recommendation ---
def generate_recommendation(metrics, y_pred, y_actual):
    """Generates recommendations based on metrics and predictions."""
    recommendation = ""
    if not metrics:
        return "No evaluation metrics available to create recommendations."

    rmse = metrics.get("RMSE")
    r2 = metrics.get("R2")
    mape = metrics.get("MAPE")

    if rmse is not None and r2 is not None and mape is not None:
        if rmse < 100 and r2 > 0.7 and mape < 20:
            recommendation += "Model yang digunakan bekerja dengan baik. Tingkat akurasi yang didapatkan cukup tinggi."
        elif rmse < 150 and r2 > 0.6 and mape < 30:
            recommendation += "Model memiliki tingkat akurasi yang cukup baik. Perlu adanya perbaikan untuk model."
        else:
            recommendation += "Model yang digunakan kurang baik. Perlu adanya pengecekan lebih lanjut pada model."

    # Membandingkan hasil prediksi dan data aktual untuk tren
    if y_actual is not None and y_pred is not None:
        if len(y_pred) > 1:
            last_actual = y_actual[-1]
            last_pred = y_pred[-1]

            if last_pred > last_actual * 1.2:  # jika prediksi meningkat lebih dari 20%
                recommendation += " Model memprediksi peningkatan kasus DBD yang signifikan. Diperlukan tindakan pencegahan dan mitigasi."
            elif last_pred < last_actual * 0.8:  # jika prediksi menurun lebih dari 20%
                recommendation += " Model memprediksi penurunan kasus DBD, namun tetap diperlukan kewaspadaan."
            else:
                recommendation += " Model memprediksi perubahan yang stabil untuk kasus DBD"

    return recommendation

# --- Compare Actual vs Predicted ---
def compare_actual_vs_predicted(y_true, y_pred):
    """Compares actual vs predicted values and returns metrics."""
    if not all(isinstance(data, np.ndarray) for data in [y_true, y_pred]):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

    metrics = evaluate_predictions(y_true, y_pred)
    plot_predictions(y_true, y_pred, title="Actual vs Predicted")

    return metrics

# --- Transformer Prediction Final ---
def transformer_prediction_final(df, forecast_months):
    """
    Melakukan prediksi menggunakan model Transformer yang dilatih pada seluruh data (final_model.h5).
    """
    final_model = load_transformer_model_final()  # Load the final model
    if not final_model:
        st.error("Final Transformer model not found!")
        return []

    # Load scaler
    scaler_informer = load_informer_scaler() # Panggil fungsi untuk memuat scaler
    if not scaler_informer:
        return []

    # Preprocessing Data
    x_data, _ = preprocess_dataset_informer(df)
    if x_data is None:
        return []

    # Scaling data harus dilakukan sebelum membuat sequence
    # scaler = MinMaxScaler()  # HAPUS BARIS INI
    scaled_data = scaler_informer.transform(x_data)  # Fit scaler dengan x_data # GANTI DENGAN INI
    X, _ = create_sequences(scaled_data, seq_length=12)  # Assuming seq_length is 12

    if X is None:
        st.error("Error during data preprocessing for Transformer model")
        return []

    # Make Prediction
    try:
        #num_sequence = len(df) - 12
        #predictions_scaled = []
        #for i in range(num_sequence,len(df)):
        #  input_sequence = scaled_data[i-12:i]
        #  input_sequence = np.expand_dims(input_sequence, axis=0)
        #  predictions_scaled.append(final_model.predict(input_sequence)[0])

        # PREDIKSI LANGSUNG (JIKA MODEL DILATIH UNTUK INI)
        predictions_scaled = final_model.predict(X) #Ganti semua baris di atas dengan baris ini
        predictions_scaled = predictions_scaled[-forecast_months:] #Ambil beberapa bulan terakhir

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        logging.exception(f"Error during prediction: {e}")
        return []

    # Invert Scaling
    try:
        predictions = scaler_informer.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()  # flatten the predictions
        #predictions = predictions[-forecast_months:] #HAPUS BARIS INI
    except Exception as e:
        st.error(f"Error inverting scaling: {e}")
        logging.exception(f"Error inverting scaling: {e}")
        return []

    return predictions

# --- No Load Transformer Model By Fold---
def load_transformer_model(fold):
    return None

# --- No Split Data for Cross Validation (Informer) ---
def split_data_informer(X, y, n_splits=5, shuffle=True, random_state=42):
    return None

def evaluate_transformer(df, fold, forecast_months):
   return None, None

def evaluate_arima_model(df, forecast_months):
    logging.info("Starting evaluate_arima_model")
    # Load Model
    model_dict = load_arima_model()
    if not model_dict:
        logging.error("ARIMA model not loaded.")
        return None, None
    model = model_dict["model"]
    scaler = model_dict["scaler"]

    y_actual = df["jumlah_kasus"].values[-forecast_months:].astype(np.float32)
    logging.info(f"y_actual shape: {y_actual.shape}, first 5 values: {y_actual[:5]}")

    # Preprocessing data for ARIMA
    scaled_data, _ = scale_data(df) # Pastikan scaler yang digunakan sudah benar disini.
    logging.info(f"Shape of scaled_data before stationarity check: {scaled_data.shape}")
    scaled_data = check_stationarity(scaled_data)

    if scaled_data is not None:
        logging.info(f"Shape of scaled_data after stationarity check: {scaled_data.shape}")
        # split the train data
        train_data, test_data = split_data_arima(scaled_data)
        logging.info(f"train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
        try:
            y_pred = model.forecast(len(y_actual))
            logging.info(f"Shape of y_pred before inverse transform: {y_pred.shape}")
            y_pred_original = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
            logging.info(f"Shape of y_pred_original: {y_pred_original.shape}, first 5 values: {y_pred_original[:5]}")

            # Evaluasi
            metrics = evaluate_predictions(y_actual, y_pred_original)
            logging.info(f"Metrics: {metrics}")

            return metrics, y_pred_original
        except Exception as e:
            st.error(f"Error during ARIMA forecasting: {e}")
            logging.exception(f"Error during ARIMA forecasting: {e}")
            return None, None
    else:
        logging.warning("Data preprocessing failed.")
        return None, None

def is_valid_password(password):
    """
    Memeriksa apakah password memenuhi kriteria berikut:
    - Minimal 8 karakter
    - Setidaknya 1 huruf besar
    - Setidaknya 1 huruf kecil
    - Setidaknya 1 angka
    - Setidaknya 1 karakter khusus
    """
    if len(password) < 8:
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[a-z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    if not re.search("[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True
