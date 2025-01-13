import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

def preprocess_data(data_path):
    """
    Melakukan preprocessing pada dataset time series.

    Args:
        data_path (str): Path menuju file CSV dataset.

    Returns:
        tuple: Tuple berisi data yang sudah di-preprocess, scaler, dan order differencing.
               (train_data, test_data, train_exog, test_exog, scaler_y, scaler_x, target_diff_order, exog_diff_orders)
    """

    # Load Dataset
    df = pd.read_csv(data_path)
    df['periode'] = pd.to_datetime(df['periode'])
    df.set_index('periode', inplace=True)

    # Time Series Data
    time_series = df['jumlah_kasus']

    # Exogenous Variables
    exog_data = df[['Tavg', 'RH_avg', 'RR']]

    # Scaling
    scaler_y = MinMaxScaler()
    scaler_x = MinMaxScaler()
    scaled_time_series = scaler_y.fit_transform(time_series.values.reshape(-1, 1)).flatten()
    scaled_exog_data = scaler_x.fit_transform(exog_data)

     # Fungsi untuk Cek Stasionaritas dan Differencing
    def check_stationarity(series, name, max_diff=2):
        d = 0
        temp_series = series.copy()
        while d <= max_diff:
            result = adfuller(temp_series)
            if result[1] <= 0.05:
                print(f"{name} sudah stasioner setelah differencing {d} kali.")
                return temp_series, d
            else:
                print(f"{name} tidak stasioner, differencing ke-{d+1}.")
                temp_series = np.diff(temp_series)
                d += 1
        raise ValueError(f"{name} tidak stasioner setelah {max_diff} kali differencing. Mungkin butuh transformasi lain.")

    # Cek Stasionaritas Target
    scaled_time_series, target_diff_order = check_stationarity(scaled_time_series, "Target (Jumlah Kasus)")

    # Cek Stasionaritas Exogenous dan lakukan differencing jika perlu
    exogenous_scaled_processed = {}  # Ubah menjadi dictionary
    exog_diff_orders = {}     # Ubah menjadi dictionary
    for i, col in enumerate(exog_data.columns):
        exog_series, diff_order = check_stationarity(scaled_exog_data[:, i], f"Exogenous ({col})")
        exogenous_scaled_processed[col] = exog_series #simpan per kolom
        exog_diff_orders[col] = diff_order #simpan per kolom

    # Pad series eksogen dengan NaN agar memiliki panjang yang sama
    max_len = max(len(series) for series in exogenous_scaled_processed.values())
    for col, series in exogenous_scaled_processed.items():
        if len(series) < max_len:
            padding_len = max_len - len(series)
            exogenous_scaled_processed[col] = np.concatenate([series, np.full(padding_len, np.nan)])

    exogenous_scaled_processed = pd.DataFrame(exogenous_scaled_processed) #ubah jadi dataframe
    # Train-Test Split
    train_size = int(len(scaled_time_series) * 0.8)
    train_data = scaled_time_series[:train_size]
    test_data = scaled_time_series[train_size:]
    train_exog = exogenous_scaled_processed.iloc[:train_size]
    test_exog = exogenous_scaled_processed.iloc[train_size:]
    return train_data, test_data, train_exog, test_exog, scaler_y, scaler_x, target_diff_order, exog_diff_orders


if __name__ == '__main__':
    data_path = r"C:\Users\ASUS\Downloads\ARIMA\data_kasus_dbd_dki_jakarta_2015_2020.csv"
    train_data, test_data, train_exog, test_exog, scaler_y, scaler_x, target_diff_order, exog_diff_orders = preprocess_data(data_path)

    print("Shape of train_data:", train_data.shape)
    print("Shape of test_data:", test_data.shape)
    print("Shape of train_exog:", train_exog.shape)
    print("Shape of test_exog:", test_exog.shape)
    print("Target diff order:", target_diff_order)
    print("Exog diff orders:", exog_diff_orders)