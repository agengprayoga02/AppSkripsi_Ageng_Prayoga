import streamlit as st
from streamlit_option_menu import option_menu
import utils
import db
import os
import time
import auth  # Import file otentikasi
# import datasets  # Hapus import ini
import models
import prediction
import compare
import history
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy
import sqlite3

DATASETS_DIR = "datasets"  # Sesuaikan path

# Initialize database connection in app.py (only once)
db_conn_result = db.init_db()
if db_conn_result is None:
    st.error("Failed to connect to the database.")
    st.stop()  # Terminate the app if database connection fails
conn, cursor = db_conn_result

def clear_model_state():
    for key in st.session_state.keys():
        if key.startswith("transformer_model_fold_"):
            del st.session_state[key]
    if 'y_pred' in st.session_state:
        del st.session_state['y_pred']

# Function for EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")

    # 1. Memahami Struktur Dataset
    st.markdown("### 1. Struktur Dataset")
    st.write("Beberapa baris pertama dataset:")
    st.write(df.head())
    st.write(f"Dimensi dataset: {df.shape}")
    st.write("Informasi dataset:")
    # buffer = df.info(memory_usage=False)  #df.info() returns None, so buffer is undefined
    # st.text(buffer)
    df.info() # Print to console so streamlit will capture it
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

        # st.write("Pairplot untuk melihat hubungan antar beberapa variabel numerik:")
        # sns.pairplot(df[numerical_cols]) # Pairplots are very slow in streamlit, avoid
        # st.pyplot(plt)
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

def load_dataset(file_name):
    file_path = os.path.join(DATASETS_DIR, file_name)
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_name}")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def datasets_page(conn, cursor, username): # pindahkan ke app.py
    st.header("Datasets")
    st.subheader("Hello, {}".format(username))

    st.subheader("Available Datasets")

    # Load existing datasets from the database
    all_datasets = get_all_datasets(cursor)
    existing_datasets = {dataset[1]: dataset[0] for dataset in all_datasets}  # {dataset_name: dataset_id}

    # Display available datasets with IDs and load button
    files = os.listdir(DATASETS_DIR)
    if files:
        for file_name in files:
            col1, col2 = st.columns([0.8, 0.2])  # Adjust column widths as needed
            with col1:
                if file_name in existing_datasets:
                    dataset_id = existing_datasets[file_name]
                    st.write(f"- ID: {dataset_id}, Name: {file_name}")
                else:
                    st.write(f"- Name: {file_name} (Not yet registered)")
            with col2:
                if st.button(f"Load", key=f"load_{file_name}"):
                    df = load_dataset(file_name)
                    if df is not None:
                        st.write(df.head())
                        perform_eda(df)

                        # Check if dataset already exists
                        existing_dataset = get_dataset_by_name(cursor, file_name)
                        user_id = st.session_state.user_id

                        if not existing_dataset:
                            # Insert into database
                            try:
                                cursor.execute(
                                    "INSERT INTO datasets (dataset_name, file_path, uploaded_by) VALUES (?, ?, ?)",
                                    (file_name, os.path.join(DATASETS_DIR, file_name), user_id),
                                )
                                conn.commit()
                                st.success(f"Dataset '{file_name}' metadata added to database.")
                            except sqlite3.Error as e:
                                st.error(f"Failed to add dataset metadata to database: {e}")
                        else:
                            st.warning(f"Dataset '{file_name}' already exists with ID: {existing_dataset[0]}.")

                        # Refresh all_datasets after upload
                        all_datasets = get_all_datasets(cursor)
                        existing_datasets = {dataset[1]: dataset[0] for dataset in all_datasets}  # Update cache
                        st.rerun()  # Force Streamlit to reload the page
    else:
        st.write("No datasets found in the directory.")

    # Upload Dataset
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        file_name = uploaded_file.name
        file_path = os.path.join(DATASETS_DIR, file_name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{file_name}' saved to '{DATASETS_DIR}'.")

            # Load the uploaded file and check for NaNs in 'jumlah_kasus'
            df_uploaded = load_dataset(file_name)
            if df_uploaded is not None:
                if "jumlah_kasus" not in df_uploaded.columns:
                    st.error("The column 'jumlah_kasus' does not exist. Please check your dataset.")
                elif df_uploaded["jumlah_kasus"].isna().any():
                    st.error("The column 'jumlah_kasus' contains NaN. Please fix it first.")
                else:
                    # Check if dataset already exists
                    existing_dataset = get_dataset_by_name(cursor, file_name)
                    user_id = st.session_state.user_id

                    if not existing_dataset:
                        # Insert into database
                        try:
                            cursor.execute(
                                "INSERT INTO datasets (dataset_name, file_path, uploaded_by) VALUES (?, ?, ?)",
                                (file_name, file_path, user_id),
                            )
                            conn.commit()
                            st.success(f"Dataset '{file_name}' metadata added to database.")
                        except sqlite3.Error as e:
                            st.error(f"Failed to add dataset metadata to database: {e}")
                    else:
                        st.warning(f"Dataset '{file_name}' already exists with ID: {existing_dataset[0]}.")

                    # Refresh all_datasets after upload
                    all_datasets = get_all_datasets(cursor)
                    existing_datasets = {dataset[1]: dataset[0] for dataset in all_datasets}  # Update cache
                    st.rerun()  # Force Streamlit to reload the page

        except Exception as e:
            st.error(f"Error uploading file: {e}")

    # Dataset Management
    st.subheader("Manage Datasets")
    if all_datasets:
        st.write("All Datasets:")
        for dataset_id, dataset_name, file_path, uploaded_at in all_datasets:
            st.write(f"- ID: {dataset_id}, Name: {dataset_name}, Path: {file_path}, Uploaded At: {uploaded_at}")

        dataset_id_to_update = st.number_input("Select dataset ID to Update", min_value=1, step=1, value=1, key="update_dataset", format="%d")
        dataset_details = get_dataset_by_id(cursor, dataset_id_to_update)

        if dataset_details:
            st.write(f"Details of dataset {dataset_id_to_update}")
            st.write(f"dataset_id: {dataset_details[0]}, dataset_name: {dataset_details[1]}, file_path: {dataset_details[2]}")

            update_dataset_name = st.text_input("New Dataset Name", value=dataset_details[1])
            update_file_path = st.text_input("New file_path", value=dataset_details[2])
            update_description = st.text_input("New Description", value=dataset_details[3] if dataset_details[3] else "")

            if st.button("Update Dataset", key="update_dataset_button"):
                user_id = st.session_state.user_id
                if update_dataset(
                    cursor, dataset_id_to_update, update_dataset_name, update_file_path, update_description, user_id
                ):
                    conn.commit()
                    st.success("Dataset updated successfully")
                else:
                    st.error("Failed to update the Dataset")

            dataset_id_to_delete = st.number_input(
                "Select Dataset ID to delete", min_value=1, step=1, value=1, key="delete_dataset_id", format="%d"
            )
            if st.button("Delete Dataset", key="delete_dataset_button"):
                user_id = st.session_state.user_id
                if delete_dataset(cursor, dataset_id_to_delete, user_id):
                    conn.commit()
                    st.success("Dataset deleted successfully")
                else:
                    st.error("Failed to delete dataset")
        else:
            st.error("Dataset Not Found")
    else:
        st.write("No datasets found in the database.")

def main():
    st.set_page_config(
        page_title="DBD Prediction Application",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "last_login_attempt" not in st.session_state:
        st.session_state.last_login_attempt = 0
    if "account_locked" not in st.session_state:
        st.session_state.account_locked = False
    if 'show_forgot_password' not in st.session_state:
        st.session_state.show_forgot_password = False
    if 'show_submit_button' not in st.session_state:
        st.session_state.show_submit_button = False
    if 'reset_password' not in st.session_state:
        st.session_state.reset_password = False
    if 'reset_password_start_time' not in st.session_state:
        st.session_state.reset_password_start_time = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Login'
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None

    # Authentication section
    if not st.session_state.logged_in:
        username, password, auth_status, user_id = login_section()
        if auth_status:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.current_page = 'Datasets'  # Redirect to Datasets page after login
            st.session_state.user_id = user_id
            st.rerun()  # Rerun the app to reflect the new state
        return  # Exit main function if not logged in

    # Main content section
    if st.session_state.logged_in:
        # Create a sidebar menu
        with st.sidebar:
            menu_options = ["Datasets", "Models", "Prediksi & Evaluasi", "Compare Models", "History"]
            selected = option_menu(
                menu_title="Menu",
                options=menu_options,
                icons=['database', 'database', 'gear', 'activity', 'arrow-left-right', 'clock-history'],
            )
            if st.sidebar.button(f"Profile: {st.session_state.username}"):
              st.sidebar.write(f"User ID: {st.session_state.user_id}")
              if st.sidebar.button("Logout"):
                  st.session_state.logged_in = False
                  st.session_state.user_id = None
                  st.session_state.username = None
                  st.rerun()

         # Route to the selected page
        if selected == "Datasets":
            datasets_page(conn, cursor, st.session_state.username)  # Call datasets_page from app.py
        elif selected == "Models":
            models.models_page(conn, cursor, st.session_state.username)
        elif selected == "Prediksi & Evaluasi":
            prediction.prediction_page(conn, cursor)
        elif selected == "Compare Models":
            compare.compare_models_page(conn, cursor)
        elif selected == "History":
            history.history_page(conn, cursor)

def login_section():
    """
    Display login and registration options, and handle authentication logic.
    """
    st.title("DBD Prediction Application")
    
    auth_page = st.radio("Select", ["Register", "Login"])
        
    if auth_page == "Register":
        st.header("Register")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password", help="Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.")
        new_password_confirm = st.text_input("Confirm Password", type="password")
        new_email = st.text_input("Email")

        if st.button("Register"):
            if not new_username:
                st.error("Username is required.")
            elif not new_password:
                st.error("Password is required.")
            elif not new_email:
                                st.error("Email is required.")
            elif not db.is_valid_email(new_email):
                st.error("Invalid email format")
            elif new_password != new_password_confirm:
                st.error("Passwords do not match")
            elif not utils.is_valid_password(new_password):
              st.error("Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.")
            else:
                try:
                    hashed_password, salt = db.hash_password(new_password)
                    cursor.execute("INSERT INTO users (username, password_hash, email, salt) VALUES (?, ?, ?, ?)", (new_username, hashed_password, new_email, salt))
                    conn.commit()
                    st.success("Registration successful. Please log in.")
                    return None, None, False, None
                except sqlite3.IntegrityError:            
                    st.error("Username or email already exists.")
        return None, None, False, None
    elif auth_page == "Login":
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Forgot Password Button
        if st.button("Forgot Password?"):
            st.session_state.show_forgot_password = True
            st.rerun()
        if st.session_state.get("show_forgot_password"):
            st.header("Forgot Password")
            forgot_username = st.text_input("Enter your Username")
            if st.button("Search Username"):
                if forgot_username:
                        cursor.execute("SELECT user_id, username, password_hash, salt FROM users WHERE username = ?", (forgot_username,))
                        user = cursor.fetchone()
                        if user:
                            st.write(f"Your password hash: {user[2]}")
                            st.session_state.forgot_user_id = user[0]  # Store the user_id
                            st.session_state.forgot_username = user[1]  # Store the username
                            st.session_state.show_submit_button = True  # Show Submit Username Button
                            st.session_state.reset_password = True # show reset password button
                            st.session_state.reset_password_start_time = time.time()
                        else:
                            st.error("Username not found.")
                else:
                    st.error("Please enter your username.")
        
        if st.session_state.get("show_submit_button"):
            if st.session_state.get("reset_password"):
                 st.header("Reset Password")
                 new_password_reset = st.text_input("New Password", type="password", help="Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.")
                 new_password_confirm_reset = st.text_input("Confirm New Password", type = "password")
                 now = time.time()
            if st.session_state.reset_password_start_time is not None and now - st.session_state.reset_password_start_time > 30:
                    st.error("Reset password time limit exceeded. Please try again.")
                    st.session_state.reset_password = False
                    st.session_state.show_submit_button = False
                    st.session_state.show_forgot_password = False
                    st.session_state.reset_password_start_time = None
            elif st.button("Reset"):
                    if new_password_reset and new_password_confirm_reset:
                      if new_password_reset != new_password_confirm_reset:
                          st.error("New Password not same")
                      elif not utils.is_valid_password(new_password_reset):
                        st.error("Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.")
                      else:
                          try:
                              hashed_password, salt = db.hash_password(new_password_reset)
                              user_id = st.session_state.forgot_user_id
                              cursor.execute("UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?", (hashed_password, salt, user_id))
                              conn.commit()
                              st.success("Password reset successful.")
                              st.session_state.reset_password = False
                              st.session_state.show_submit_button = False
                              st.session_state.show_forgot_password = False
                              st.rerun() # Re-run app to go back to Login page
                          except sqlite3.Error as e:
                              st.error(f"Failed to reset password: {e}")
                    else:
                        st.error("Please fill the New Password.")
            else:
                if st.button("Submit Username"):
                    st.session_state.logged_in = True
                    st.session_state.user_id = st.session_state.forgot_user_id
                    st.session_state.username = st.session_state.forgot_username
                    st.success("Login successful")
                    st.session_state.show_submit_button = False
                    st.session_state.show_forgot_password = False
                    st.rerun()
        else:
            if st.button("Login"):
                if st.session_state.account_locked:
                    st.error("Account locked due to multiple failed login attempts. Please try again after some time.")
                    return None, None, False, None # Stop logic if account is locked
                elif username and password:
                    now = time.time()
                    # Rate limiting implementation
                    if now - st.session_state.last_login_attempt <= 60:  # only process if last login attempt is less than 1 minute
                        st.session_state.login_attempts += 1
                    else:  # reset login attempt after 1 minute
                        st.session_state.login_attempts = 1
                    st.session_state.last_login_attempt = now
                    if st.session_state.login_attempts > 3:
                        st.session_state.account_locked = True
                        st.error("Too many failed login attempts. Account locked.")
                        return None, None, False, None
                    
                    cursor.execute("SELECT user_id, username, password_hash, salt FROM users WHERE username = ?", (username,))
                    user = cursor.fetchone()
                    if user and db.check_password(password, user[2], user[3]):
                        st.session_state.logged_in = True
                        st.session_state.user_id = user[0]
                        st.session_state.username = user[1]
                        st.session_state.login_attempts = 0
                        st.session_state.last_login_attempt = 0
                        st.session_state.account_locked = False
                        st.success("Login successful")
                        return username, password, True, user[0]
                    else:
                        st.error("Username or password not valid")
                else:
                    st.error("Please enter username and password")
        return None, None, False, None

if __name__ == "__main__":
    main()
