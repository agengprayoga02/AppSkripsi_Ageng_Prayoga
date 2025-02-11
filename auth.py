import streamlit as st
import db
import utils
import time
import sqlite3
import re

def login_section(conn, cursor):
    """
    Display login and registration options, and handle authentication logic.
    Returns:
        Tuple: (username, user_id, authentication_status)
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
                    return None, None, False
                except sqlite3.IntegrityError:
                    st.error("Username or email already exists.")
        return None, None, False

    elif auth_page == "Login":
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        # Forgot Password Button
        if st.button("Forgot Password?"):
            st.session_state.show_forgot_password = True
            st.experimental_rerun()

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
                        st.session_state.reset_password = True  # show reset password button
                        st.session_state.reset_password_start_time = time.time()
                    else:
                        st.error("Username not found.")
                else:
                    st.error("Please enter your username.")

        if st.session_state.get("show_submit_button"):
            if st.session_state.get("reset_password"):
                st.header("Reset Password")
                new_password_reset = st.text_input("New Password", type="password", help="Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.")
                new_password_confirm_reset = st.text_input("Confirm New Password", type="password")
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
                            st.session_state.reset_password_start_time = None
                            st.experimental_rerun()  # Re-run app to go back to Login page
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
                    st.experimental_rerun()
        else:
            if st.button("Login"):
                if st.session_state.account_locked:
                    st.error("Account locked due to multiple failed login attempts. Please try again after some time.")
                    return None, None, False
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
                        return None, None, False

                    cursor.execute("SELECT user_id, username, password_hash, salt FROM users WHERE username = ?", (username,))
                    user = cursor.fetchone()
                    if user and db.check_password(password, user[2], user[3]):
                        st.session_state.logged_in = True
                        user_id = user[0]
                        return username, user_id, True
                    else:
                        st.error("Username or password not valid")
                else:
                    st.error("Please enter username and password")
        return None, None, False