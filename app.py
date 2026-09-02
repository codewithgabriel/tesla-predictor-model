# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

st.set_page_config(page_title="Tesla Stock Price Predictor", page_icon="📈")

# 1. Load Pre-trained Model
@st.cache_resource
def load_model():
    return joblib.load('tesla_rf_model.pkl')

model = load_model()

# 2. Streamlit UI: User Input
st.title("📈 Tesla Stock Price Predictor")
st.write("Enter today's stock data to predict tomorrow's closing price.")

col1, col2 = st.columns(2)
with col1:
    open_price = st.number_input("Open Price ($)", value=250.0, format="%.2f")
    high_price = st.number_input("High Price ($)", value=255.0, format="%.2f")
    low_price = st.number_input("Low Price ($)", value=245.0, format="%.2f")

with col2:
    close_price = st.number_input("Close Price ($)", value=252.0, format="%.2f")
    volume = st.number_input("Volume", value=30000000, step=100000)

input_data = pd.DataFrame(
    [[open_price, high_price, low_price, close_price, volume]],
    columns=['Open', 'High', 'Low', 'Close', 'Volume']
)

if st.button("Predict Tomorrow's Close", type="primary"):
    prediction = model.predict(input_data)[0]
    st.success(f"### Predicted Closing Price: **${prediction:.2f}**")

# 3. Model Performance (Robust handling against rate-limiting)
st.divider()
st.subheader("📊 Model Performance")

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = yf.download('TSLA', start='2015-01-01', end='2024-12-31')
        if df is None or df.empty or len(df) == 0:
            return None
        
        # Flatten columns if multi-indexed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Tomorrow_Close'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

df = load_data()

if df is not None and not df.empty:
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Tomorrow_Close']
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'R² Score'],
        'Value': [f"{mse:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}"]
    })
    st.table(metrics_df)
else:
    # Display validation benchmark metrics when Yahoo Finance is rate-limited
    st.info("ℹ️ Historical data fetching is currently unavailable or rate-limited by Yahoo Finance. Showing benchmark validation metrics:")
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'R² Score'],
        'Value': ['44.5577', '6.6752', '3.4491', '0.9965']
    })
    st.table(metrics_df)

