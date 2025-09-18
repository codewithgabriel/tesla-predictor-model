# streamlit_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

# Load model and data
@st.cache_data
def load_data():
    df = yf.download('TSLA', start='2015-01-01', end='2024-12-31')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

df = load_data()

# Train model
X = df.drop('Tomorrow_Close', axis=1)
y = df['Tomorrow_Close']
import joblib

model = joblib.load('tesla_rf_model.pkl')


# Streamlit UI
st.title("📈 Tesla Stock Price Predictor")
st.write("Enter today's stock data to predict tomorrow's closing price.")

open_price = st.number_input("Open Price", value=250.0)
high_price = st.number_input("High Price", value=255.0)
low_price = st.number_input("Low Price", value=245.0)
close_price = st.number_input("Close Price", value=252.0)
volume = st.number_input("Volume", value=30000000)

input_data = pd.DataFrame([[open_price, high_price, low_price, close_price, volume]],
                          columns=['Open', 'High', 'Low', 'Close', 'Volume'])

if st.button("Predict Tomorrow's Close"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Closing Price: ${prediction:.2f}")

# Show evaluation metrics
st.subheader("📊 Model Performance")
mse = mean_squared_error(y, model.predict(X))
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, model.predict(X))
r2 = r2_score(y, model.predict(X))

metrics_df = pd.DataFrame({
    'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R² Score'],
    'Value': [mse, rmse, mae, r2]
})

st.table(metrics_df)
