# tesla_rf_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf

# Load Tesla stock data
df = yf.download('TSLA', start='2015-01-01', end='2024-12-31')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# Feature Engineering
df['Tomorrow_Close'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df.drop('Tomorrow_Close', axis=1)
y = df['Tomorrow_Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics in a table
metrics_df = pd.DataFrame({
    'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R² Score'],
    'Value': [mse, rmse, mae, r2]
})

print(metrics_df)

import joblib
# Save the model
joblib.dump(model, 'tesla_rf_model.pkl')    
