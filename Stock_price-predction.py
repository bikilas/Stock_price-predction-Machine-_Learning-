# Import necessary libraries
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# Define the stock ticker symbol
ticker_symbol = 'AAPL'

# Define the data range (e.g., past 5 years)
start_date = '2018-01-01'
end_date = '2023-01-01'

# Download stock data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the first few rows of the dataset
print("Sample Data:")
print(data.head())

# Save the data to a CSV file
data.to_csv('AAPL_Stock.csv')
print("Data saved to 'AAPL_Stock.csv'")


# Display the first few rows to verify loading
print(data.head())

# Data Cleaning and Preprocessing
# Drop rows with missing values
data.dropna(inplace=True)

# Check if 'Date' column is present and convert it to datetime
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])  # Convert to datetime format
    data.set_index('Date', inplace=True)  # Set 'Date' as the index

# Select features and target variable
X = data[['Open', 'High', 'Low', 'Volume']]  # Features
y = data['Close']  # Target variable

# Standardize the features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check linear regression assumptions (Optional)
# Linearity: Pair plot of features against the target
sns.pairplot(data[['Open', 'High', 'Low', 'Volume', 'Close']])
plt.show()

# Independence: Durbin-Watson test for autocorrelation
model = sm.OLS(y_train, X_train).fit()
print("Durbin-Watson test statistic:", sm.stats.durbin_watson(model.resid))

# Homoscedasticity & Normality: Q-Q plot of residuals
residuals = model.resid
sm.qqplot(residuals, line='s')
plt.show()

# Train the Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_val_pred = linear_model.predict(X_val)

# Hyperparameter Tuning with Ridge Regression (Optional)
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_model = Ridge()
grid_search = GridSearchCV(ridge_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best alpha value
print("Best alpha from Ridge:", grid_search.best_params_['alpha'])

# Train Ridge model with the best alpha
ridge_best = Ridge(alpha=grid_search.best_params_['alpha'])
ridge_best.fit(X_train, y_train)
y_val_pred_ridge = ridge_best.predict(X_val)

# Evaluate Model Performance
# Linear model performance on validation set
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)
print("Linear Model Validation MAE:", val_mae)
print("Linear Model Validation MSE:", val_mse)
print("Linear Model Validation R²:", val_r2)

# Ridge model performance on validation set
val_mae_ridge = mean_absolute_error(y_val, y_val_pred_ridge)
val_mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)
val_r2_ridge = r2_score(y_val, y_val_pred_ridge)
print("Ridge Model Validation MAE:", val_mae_ridge)
print("Ridge Model Validation MSE:", val_mse_ridge)
print("Ridge Model Validation R²:", val_r2_ridge)

# Test the final model on the test set (use Ridge model for better results if validated)
y_test_pred = ridge_best.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Test MAE:", test_mae)
print("Test MSE:", test_mse)
print("Test R²:", test_r2)

# Visualize Results
plt.figure(figsize=(10, 1))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_test_pred, label='Predicted Prices (Ridge)', color='red')
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.show()
