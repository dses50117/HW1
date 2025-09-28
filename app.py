import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Model Code from linear_regression.py ---

def initialize_parameters():
    weight = np.random.randn(1, 1)
    bias = np.zeros((1, 1))
    return weight, bias

def linear_regression_model(X, weight, bias):
    return np.dot(X, weight) + bias

def cost_function(y_true, y_pred):
    m = len(y_true)
    return (1/(2*m)) * np.sum(np.square(y_pred - y_true))

def gradient_descent(X, y, weight, bias, learning_rate, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_pred = linear_regression_model(X, weight, bias)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        weight -= learning_rate * dw
        bias -= learning_rate * db
        
        cost = cost_function(y, y_pred)
        costs.append(cost)
        
    return weight, bias, costs

def r_squared(y_true, y_pred):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    if ss_tot == 0:
        return 1.0 # Perfect prediction
    return 1 - (ss_res / ss_tot)

# --- Streamlit App ---

st.title('Interactive Linear Regression Model')

# Sidebar for user inputs
st.sidebar.header('Model Parameters')
num_points = st.sidebar.slider('Number of data points (n)', 50, 500, 100)
slope_a = st.sidebar.slider("Coefficient 'a' (y = ax + b + noise)", -10.0, 10.0, 3.0)
noise = st.sidebar.slider('Noise Variance (var)', 0.0, 10.0, 1.0)

# Generate synthetic data based on user input
np.random.seed(0)
X = 2 * np.random.rand(num_points, 1)
y = 4 + slope_a * X + np.random.randn(num_points, 1) * noise

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize features
X_mean, X_std = np.mean(X_train), np.std(X_train)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Train the model
learning_rate = 0.01
iterations = 1000
weight, bias = initialize_parameters()
trained_weight, trained_bias, costs = gradient_descent(X_train_scaled, y_train, weight, bias, learning_rate, iterations)

# Evaluate the model
y_pred_test = linear_regression_model(X_test_scaled, trained_weight, trained_bias)
r2 = r_squared(y_test, y_pred_test)
st.write(f'**R-squared score:** {r2:.4f}')

# --- New Features ---

# Display Model Coefficients
st.subheader('Model Coefficients')
st.write(f"**Slope (a):** {trained_weight[0][0]:.4f}")
st.write(f"**Intercept (b):** {trained_bias[0][0]:.4f}")

# Top 5 Outliers
st.subheader('Top 5 Outliers')
y_pred_all_data = linear_regression_model((X - X_mean) / X_std, trained_weight, trained_bias)
errors = np.abs(y - y_pred_all_data)
outlier_indices = np.argsort(errors, axis=0)[-5:].flatten()
outlier_data = {
    'X': X[outlier_indices].flatten(),
    'y_actual': y[outlier_indices].flatten(),
    'y_predicted': y_pred_all_data[outlier_indices].flatten(),
    'Error': errors[outlier_indices].flatten()
}
outlier_df = pd.DataFrame(outlier_data)
st.dataframe(outlier_df)

# --- Plots ---

# Regression plot
st.subheader('Linear Regression Plot')
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(X_train, y_train, color='blue', label='Training data')
ax1.scatter(X_test, y_test, color='green', label='Testing data')
# Highlight outliers
ax1.scatter(X[outlier_indices], y[outlier_indices], color='red', s=100, label='Top 5 Outliers', edgecolors='black')
y_pred_plot = linear_regression_model(X_test_scaled, trained_weight, trained_bias)
ax1.plot(X_test, y_pred_plot, color='red', linewidth=2, label='Regression line')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression')
ax1.legend()
st.pyplot(fig1)

# Cost function plot
st.subheader('Cost Function Convergence')
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(range(iterations), costs)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Cost Function Convergence')
st.pyplot(fig2)
