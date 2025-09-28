import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize the features
X_mean, X_std = np.mean(X_train), np.std(X_train)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Model Implementation
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

# Training and Evaluation
learning_rate = 0.01
iterations = 1000

weight, bias = initialize_parameters()
trained_weight, trained_bias, costs = gradient_descent(X_train_scaled, y_train, weight, bias, learning_rate, iterations)

# Evaluate the model
y_pred_test = linear_regression_model(X_test_scaled, trained_weight, trained_bias)

def r_squared(y_true, y_pred):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)

r2_score = r_squared(y_test, y_pred_test)
print("R-squared score:", r2_score)

# Visualization
# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
y_pred_all = linear_regression_model(X_test_scaled, trained_weight, trained_bias)
plt.plot(X_test, y_pred_all, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.savefig('linear_regression_plot.png')
plt.close()

# Plot the cost function
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.savefig('cost_function_plot.png')
plt.close()

print("Plots saved as linear_regression_plot.png and cost_function_plot.png")
