# ==========================================
# MATHEMATICS BEHIND AN ML MODEL
# Student Performance Prediction
# ==========================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv("student_performance_dataset.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)


# ==========================================
# 3. Select Features and Target
# ==========================================

# Input Features
features = [
    'study_time_hours',
    'attendance_percent',
    'sleep_hours',
    'previous_grade'
]

# Target Variable
target = 'final_exam_score'


# ==========================================
# 4. Create Feature Matrix X
# ==========================================

X = df[features].values

# Target Vector Y
Y = df[target].values.reshape(-1, 1)


print("\nFeature Matrix X:")
print(X[:5])

print("\nShape of X:")
print(X.shape)


print("\nTarget Vector Y:")
print(Y[:5])

print("\nShape of Y:")
print(Y.shape)


# ==========================================
# 5. Statistical Analysis
# ==========================================

print("\nStatistical Summary:")
print(df[features + [target]].describe())


# Correlation Matrix
correlation = df[features + [target]].corr()

print("\nCorrelation Matrix:")
print(correlation)


# ==========================================
# 6. Feature Normalization
# ==========================================

# Formula:
# X_normalized = (X - mean) / standard deviation

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_normalized = (X - X_mean) / X_std


print("\nNormalized Feature Matrix:")
print(X_normalized[:5])


# ==========================================
# 7. Initialize Model Parameters
# ==========================================

# Number of features
n_features = X_normalized.shape[1]

# Initialize weights
W = np.zeros((n_features, 1))

# Initialize bias
b = 0

# Learning rate
learning_rate = 0.01

# Number of iterations
iterations = 1000

# Number of students
m = len(Y)


# ==========================================
# 8. Gradient Descent Function
# ==========================================

def gradient_descent(X, Y, W, b, learning_rate, iterations):

    loss_history = []

    for i in range(iterations):

        # ----------------------------------
        # Forward Propagation
        # Mathematical Model:
        #
        # Y_pred = XW + b
        # ----------------------------------

        Y_pred = np.dot(X, W) + b


        # ----------------------------------
        # Loss Function
        #
        # MSE = (1/m) Σ(Y - Y_pred)^2
        # ----------------------------------

        loss = np.mean((Y - Y_pred) ** 2)

        loss_history.append(loss)


        # ----------------------------------
        # Calculate Gradients
        #
        # dW = (2/m) X^T(Y_pred - Y)
        #
        # db = (2/m) Σ(Y_pred - Y)
        # ----------------------------------

        dW = (2 / m) * np.dot(X.T, (Y_pred - Y))

        db = (2 / m) * np.sum(Y_pred - Y)


        # ----------------------------------
        # Update Parameters
        #
        # W = W - α(dW)
        #
        # b = b - α(db)
        # ----------------------------------

        W = W - learning_rate * dW

        b = b - learning_rate * db


    return W, b, loss_history


# ==========================================
# 9. Train Model
# ==========================================

W, b, loss_history = gradient_descent(
    X_normalized,
    Y,
    W,
    b,
    learning_rate,
    iterations
)


print("\nFinal Weights:")
print(W)

print("\nFinal Bias:")
print(b)


# ==========================================
# 10. Make Predictions
# ==========================================

Y_pred = np.dot(X_normalized, W) + b


print("\nActual vs Predicted Scores:")

results = pd.DataFrame({
    'Actual Score': Y.flatten(),
    'Predicted Score': Y_pred.flatten()
})

print(results.head(10))


# ==========================================
# 11. Calculate Final Mean Squared Error
# ==========================================

final_mse = np.mean((Y - Y_pred) ** 2)

print("\nFinal Mean Squared Error:")
print(final_mse)


# ==========================================
# 12. Visualize Loss Reduction
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

plt.title("Gradient Descent Optimization")

plt.show()


# ==========================================
# 13. Actual vs Predicted Graph
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(Y, Y_pred)

plt.xlabel("Actual Final Exam Score")
plt.ylabel("Predicted Final Exam Score")

plt.title("Actual vs Predicted Scores")

plt.show()


# ==========================================
# 14. Predict a New Student
# ==========================================

# Example student:
# Study Time = 5 hours
# Attendance = 90%
# Sleep = 7 hours
# Previous Grade = 80

new_student = np.array([[5, 90, 7, 80]])


# Normalize using training data mean and std

new_student_normalized = (
    new_student - X_mean
) / X_std


# Prediction

predicted_score = np.dot(
    new_student_normalized,
    W
) + b


print("\nPredicted Final Exam Score:")
print(predicted_score[0][0])