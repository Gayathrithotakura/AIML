"""
Tutorial 3 — Manual Linear and Logistic Regression using small datasets
CO2
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# -----------------------------
# PART A: Manual Linear Regression
# -----------------------------
# Hours studied -> exam marks
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([42, 48, 55, 63, 70], dtype=float)

x_mean, y_mean = x.mean(), y.mean()
slope = np.sum((x-x_mean)*(y-y_mean)) / np.sum((x-x_mean)**2)
intercept = y_mean - slope*x_mean

print("Manual linear regression")
print("slope =", slope)
print("intercept =", intercept)

y_hat = intercept + slope*x
print("Predictions =", y_hat)
print("MSE =", mean_squared_error(y, y_hat))

new_x = 6
print("Prediction for 6 hours =", intercept + slope*new_x)

# Verify with sklearn
lr = LinearRegression()
lr.fit(x.reshape(-1,1), y)
print("\nscikit-learn slope/intercept:", lr.coef_[0], lr.intercept_)

# -----------------------------
# PART B: Manual Logistic Regression idea
# -----------------------------
# We demonstrate the sigmoid and threshold first.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Example model: z = -4 + 0.08*hours + 1.5*attendance_fraction
hours = 40
attendance = 0.85
z = -4 + 0.08*hours + 1.5*attendance
p = sigmoid(z)

print("\nLogistic regression")
print("z =", z)
print("P(pass) =", p)
print("Class at threshold 0.5 =", int(p >= 0.5))
print("Class at threshold 0.7 =", int(p >= 0.7))

# Train a real logistic model on a small synthetic dataset
X = np.array([
    [2, 0.50], [3, 0.55], [4, 0.60], [5, 0.70],
    [6, 0.75], [7, 0.80], [8, 0.90], [9, 0.95]
])
y_class = np.array([0,0,0,0,1,1,1,1])

clf = LogisticRegression()
clf.fit(X, y_class)

print("\nLearned coefficients:", clf.coef_)
print("Intercept:", clf.intercept_)
print("Predicted probabilities:", clf.predict_proba(X)[:,1])
print("Accuracy:", accuracy_score(y_class, clf.predict(X)))
