# ============================================================
# ALM-2 - Module 2
# BUILD THE BEST SUPERVISED LEARNING BASELINE
# Abalone Dataset
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

file_path = "abalone.csv"

df = pd.read_csv(file_path)

print("\n==============================")
print("DATASET")
print("==============================")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 3. UNDERSTAND THE DATASET
# ------------------------------------------------------------

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 4. CHECK DUPLICATE VALUES
# ------------------------------------------------------------

print("\nNumber of Duplicate Rows:",
      df.duplicated().sum())


# ------------------------------------------------------------
# 5. BASIC STATISTICS
# ------------------------------------------------------------

print("\n==============================")
print("STATISTICAL SUMMARY")
print("==============================")

print(df.describe())


# ------------------------------------------------------------
# 6. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=["Rings"])

y = df["Rings"]


print("\n==============================")
print("FEATURES AND TARGET")
print("==============================")

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Rings")


# ------------------------------------------------------------
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ------------------------------------------------------------
# 8. DATA PREPROCESSING
# ------------------------------------------------------------

# Numerical features:
# Missing values are replaced with median
# Features are standardized

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical features:
# Missing values are replaced with most frequent value
# Categories are converted into numerical values

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine numerical and categorical preprocessing

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ------------------------------------------------------------
# 9. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================")

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# MODEL 1
# LINEAR REGRESSION
# ============================================================

model1 = Pipeline([
    ("preprocessing", preprocessor),

    ("model", LinearRegression())
])


# ------------------------------------------------------------
# 10. TRAIN LINEAR REGRESSION
# ------------------------------------------------------------

model1.fit(X_train, y_train)


# ------------------------------------------------------------
# 11. PREDICT USING LINEAR REGRESSION
# ------------------------------------------------------------

y_pred1 = model1.predict(X_test)


# ------------------------------------------------------------
# 12. EVALUATE LINEAR REGRESSION
# ------------------------------------------------------------

mae1 = mean_absolute_error(y_test, y_pred1)

mse1 = mean_squared_error(y_test, y_pred1)

rmse1 = np.sqrt(mse1)

r2_1 = r2_score(y_test, y_pred1)


print("\n==============================")
print("MODEL 1 - LINEAR REGRESSION")
print("==============================")

print("MAE  :", round(mae1, 4))
print("MSE  :", round(mse1, 4))
print("RMSE :", round(rmse1, 4))
print("R2   :", round(r2_1, 4))


# ============================================================
# MODEL 2
# RANDOM FOREST REGRESSOR
# ============================================================

model2 = Pipeline([
    ("preprocessing", preprocessor),

    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# ------------------------------------------------------------
# 13. TRAIN RANDOM FOREST
# ------------------------------------------------------------

model2.fit(X_train, y_train)


# ------------------------------------------------------------
# 14. PREDICT USING RANDOM FOREST
# ------------------------------------------------------------

y_pred2 = model2.predict(X_test)


# ------------------------------------------------------------
# 15. EVALUATE RANDOM FOREST
# ------------------------------------------------------------

mae2 = mean_absolute_error(y_test, y_pred2)

mse2 = mean_squared_error(y_test, y_pred2)

rmse2 = np.sqrt(mse2)

r2_2 = r2_score(y_test, y_pred2)


print("\n==============================")
print("MODEL 2 - RANDOM FOREST")
print("==============================")

print("MAE  :", round(mae2, 4))
print("MSE  :", round(mse2, 4))
print("RMSE :", round(rmse2, 4))
print("R2   :", round(r2_2, 4))


# ============================================================
# 16. 5-FOLD CROSS VALIDATION
# ============================================================

print("\n==============================")
print("5-FOLD CROSS VALIDATION")
print("==============================")


# Linear Regression

cv1 = cross_val_score(
    model1,
    X,
    y,
    cv=5,
    scoring="r2"
)


# Random Forest

cv2 = cross_val_score(
    model2,
    X,
    y,
    cv=5,
    scoring="r2"
)


print("\nLinear Regression CV R2 Scores:")
print(cv1)

print("Mean CV R2:",
      round(cv1.mean(), 4))


print("\nRandom Forest CV R2 Scores:")
print(cv2)

print("Mean CV R2:",
      round(cv2.mean(), 4))


# ============================================================
# 17. MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        mae1,
        mae2
    ],

    "MSE": [
        mse1,
        mse2
    ],

    "RMSE": [
        rmse1,
        rmse2
    ],

    "Test R2": [
        r2_1,
        r2_2
    ],

    "Mean CV R2": [
        cv1.mean(),
        cv2.mean()
    ]
})


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison.round(4).to_string(index=False))


# ============================================================
# 18. SELECT BEST BASELINE MODEL
# ============================================================

if cv1.mean() > cv2.mean():

    best_model = "Linear Regression"
    best_score = cv1.mean()

else:

    best_model = "Random Forest"
    best_score = cv2.mean()


print("\n==============================")
print("FINAL BASELINE MODEL")
print("==============================")

print("Recommended Model:", best_model)

print("Mean CV R2:",
      round(best_score, 4))


# ============================================================
# 19. FINAL JUSTIFICATION
# ============================================================

print("\n==============================")
print("MODEL JUSTIFICATION")
print("==============================")


if best_model == "Linear Regression":

    print("""
Linear Regression is selected as the final baseline model.

Reasons:
1. It achieved better cross-validation R2 performance.
2. It is simple and computationally efficient.
3. It is easy to interpret.
4. It provides a strong baseline for predicting abalone rings.
5. It can be used as a reference model for future improvements.
""")


else:

    print("""
Random Forest Regressor is selected as the final baseline model.

Reasons:
1. It achieved better cross-validation R2 performance.
2. It can capture non-linear relationships between the
   physical measurements and the number of rings.
3. It generally performs well with mixed feature relationships.
4. It provides better predictive performance than the
   Linear Regression baseline.
5. Therefore, it is selected as the final baseline model.
""")


# ============================================================
# 20. SAMPLE PREDICTION
# ============================================================

print("\n==============================")
print("SAMPLE PREDICTION")
print("==============================")


# Take one existing test sample

sample = X_test.iloc[[0]]

actual_value = y_test.iloc[0]

prediction = model2.predict(sample)[0]


print("Actual Rings    :", actual_value)

print("Predicted Rings :", round(prediction, 2))