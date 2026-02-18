# 🚗 Vehicle Price Prediction using Machine Learning

## 📌 Project Overview

This project builds a regression model to predict vehicle prices based on technical and categorical specifications.

The dataset contains real-world vehicle listing data with high variance and noise. Multiple regression models were tested and compared to determine the best-performing algorithm.

---

## 🎯 Objective

To predict vehicle price (continuous target variable) using machine learning techniques.

---

## 🧠 Problem Type

Supervised Learning – Regression

---

## 📊 Dataset Description

The dataset includes features such as:

- Engine volume
- Fuel type
- Manufacturer
- Gear box type
- Drive wheels
- Mileage
- Interior features

Some preprocessing steps were required due to:
- Missing values
- Non-numeric columns
- Outliers
- High-cardinality categorical variables

---

## 🛠️ Data Preprocessing

The following steps were performed:

- Replaced '-' with NaN
- Converted numeric-like object columns to numeric
- Removed irrelevant columns (ID, Model, Doors, Color, Wheel)
- Encoded categorical variables using Label Encoding
- Dropped missing values
- Removed outliers using IQR method (on target variable)
- Train-test split (80%-20%)

---

## 📈 Models Implemented

The following regression models were trained and evaluated:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Gradient Boosting Regressor
- Random Forest Regressor

---

## 📏 Evaluation Metrics

Models were evaluated using:

- R² Score
- RMSE (Root Mean Squared Error)
- 5-Fold Cross Validation

---

## 🏆 Final Model

**RandomForestRegressor** was selected as the final model because:

- Highest Test R² (~0.79)
- Lowest RMSE
- Stable cross-validation performance
- Better handling of non-linear relationships

---

## 🔍 Key Observations

- Linear models underfit due to non-linear pricing patterns.
- Tree-based ensemble models handled feature interactions effectively.
- Vehicle pricing depends on multiple interacting variables rather than a single dominant feature.

---

## ⚠️ Limitations

- External market factors not included
- No extensive hyperparameter tuning
- Slight overfitting observed in Random Forest

Future improvements may include:
- Hyperparameter tuning
- Advanced feature engineering
- Feature selection optimization

---

## 💾 Model Saving

The trained Random Forest model is saved as:


