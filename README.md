# 🚗 Vehicle Price Prediction using Machine Learning

## 📌 Project Overview

This project builds a regression model to predict vehicle prices based on technical and categorical specifications.

The dataset contains real-world vehicle listing data with high variance and noise. Multiple regression models were trained, evaluated, and compared to determine the best-performing algorithm.

---

## 🎯 Objective

To predict vehicle price (continuous target variable) using machine learning techniques while ensuring good generalization and low prediction error.

---

## 🧠 Problem Type

Supervised Learning – Regression

---

## 📊 Dataset Description

The dataset includes vehicle listing features such as:

- Engine volume  
- Fuel type  
- Manufacturer  
- Gear box type  
- Drive wheels  
- Mileage  
- Interior features  

### Data Challenges:

- Missing values (`-` entries)
- Non-numeric columns stored as objects
- High-cardinality categorical features
- Outliers in price distribution
- Noisy real-world data

---

## 🛠️ Data Preprocessing

The following preprocessing steps were performed:

- Replaced '-' values with NaN  
- Converted numeric-like object columns to numeric  
- Removed irrelevant columns (ID, Model, Doors, Color, Wheel)  
- Encoded categorical variables using Label Encoding  
- Dropped missing values  
- Removed outliers using IQR method (applied only to target variable)  
- Train-test split (80% training, 20% testing)  

---

## 📈 Exploratory Data Analysis (EDA)

- Analyzed price distribution (right-skewed distribution observed)
- Generated correlation heatmap
- Identified that no single feature dominates price prediction
- Observed multi-feature interaction patterns

These findings justified the use of tree-based ensemble models.

---

## 🤖 Models Implemented

The following regression models were trained and evaluated:

- Linear Regression  
- Lasso Regression  
- Ridge Regression  
- Gradient Boosting Regressor  
- Random Forest Regressor  

---

## 📏 Evaluation Metrics

Models were evaluated using:

- **R² Score**
- **RMSE (Root Mean Squared Error)**
- **5-Fold Cross Validation**

---

## 🏆 Model Performance Summary

| Model                | Test R² | RMSE  |
|----------------------|----------|--------|
| Linear Regression    | ~0.22    | ~11,300 |
| Lasso Regression     | ~0.22    | ~11,300 |
| Ridge Regression     | ~0.22    | ~11,300 |
| Gradient Boosting    | ~0.67    | ~7,340  |
| Random Forest        | ~0.79    | ~5,887  |

---

## 🥇 Final Model Selection

**RandomForestRegressor** was selected as the final model because:

- Highest Test R² (~0.79)
- Lowest RMSE
- Stable 5-fold cross-validation performance
- Strong ability to capture non-linear relationships
- Robust handling of feature interactions

---

## 🔍 Key Observations

- Linear models underfit due to strong non-linear pricing patterns.
- Tree-based ensemble models significantly improved predictive performance.
- Vehicle pricing depends on multiple interacting variables rather than a single dominant feature.
- Slight overfitting observed in Random Forest, but acceptable generalization.

---

## ⚠️ Limitations

- External market factors (demand trends, accident history, resale value) not included
- No extensive hyperparameter tuning
- Label encoding may introduce implicit ordinal relationships (acceptable for tree-based models)

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Advanced feature engineering
- Log transformation of skewed target variable
- Feature selection optimization
- Deployment using Flask or FastAPI

---

## 💾 Model Saving

The trained Random Forest model is saved as:

vehicle_price_model.pkl

It can be loaded using:

```python
import joblib

model = joblib.load("vehicle_price_model.pkl")
```

---

## 📂 Project Structure

```
vehicle-price-prediction-ml/
│
├── vehicle_price_prediction.ipynb
├── vehicle_price_model.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

Clone the repository:

```
git clone https://github.com/Monisha-b-r/vehicle-price-prediction-ml.git
cd vehicle-price-prediction-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the notebook:

```
jupyter notebook
```

---

## 📌 Conclusion

Among all regression models evaluated, Random Forest provided the best balance between accuracy and generalization.

This project demonstrates:

- End-to-end ML workflow  
- Data preprocessing & feature engineering  
- Model comparison & evaluation  
- Cross-validation  
- Model persistence  

It highlights the importance of choosing the right algorithm for non-linear, real-world datasets.
