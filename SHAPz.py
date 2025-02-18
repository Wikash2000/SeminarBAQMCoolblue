# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:48:20 2025

@author: 531725ns
"""

import pandas as pd
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


file_path = "SHAPinput.csv"
data = pd.read_csv(file_path)

data.loc[data['uplift'] < 0, 'uplift'] = 0
     
data['norm_peak'] = data['uplift'] / data['indexed_gross_rating_point']
data.loc[data['indexed_gross_rating_point'] < 1, 'norm_peak'] = data['uplift']
data['norm_peak'] = data['norm_peak'] * 1000


data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

# Extract useful datetime features
data['hour'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek
data['month'] = data['Datetime'].dt.month
data['hour'] = data['hour'].astype('category')
data['day_of_week'] = data['day_of_week'].astype('category')
data['month'] = data['month'].astype('category')

target_column = 'norm_peak'  # Change this to your actual target column
X = data.drop(columns=['norm_peak','uplift','Datetime','indexed_gross_rating_point', 'commercial_id'])  
y = data[target_column]  

cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Define One-Hot Encoder
encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None), cat_features)],
    remainder='passthrough'
)

# Fit and transform
X_encoded = encoder.fit_transform(X)

# Get feature names
encoded_feature_names = encoder.get_feature_names_out()

# Convert to DataFrame
X_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)

# 🔹 Split Data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 🔹 Train XGBoost Model (Classification)
model = XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 🔹 Compute SHAP Values 
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 🔹 Feature Importance Summary Plot
shap.summary_plot(shap_values, X_test)

for cat in cat_features:
    # Find all encoded columns related to this categorical feature
    cat_cols = [col for col in X_test.columns if cat in col]
    
    # Sum SHAP values for all one-hot encoded versions of the categorical variable
    cat_shap_values = X_test[cat_cols].values * shap_values.values[:, [X_test.columns.get_loc(col) for col in cat_cols]]

    # Compute mean SHAP effect per category (group by categories)
    shap_mean = np.mean(cat_shap_values, axis=0)  # Mean across samples, keeping category dimension

    # Plot Partial Dependence (SHAP Effect on Probability)
    plt.figure(figsize=(8, 5))
    plt.bar(cat_cols, shap_mean)  # Use categorical feature names
    plt.xticks(rotation=45)
    plt.xlabel(cat)
    plt.ylabel("SHAP Value")
    plt.title(f"Partial Dependence Plot for {cat}")
    plt.show()