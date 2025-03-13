# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:40:09 2025

@author: 531725ns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


#version = "visits_web_scaled"
version = "visits_app_scaled"

medium = version.split("_")[1]

file_path = f"SHAPinput_{version}.csv"
data = pd.read_csv(file_path)



category_counts = {col: data[col].value_counts() for col in data.select_dtypes(include=['object', 'category'])}
print(category_counts)

# # print(Commercial['flight_description'].unique())
p = data[data["channel"] == "Ziggo Sport Tennis"]
print(data[data["program_cat_after"] == "kinderen (t/m + 12 jaar)"])
p = data[data["program_cat_before"] == "populaire muziek & dans"]
#print(data.groupby("program_cat_before")["norm_peak"].describe())


data.loc[data['uplift'] < 0, 'uplift'] = 0

data['norm_peak'] = data['uplift'] / (data['indexed_gross_rating_point']+1)

print(min(data['norm_peak']))
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

# Extract useful datetime features
data['hour'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek
data['month'] = data['Datetime'].dt.month
data['hour'] = data['hour'].astype('category')
data['day_of_week'] = data['day_of_week'].astype('category')
data['month'] = data['month'].astype('category')
data['same_program'] = data['same_program'].astype('category')
data['tag_ons'] = data['tag_ons'].astype('category')


target_column = 'norm_peak' 
X = data.drop(columns=['norm_peak','uplift','indexed_gross_rating_point','Datetime', 'commercial_id'])  
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

# # 🔹 Split Data
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 🔹 Train XGBoost Model (Regression)

model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=200,
    use_label_encoder=False
)

model.fit(X_encoded, y)

# ---- Plot Partial Dependence for Categorical Features ----

for cat in cat_features:
    cat_cols = [col for col in X_encoded.columns if cat in col]

    pd_values = []
    for col in cat_cols:
        X_temp = X_encoded.copy()
        X_temp[cat_cols] = 0  # Set all categories in this feature to 0
        X_temp[col] = 1  # Activate only this category
        pd_values.append(model.predict(X_temp).mean())  # Get mean prediction

    # Plot Partial Dependence as a bar plot
    x_positions = np.arange(len(cat_cols))
    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, pd_values)
    plt.xticks(x_positions, cat_cols, rotation=45, ha="right")
    plt.xlabel(cat)
    plt.ylabel("Effect on Predicted Peak Impact")
    plt.title(f"Partial Dependence Plot for {cat}")
    plt.show()
    
    
# ---- Plot 2D Partial Dependence for interesting Categorical Features ----


# Variables for the plot
variable_x = 'spotlength'  # Example categorical variable for x-axis
variable_y = 'flight_description'  # Example categorical variable for y-axis

# Get the unique categories for each variable (without encoding)
cat_x_cols = [col for col in X_encoded.columns if variable_x in col]
cat_y_cols = [col for col in X_encoded.columns if variable_y in col]

# Initialize an array to hold the PDP values
pdp_values = np.zeros((len(cat_y_cols), len(cat_x_cols)))

# Iterate over all combinations of x and y categories in the grid
for i, y_cat in enumerate(cat_y_cols):  # Switch order of iteration
    for j, x_cat in enumerate(cat_x_cols):  # Switch order of iteration
        # Create a temporary dataset for the current combination of x and y
        X_temp = X_encoded.copy()
        
        # Set the values for x and y in the temporary dataset
        X_temp[cat_x_cols] = 0
        X_temp[x_cat] = 1  # Activate only this category for x
        X_temp[cat_y_cols] = 0
        X_temp[y_cat] = 1  # Activate only this category for y
        
        # Make predictions using the model
        pdp_values[i, j] = model.predict(X_temp).mean()  # Mean prediction for this combination

# Plot the 2D Partial Dependence Plot
plt.figure(figsize=(12, 8))

# Create a heatmap with categorical values
plt.imshow(pdp_values, cmap="viridis", aspect="auto", origin="lower")

# Set the ticks for x and y axes to use category names (swap axes)
plt.xticks(np.arange(len(cat_x_cols)), cat_x_cols, rotation=45, ha="right")
plt.yticks(np.arange(len(cat_y_cols)), cat_y_cols)

# Add a colorbar to show the effect on predicted values
plt.colorbar(label="Effect on Predicted Peak Impact")

plt.xlabel(variable_x)
plt.ylabel
