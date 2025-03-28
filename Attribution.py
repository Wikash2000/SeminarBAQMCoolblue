# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:52:13 2025

@author: nicho
"""

"""
This script fits a second xgboost to attribute the uplift scores to specific features of the commercials, to understand which factors drive a successful commercial. 
Interpretations of effect are provided by feature importance and partial dependence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay

# Set global font size for readability
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})

#Select outcome variable
version = "visits_web_scaled"
#version = "visits_app_scaled"

medium = version.split("_")[1]

file_path = f"SHAPinput_{version}.csv"
data = pd.read_csv(file_path)

data.loc[data['uplift'] < 0, 'uplift'] = 0
data['norm_peak'] = data['uplift'] / (data['indexed_gross_rating_point']+1)

outcome = "uplift"

#--------------------------------------------------------------------------------------------------------
#Print top 20 commercials for uplift and normalized peak
#--------------------------------------------------------------------------------------------------------

filtered_data = data.loc[data['flight_description'] != 'Coolblue_2023_11_Black Friday wk 45-47'] # sort if needed

# Sort the filtered data by the chosen outcome in descending order
sorted_data = filtered_data.sort_values(by=outcome, ascending=False)

top20 = sorted_data.head(20)
# Export the top 20 rows of sorted data to a CSV file
top20.to_csv(f'top_20_{version}_{outcome}_filtered.csv',index=False)


#--------------------------------------------------------------------------------------------------------
#Build models
#--------------------------------------------------------------------------------------------------------

# Feature engineering
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data['hour'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek
data['month'] = data['Datetime'].dt.month
data['hour'] = data['hour'].astype('category')
data['day_of_week'] = data['day_of_week'].astype('category')
data['month'] = data['month'].astype('category')
data['same_program'] = data['same_program'].astype('category')
data['tag_ons'] = data['tag_ons'].astype('category')
data['time_of_day'] = data['Datetime'].dt.hour + (data['Datetime'].dt.minute / 60.0)  # Hour + fractional minutes

# Build the model
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

#--------------------------------------------------------------------------------------------------------
#Model1 Get GRP pdp
#--------------------------------------------------------------------------------------------------------
target_column = 'uplift' 
X = data.drop(columns=['norm_peak','uplift','Datetime', 'commercial_id'])  
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
X_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names, dtype='float32')
X_encoded.columns = [col if col != 'remainder__indexed_gross_rating_point' else 'Indexed GRP' for col in X_encoded.columns]

model.fit(X_encoded, y)
# Generate the PDP for GRP
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    model, X_encoded, features=['Indexed GRP'],  # Use the shortened version
    grid_resolution=100, ax=ax
)

ax.set_title("Partial Dependence Plot for Indexed Gross Rating Point") 
ax.set_ylabel("Average predicted uplift") # Set title
plt.show()


#--------------------------------------------------------------------------------------------------------
#Model2 Get categorcial PDP + FI
#--------------------------------------------------------------------------------------------------------
target_column = 'norm_peak' 
X = data.drop(columns=['norm_peak','uplift','indexed_gross_rating_point','time_of_day', 'Datetime', 'commercial_id'])  
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
X_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names, dtype='float32')

# Fit the model
model.fit(X_encoded, y)

# Define category label mappings
category_label_mapping = {
    col: col.split('_')[-1]  # Extracts only the category name (e.g., "category_A" → "A")
    for cat in cat_features for col in X_encoded.columns if cat in col
}

# Special mapping for day_of_week
day_of_week_mapping = {
    '0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday',
    '4': 'Friday', '5': 'Saturday', '6': 'Sunday'
}

# Special mapping for month
month_mapping = {
    '1': 'January', '2': 'February', '3': 'March', '4': 'April', '5': 'May',
    '6': 'June', '7': 'July', '8': 'August', '9': 'September',
    '10': 'October', '11': 'November', '12': 'December'
}

# Update mapping for specific categorical variables
for col in X_encoded.columns:
    if 'day_of_week' in col:
        day_value = col.split('_')[-1]  # Extract numeric day
        category_label_mapping[col] = day_of_week_mapping.get(day_value, day_value)
    elif 'month' in col:
        month_value = col.split('_')[-1]  # Extract numeric month
        category_label_mapping[col] = month_mapping.get(month_value, month_value)


# Categorical PDP with renamed labels
for cat in cat_features:
    cat_cols = [col for col in X_encoded.columns if cat in col]

    pd_values = []
    for col in cat_cols:
        X_temp = X_encoded.copy()
        X_temp[cat_cols] = 0  # Set all categories in this feature to 0
        X_temp[col] = 1  # Activate only this category
        pd_values.append(model.predict(X_temp).mean())  # Get mean prediction

    # Generate clean labels
    x_labels = [category_label_mapping[col] for col in cat_cols]

    # Plot Partial Dependence as a bar plot
    x_positions = np.arange(len(cat_cols))
    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, pd_values)
    plt.xticks(x_positions, x_labels, rotation=45, ha="right")
    plt.xlabel(cat)
    plt.ylabel("Average predicted uplift")
    plt.title(f"Partial Dependence Plot for {cat}")
    plt.show()

# Feature importance
feature_importances = model.feature_importances_

# Create a DataFrame with feature names and their importances
importance_df = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': feature_importances
})

# Aggregating the importances for categorical features
aggregated_importances = {}
for cat in cat_features:
    cat_cols = [col for col in X_encoded.columns if cat in col]
    aggregated_importances[cat] = importance_df[importance_df['feature'].isin(cat_cols)]['importance'].sum()

# Convert to DataFrame and sort
aggregated_importances_df = pd.DataFrame(
    list(aggregated_importances.items()), columns=['Feature', 'Aggregated Importance']
).sort_values(by='Aggregated Importance', ascending=True)

# Plot aggregated feature importances
plt.figure(figsize=(10, 6))
plt.barh(aggregated_importances_df['Feature'], aggregated_importances_df['Aggregated Importance'], color='skyblue')
plt.xlabel('Aggregated Importance')
plt.title('Aggregated Feature Importances for Categorical Variables')
plt.show()
