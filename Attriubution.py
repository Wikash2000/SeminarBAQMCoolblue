import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

#----------------------------------------------------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
#----------------------------------------------------------------------------------------------------------------------

file_path = "SHAPinput.csv"
data = pd.read_csv(file_path)

data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

# Extract useful datetime features
data['hour'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek
data['month'] = data['Datetime'].dt.month

# Function to evaluate 'spotlength' expressions and convert to numeric
def evaluate_spotlength(spotlength_str):
    numbers = spotlength_str.split(' + ')  # Split the string by '+'
    return sum(map(int, numbers))  # Convert the split parts into integers and return the sum

# Apply the function to the 'spotlength' column to convert it to numeric values
data['spotlength_numeric'] = data['spotlength'].apply(evaluate_spotlength)

# Define input (X) and output (y)
categorical_features = ['day_of_week', 'month', 'channel', 'position_in_break', 'program_cat_before', 'program_cat_after']
numerical_features = ['indexed_gross_rating_point', 'hour', 'spotlength_numeric']  # Add the new spotlength feature
target = 'uplift'

X = data[categorical_features + numerical_features]
y = data[target]

# Preprocessing: Encode categorical variables and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop=None), categorical_features),  # Keep all categories
        ('num', StandardScaler(), numerical_features)
    ]
)

#----------------------------------------------------------------------------------------------------------------------
# SHAP ANALYSIS
# The SHAP values explain how much each feature contributes to the predicted value.
# The predicted value itself remains the same as the original peak_impact (or the target value in your case).
#----------------------------------------------------------------------------------------------------------------------

# Fit the preprocessing steps and transform the data
X_encoded = preprocessor.fit_transform(X)

# Convert sparse matrix to dense array for SHAP
X_encoded_dense = X_encoded.toarray()

# Create SHAP explainer without a model (using the existing peak_impact values)
explainer = shap.Explainer(lambda x: y[:len(x)], X_encoded_dense)  # Lambda function simply returns the pre-existing peak_impact values

# Compute SHAP values
shap_values = explainer(X_encoded_dense)

# **Set SHAP values to zero for missing features**
shap_values.values[X_encoded_dense == 0] = 0  # Ensuring missing features have zero SHAP contribution

print("Baseline values for first 10 observations:\n", shap_values.base_values[:10])
print("Mean of y:", np.mean(y))


# Compute the sum of SHAP values for each observation
shap_sum = shap_values.values.sum(axis=1)

# Extract baseline values
baseline = shap_values.base_values

# Compute total contribution
total_contribution = baseline + shap_sum

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # No column wrapping

# Compare with actual output (y)
comparison_df = pd.DataFrame({
    "Baseline": baseline,
    "Sum of SHAP Values": shap_sum,
    "Predicted (Baseline + SHAP)": total_contribution,
    "Actual Peak Impact (y)": y
})

# Print first 10 rows to check
print(comparison_df.head(10))

#----------------------------------------------------------------------------------------------------------------------
# SHAP DATAFRAME
# Each row: a commercial, Each column: one feature, The value: how much each feature contributes to the final Peak Impact prediction
# Positive value → The feature increased Peak Impact, Negative value → The feature decreased Peak Impact
#----------------------------------------------------------------------------------------------------------------------

# Extract SHAP values as a NumPy array
shap_array = shap_values.values  # This gives you the SHAP values for each feature for each observation

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Convert SHAP values to a DataFrame
shap_df = pd.DataFrame(shap_array, columns=feature_names)

# Ensure all values are displayed in the DataFrame
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # No column wrapping

# Display the first few rows of the SHAP feature importance
print("\nSHAP Feature Importance:\n", shap_df.head(10))  # Display the first 10 rows of the SHAP values DataFrame

# ----------------------------------------------------------------------------------------------------------------------
# SHAP FEATURE IMPORTANCE BAR PLOT
# ----------------------------------------------------------------------------------------------------------------------

# Compute SHAP global feature importance using mean absolute values
shap_importance = np.abs(shap_values.values).mean(axis=0)

# Create a DataFrame for sorting
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean Absolute SHAP': shap_importance})

# Sort features by decreasing importance
shap_importance_df = shap_importance_df.sort_values(by='Mean Absolute SHAP', ascending=False)

# Plot SHAP feature importance in descending order
plt.figure(figsize=(10, 6))
plt.barh(shap_importance_df['Feature'], shap_importance_df['Mean Absolute SHAP'])
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('SHAP Feature Importance (Sorted)')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()

#----------------------------------------------------------------------------------------------------------------------
# SHAP SUMMARY PLOT
# Y-axis: The Y-axis itself has no numerical meaning for the individual features (all features together are ordered by the most impact, but this is not specific to any individual feature). It’s purely a visual trick to better spread the points, so that you can see the density of the values.
# X-axis (the SHAP value): This is the value impact. A negative value here means that the hour reduces the predicted Peak Impact -> for example, if a point on SHAP = -1.5, it means that the hour decreased the prediction by -1.5.
# SHAP value close to 0 on the X-axis → This hour has little impact on the prediction.
# Blue color: Low value of the feature -> meaning commercials aired at a low hour (e.g., 0:00 - 6:00 AM).
# Red color: High value of the feature -> commercials aired at a high hour (e.g., 18:00 - 23:59).
# Lighter shades = Mid-range hours (e.g., 10:00 - 16:00).
#----------------------------------------------------------------------------------------------------------------------

# Create SHAP Summary Plot
cmap = plt.get_cmap("coolwarm")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_encoded_dense, feature_names=preprocessor.get_feature_names_out(), cmap="coolwarm")

#----------------------------------------------------------------------------------------------------------------------
# SHAP DEPENDENCE PLOT
# X-axis: This represents the actual feature values. Y-axis (SHAP values): This tells us how much this feature contributes to the model's prediction.
# Positive SHAP values → This feature increased the prediction in those instances.
# Colors: This represents another feature (cat__program_cat_before_non_fictie). Since it appears binary (0 or 1): Blue (0) → This means program_cat_before_non_fictie was not present. Red (1) → This means program_cat_before_non_fictie was present.
# SHAP picks an "interaction feature" based on strongest interaction effects unless explicitly specified.
#----------------------------------------------------------------------------------------------------------------------

# This is the transformed feature name for 'hour'
feature_to_analyze = 'num__hour'

# Find the index of 'num__hour' in the transformed feature names
if feature_to_analyze in feature_names:
    feature_index = np.where(feature_names == feature_to_analyze)[0][0]
    print(f"Using feature index: {feature_index} for feature: {feature_to_analyze}")
else:
    raise ValueError(f"Feature '{feature_to_analyze}' not found in transformed data!")

# Now generate the SHAP dependence plot using the feature index
shap.dependence_plot(
    feature_index,
    shap_values.values,
    X_encoded_dense,
    feature_names=feature_names
)

plt.show()  # Show the plot

#----------------------------------------------------------------------------------------------------------------------
# Partial Dependence Plots (PDPs)
# PDPs show how predictions change when a specific feature is varied while keeping all others constant
# This helps confirm if relationships are linear, non-linear, or have interactions
# PDP plots show the marginal effect of a feature on the predicted outcome. Unlike SHAP (which explains individual predictions)
#----------------------------------------------------------------------------------------------------------------------
# Create a RandomForest model (using a non-predictive model)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_encoded_dense, y)  # Fit model on encoded data

# Display PDPs for selected features
features_to_plot = ['num__spotlength_numeric', 'num__indexed_gross_rating_point']  # You can modify this list

# Plot the PDP for the selected features
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    rf_model, X_encoded_dense, features=features_to_plot, feature_names=feature_names, ax=ax
)

plt.show()





