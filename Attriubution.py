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
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month

# Define input (X) and output (y)
categorical_features = ['day_of_week', 'month', 'channel', 'position_in_break', 'program_cat_before', 'program_cat_after']
numerical_features = ['indexed_gross_rating_point', 'hour']
target = 'peak_impact'  # Adjusted target column name

X = data[categorical_features + numerical_features]
y = data[target]

# Preprocessing: Encode categorical variables and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features),
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

# The base value of SHAP is the average of all the predicted peak_impact values, and it should remain constant across the entire dataset.
base_value = np.mean(y)

# Manually assign the base value to SHAP (to ensure consistency with the mean of peak_impact)
shap_values.base_values = np.array([base_value] * len(shap_values.values))


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
pd.set_option('display.float_format', '{:.6f}'.format)  # Set decimal precision

# Display the first few rows of the SHAP feature importance
print("\nSHAP Feature Importance:\n", shap_df.head(10))  # Display the first 10 rows of the SHAP values DataFrame

# ----------------------------------------------------------------------------------------------------------------------
# FEATURE IMPORTANCE BAR PLOT
# ----------------------------------------------------------------------------------------------------------------------

# Create a RandomForest model (using a non-predictive model)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_encoded_dense, y)  # Fit model on encoded data

# Get feature importances from the RandomForest model
importances = rf_model.feature_importances_

# Create a bar plot for feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from RandomForest')
plt.gca().invert_yaxis()  # Invert the y-axis to have the most important feature at the top
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# AVERAGE SHAP VALUE BAR PLOT
# ----------------------------------------------------------------------------------------------------------------------

# Calculate the average absolute SHAP value for each feature
average_shap_values = np.mean(np.abs(shap_values.values), axis=0)

# Create a bar plot for average SHAP value per feature
plt.figure(figsize=(10, 6))
plt.barh(feature_names, average_shap_values)
plt.xlabel('Average Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('Average SHAP Value for Each Feature')
plt.gca().invert_yaxis()  # Invert the y-axis to have the most impactful feature at the top
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
# SHAP WATERFALL PLOT: visualizes how different features contributed to a specific prediction made by your model.
# The predicted output (peak impact) -> SHAP explains the model's prediction with these features not the actual observed outcome.
# The expected model output across all data (average prediction) represents the baseline prediction if no feature values were known.
#----------------------------------------------------------------------------------------------------------------------

# Let's select an individual row to explain, e.g., the first row
row_index = 0  # You can change this to any index to inspect other rows

# Extract the SHAP explanation for the specific row
shap_explanation = shap.Explanation(
    values=shap_values.values[row_index],  # SHAP contributions for the selected row
    base_values=shap_values.base_values[row_index],  # Base value for the prediction
    feature_names=preprocessor.get_feature_names_out()  # The feature names for the transformed data
)

# Generate the SHAP waterfall plot for the selected row
shap.waterfall_plot(shap_explanation)

plt.show()

#----------------------------------------------------------------------------------------------------------------------
# Partial Dependence Plots (PDPs)
# PDPs show how predictions change when a specific feature is varied while keeping all others constant
# This helps confirm if relationships are linear, non-linear, or have interactions
# PDP plots show the marginal effect of a feature on the predicted outcome. Unlike SHAP (which explains individual predictions)
#----------------------------------------------------------------------------------------------------------------------

# Display PDPs for selected features
features_to_plot = ['num__hour', 'num__indexed_gross_rating_point']  # You can modify this list

# Plot the PDP for the selected features
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    rf_model, X_encoded_dense, features=features_to_plot, feature_names=feature_names, ax=ax
)

plt.show()





