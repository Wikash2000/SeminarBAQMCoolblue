import pandas as pd
import numpy as np

"""
! note this is the first step, please also use outlierRemoval3 to replace the downpeaks due to servor errors as well!
This script processes web traffic data to address specific data inconsistencies and outliers.

Key functionalities:
1. Loads and preprocesses a dataset of web and app visits with associated product types.
2. Ensures the data is sorted by datetime for accurate temporal analysis.
3. Modifies midnight records:
   - Handles potential discrepancies around midnight for each unique day and product type.
   - Uses values from the minute before and after midnight to estimate the visits for midnight.
4. Corrects outliers:
   - Replaces anomalous spikes in app visit data for the 'other' product type during a specified period (3:00–5:00 on 1 November).
   - Uses a random distribution derived from a reference period to generate realistic replacement values.
5. Ensures data consistency:
   - Sets app visit values for non-'other' product types to zero during the spike period.
   - Clips any negative values resulting from the random generation process.
6. Outputs the processed data to a new CSV file: 'web_data_outliers_removed.csv'.

This script is designed for data preprocessing and cleaning, particularly to handle temporal anomalies and data spikes in web analytics.
"""



# Load the dataset
data = pd.read_csv('web_data_with_product_types.csv')

# Convert 'datetime' to a proper datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Sort the data by datetime to ensure proper sequence
data = data.sort_values(by='datetime')

# Create a new DataFrame to hold the modified data
modified_data = data.copy()

# Previous code for midnight adjustments (commented out to avoid rerunning)
# # Iterate over each unique day in the dataset
unique_days = data['datetime'].dt.date.unique()

for current_day in unique_days:
    # Define the timestamp for midnight (00:00) of the current day
    midnight = pd.Timestamp(f"{current_day} 00:00:00")

    # Check if midnight exists in the data
    if midnight in data['datetime'].values:
        # Get the rows for the minute before and the minute after midnight
        minute_after = midnight + pd.Timedelta(minutes=1)
        previous_day = current_day - pd.Timedelta(days=1)
        minute_before = pd.Timestamp(f"{previous_day} 23:59:00")

        # Handle categories (product_type) separately
        for product_type in data['product_type'].unique():
            # Get the relevant rows for the specific product_type
            before_row = data[(data['datetime'] == minute_before) & (data['product_type'] == product_type)]
            after_row = data[(data['datetime'] == minute_after) & (data['product_type'] == product_type)]

            # Handle missing rows (e.g., first day or last day)
            if before_row.empty and not after_row.empty:
                visits_web = after_row['visits_web'].values[0]
                visits_app = after_row['visits_app'].values[0]
            elif after_row.empty and not before_row.empty:
                visits_web = before_row['visits_web'].values[0]
                visits_app = before_row['visits_app'].values[0]
            elif not before_row.empty and not after_row.empty:
                visits_web = (before_row['visits_web'].values[0] + after_row['visits_web'].values[0]) / 2
                visits_app = (before_row['visits_app'].values[0] + after_row['visits_app'].values[0]) / 2
            else:
                continue  # Skip if both before and after rows are missing

            # Update the midnight row in the modified DataFrame
            condition = (modified_data['datetime'] == midnight) & (modified_data['product_type'] == product_type)
            modified_data.loc[condition, 'visits_web'] = visits_web

            # Ensure only 'other' has non-zero values for visits_app
            if product_type == 'other':
                modified_data.loc[condition, 'visits_app'] = visits_app
            else:
                modified_data.loc[condition, 'visits_app'] = 0.0

# Adjust spike at 3:00 to 5:00 on 1 November for the 'other' category
# Define the spike period and reference period
spike_start = pd.Timestamp("2023-11-01 03:00:00")
spike_end = pd.Timestamp("2023-11-01 05:00:00")
reference_start = pd.Timestamp("2023-11-01 02:00:00")
reference_end = pd.Timestamp("2023-11-01 02:59:00")

# Calculate the mean and variance for the 'other' category during the reference period
reference_data = data[
    (data['datetime'] >= reference_start) &
    (data['datetime'] <= reference_end) &
    (data['product_type'] == 'other')
]

mean_visits_app = reference_data['visits_app'].mean()
variance_visits_app = reference_data['visits_app'].var()

# Set a random seed for reproducibility
np.random.seed(42)

# Replace visits_app for the 'other' category during the spike period
spike_indices = modified_data[
    (modified_data['datetime'] >= spike_start) &
    (modified_data['datetime'] <= spike_end) &
    (modified_data['product_type'] == 'other')
].index

# Generate random values with the calculated mean and variance
random_values = np.random.normal(loc=mean_visits_app, scale=np.sqrt(variance_visits_app), size=len(spike_indices))

# Update the modified_data DataFrame
modified_data.loc[spike_indices, 'visits_app'] = random_values

# Ensure visits_app for all other categories remains zero during the spike period
non_other_spike_indices = modified_data[
    (modified_data['datetime'] >= spike_start) &
    (modified_data['datetime'] <= spike_end) &
    (modified_data['product_type'] != 'other')
].index
modified_data.loc[non_other_spike_indices, 'visits_app'] = 0.0

# Ensure no negative values in visits_app during the spike period
modified_data.loc[spike_indices, 'visits_app'] = modified_data.loc[spike_indices, 'visits_app'].clip(lower=0)

# Save the modified data to a new CSV file
modified_data.to_csv('web_data_outliers_removed.csv', index=False)

print("Spike at 3:00 to 5:00 on 1 November has been replaced, negative values removed, and the results saved to 'web_data_outliers_removed.csv'.")
