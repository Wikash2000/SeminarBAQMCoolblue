import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
This script processes web traffic data to correct anomalies and outliers. It consists of three main steps:

1. Midnight Adjustments:
   Ensures continuity in visit counts by interpolating values from adjacent timestamps for each product type at midnight.

2. Spike Correction for App Visits:
   Detects and corrects an anomalous spike in app visits for the 'other' category between 3:00–5:00 AM on November 1.
   Replaces these values using a random distribution derived from the previous hour’s data.

3. Negative Peak Detection & Correction:
   Identifies unusually low website visit counts for the 'other' category using a rolling median-based threshold.
   Replaces detected outliers with random values based on the preceding 10-minute window.

Finally, the cleaned dataset is saved, and plots visualize the modifications for validation.
'''

# Set a random seed for reproducibility
np.random.seed(0)

# Load the dataset
original_data = pd.read_csv('web_data_with_product_types')

# Convert 'datetime' to a proper datetime object
original_data['datetime'] = pd.to_datetime(original_data['datetime'])

# Sort the data by datetime to ensure proper sequence
original_data = original_data.sort_values(by='datetime')

# Create a new DataFrame to hold the modified data
modified_data = original_data.copy()

# Step 1: Adjusting midnight records
unique_days = original_data['datetime'].dt.date.unique()

for current_day in unique_days:
    midnight = pd.Timestamp(f"{current_day} 00:00:00")

    if midnight in original_data['datetime'].values:
        minute_after = midnight + pd.Timedelta(minutes=1)
        previous_day = current_day - pd.Timedelta(days=1)
        minute_before = pd.Timestamp(f"{previous_day} 23:59:00")

        for product_type in original_data['product_type'].unique():
            before_row = original_data[
                (original_data['datetime'] == minute_before) & (original_data['product_type'] == product_type)]
            after_row = original_data[
                (original_data['datetime'] == minute_after) & (original_data['product_type'] == product_type)]

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
                continue

            condition = (modified_data['datetime'] == midnight) & (modified_data['product_type'] == product_type)
            modified_data.loc[condition, 'visits_web'] = visits_web

            if product_type == 'other':
                modified_data.loc[condition, 'visits_app'] = visits_app
            else:
                modified_data.loc[condition, 'visits_app'] = 0.0

# Step 2: Handling spike at 3:00 to 5:00 on 1 November
spike_start = pd.Timestamp("2023-11-01 03:00:00")
spike_end = pd.Timestamp("2023-11-01 05:00:00")
reference_start = pd.Timestamp("2023-11-01 02:00:00")
reference_end = pd.Timestamp("2023-11-01 02:59:00")

reference_data = original_data[
    (original_data['datetime'] >= reference_start) &
    (original_data['datetime'] <= reference_end) &
    (original_data['product_type'] == 'other')
    ]

mean_visits_app = reference_data['visits_app'].mean()
variance_visits_app = reference_data['visits_app'].var()

spike_indices = modified_data[
    (modified_data['datetime'] >= spike_start) &
    (modified_data['datetime'] <= spike_end) &
    (modified_data['product_type'] == 'other')
    ].index

random_values = np.random.normal(loc=mean_visits_app, scale=np.sqrt(variance_visits_app), size=len(spike_indices))
modified_data.loc[spike_indices, 'visits_app'] = np.clip(random_values, a_min=0, a_max=None)

non_other_spike_indices = modified_data[
    (modified_data['datetime'] >= spike_start) &
    (modified_data['datetime'] <= spike_end) &
    (modified_data['product_type'] != 'other')
    ].index
modified_data.loc[non_other_spike_indices, 'visits_app'] = 0.0


# Step 3: Detect and replace negative peaks in visits_web for 'other' category

other_data = modified_data[modified_data['product_type'] == 'other'].copy()
other_data = other_data.sort_values(by='datetime')
other_data['rolling_median_prev_10'] = other_data['visits_web'].rolling(window=10, min_periods=1).median().shift(1)

target_threshold = 0.55
other_data['threshold'] = target_threshold * other_data['rolling_median_prev_10']
other_data['hour'] = other_data['datetime'].dt.hour
other_data['is_negative_peak'] = (other_data['visits_web'] < other_data['threshold']) & ~(
    other_data['hour'].between(1, 6))

negative_peaks = other_data[other_data['is_negative_peak']]

for index, row in negative_peaks.iterrows():
    time_spot = row['datetime']
    product_type = 'other'
    product_data = modified_data[modified_data['product_type'] == product_type]

    start_time = time_spot - pd.Timedelta(minutes=10)
    end_time = time_spot - pd.Timedelta(minutes=1)

    reference_data = product_data[
        (product_data['datetime'] >= start_time) &
        (product_data['datetime'] <= end_time)
        ]

    if reference_data.empty:
        continue

    mean_visits_web = reference_data['visits_web'].mean()
    variance_visits_web = reference_data['visits_web'].var()

    replace_start = time_spot
    replace_end = time_spot + pd.Timedelta(minutes=3)

    replace_indices = modified_data[
        (modified_data['datetime'] >= replace_start) &
        (modified_data['datetime'] <= replace_end) &
        (modified_data['product_type'] == product_type)
        ].index

    random_visits_web = np.random.normal(
        loc=mean_visits_web,
        scale=np.sqrt(variance_visits_web),
        size=len(replace_indices)
    )

    modified_data.loc[replace_indices, 'visits_web'] = np.clip(random_visits_web, a_min=0, a_max=None)

# Save final dataset
modified_data.to_csv('web_data_outlier_removed.csv', index=False)
