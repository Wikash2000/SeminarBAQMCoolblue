import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This script processes web traffic data to address temporal outliers and spikes in visits across different product types. 
!Please use outlierRemoval1 before using this one to also remove the 00:00 outliers!

Key functionalities:
1. Converts a list of timestamps into pandas `Timestamp` objects for precise datetime manipulation.
2. Iterates over specified time spots and product types to:
   - Calculate the mean and variance of web visits within a reference period (8 minutes before each time spot).
   - Replace web visit values during a defined replacement period (4 minutes after each time spot) with random values generated based on the calculated mean and variance.
3. Ensures no negative values are introduced during the replacement process.
4. Outputs a modified dataset to a new CSV file: 'web_data_outlier_removed_incl_neg_peaks.csv'.
5. Visualizes the original and modified web visits over time:
   - Provides comparison plots for individual product types.
   - Includes a summary plot of total web visits across all product types.

This script is designed for data cleaning and visualization, particularly for handling and smoothing temporal anomalies in web traffic data.
"""



# Load the dataset
data = pd.read_csv('web_data_outliers_removed.csv')

# Convert 'datetime' to a proper datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Sort the data by datetime to ensure proper sequence
data = data.sort_values(by='datetime')

# Create a new DataFrame to hold the modified data
modified_data = data.copy()

# Define the list of timestamps
# Format: YYYY-MM-DD HH:MM:SS
timestamps = [
    "2023-09-12 00:40",
    "2023-09-13 07:10", "2023-09-13 21:10",
    "2023-09-14 14:03", "2023-09-15 01:04",
    "2023-09-16 14:13", "2023-09-16 18:45",
    "2023-09-22 21:45", "2023-09-26 15:00",
    "2023-09-27 07:11", "2023-10-02 17:12", "2023-10-02 16:47",
    "2023-10-02 17:22", "2023-10-02 22:44", "2023-10-03 16:45",
    "2023-10-03 16:50", "2023-10-04 07:10",
    "2023-10-07 13:46", "2023-10-09 18:11",
    "2023-10-10 20:22", "2023-10-10 23:15",
    "2023-10-11 07:10", "2023-10-12 13:53",
    "2023-10-12 16:56", "2023-10-12 17:00",
    "2023-10-13 20:25", "2023-10-13 23:16",
    "2023-10-16 19:20", "2023-10-17 12:07", "2023-10-17 07:07",
    "2023-10-18 07:11", "2023-10-18 21:29",
    "2023-10-20 01:12", "2023-10-20 01:20",
    "2023-10-22 14:03", "2023-10-22 22:17",
    "2023-10-23 19:01", "2023-10-23 21:05",
    "2023-10-23 21:25", "2023-10-23 22:54",
    "2023-10-23 23:44", "2023-10-24 00:27",
    "2023-10-24 00:30", "2023-10-25 07:02",
    "2023-10-29 16:21", "2023-10-30 09:43",
    "2023-10-30 10:47", "2023-11-01 13:36",
    "2023-11-01 13:38", "2023-11-02 15:37",
    "2023-11-05 08:56", "2023-11-05 23:45",
    "2023-11-13 23:26", "2023-11-15 23:39",
    "2023-11-17 21:52", "2023-11-18 16:09",
    "2023-11-18 17:07", "2023-11-18 19:55",
    "2023-11-18 21:30", "2023-11-19 19:20",
    "2023-11-20 13:59", "2023-11-20 14:03",
    "2023-11-21 09:16", "2023-11-22 22:08",
    "2023-11-24 22:59", "2023-11-24 23:05",
    "2023-11-24 23:09", "2023-11-24 23:14",
    "2023-11-24 23:19", "2023-11-25 22:44",
    "2023-11-27 18:38", "2023-11-28 22:52",
    "2023-11-30 22:41", "2023-12-01 20:05",
    "2023-12-01 21:46", "2023-12-02 21:43",
    "2023-12-02 22:39", "2023-12-04 01:20",
    "2023-12-04 12:25", "2023-12-04 12:30", "2023-12-04 18:53",
    "2023-12-04 19:02", "2023-12-04 19:13",
    "2023-12-04 21:00", "2023-12-04 21:30",
    "2023-12-05 07:15", "2023-12-05 16:19",
    "2023-12-07 08:10", "2023-12-07 12:16",
    "2023-12-09 01:20", "2023-12-16 19:12",
    "2023-12-17 08:58", "2023-12-17 20:13",
    "2023-12-19 09:37", "2023-12-19 20:42",
    "2023-12-19 21:20", "2023-12-20 06:05", "2023-12-20 16:38",
    "2023-12-20 22:03", "2023-12-21 20:28",
    "2023-12-21 22:05", "2023-12-22 09:33",
    "2023-12-22 22:44", "2023-12-22 23:59",
    "2023-12-23 00:05", "2023-12-23 00:10",
    "2023-12-23 00:16", "2023-12-24 02:24",
    "2023-12-24 02:27", "2023-12-24 02:31",
    "2023-12-24 02:56", "2023-12-26 14:09",
]

# Convert the list of timestamps into pandas Timestamp objects
time_spots = [pd.Timestamp(ts) for ts in timestamps]
# Iterate over each specified time spot
for time_spot in time_spots:
    # Convert the string time spot to a Timestamp object
    time_spot = pd.Timestamp(time_spot)

    # Iterate over each unique product_type in the dataset
    for product_type in data['product_type'].unique():
        # Filter data for the current product_type
        product_data = data[data['product_type'] == product_type]

        # Define the period for mean and variance calculation (8 minutes before)
        start_time = time_spot - pd.Timedelta(minutes=8)
        end_time = time_spot - pd.Timedelta(minutes=1)

        # Calculate mean and variance for the 8-minute window
        reference_data = product_data[
            (product_data['datetime'] >= start_time) &
            (product_data['datetime'] <= end_time)
            ]

        # Skip if there is no data in the reference period
        if reference_data.empty:
            continue

        mean_visits_web = reference_data['visits_web'].mean()
        variance_visits_web = reference_data['visits_web'].var()

        # Define the exact replacement period (current time spot + 4 minutes)
        replace_start = time_spot
        replace_end = time_spot + pd.Timedelta(minutes=4)

        # Get the indices for the rows to replace for this product_type
        replace_indices = modified_data[
            (modified_data['datetime'] >= replace_start) &
            (modified_data['datetime'] <= replace_end) &
            (modified_data['product_type'] == product_type)
            ].index

        # Set a random seed for reproducibility
        np.random.seed(42)

        # Generate random values based on the calculated mean and variance
        random_visits_web = np.random.normal(
            loc=mean_visits_web,
            scale=np.sqrt(variance_visits_web),
            size=len(replace_indices)
        )

        # Update the modified_data DataFrame
        modified_data.loc[replace_indices, 'visits_web'] = random_visits_web

        # Ensure no negative values are present in 'visits_web'
        modified_data.loc[replace_indices, 'visits_web'] = modified_data.loc[replace_indices, 'visits_web'].clip(
            lower=0)

# Save the modified data to a new CSV file
modified_data.to_csv('web_data_outlier_removed_incl_neg_peaks.csv', index=False)

print("Data modification complete. The results have been saved to 'web_data_outlier_removed_incl_neg_peaks.csv'.")

# Plotting combined data for all product types
plt.figure(figsize=(14, 7))

# Plot individual product types' web visits for original and modified data
for product_type in data['product_type'].unique():
    # Filter data for the current product type
    original_subset = data[data['product_type'] == product_type]
    modified_subset = modified_data[modified_data['product_type'] == product_type]

    # Plot the original visits_web
    plt.plot(original_subset['datetime'], original_subset['visits_web'], alpha=0.5, label=f"Original {product_type}")

    # Plot the modified visits_web
    plt.plot(modified_subset['datetime'], modified_subset['visits_web'], alpha=0.8, linestyle='--',
             label=f"Modified {product_type}")

# Calculate total web visits
data['total_visits_web'] = data.groupby('datetime')['visits_web'].transform('sum')
modified_data['total_visits_web'] = modified_data.groupby('datetime')['visits_web'].transform('sum')

# Plot total web visits for original and modified data
plt.plot(data['datetime'], data['total_visits_web'], color='blue', linewidth=2, label="Total Original Web Visits")
plt.plot(modified_data['datetime'], modified_data['total_visits_web'], color='red', linewidth=2, linestyle='--',
         label="Total Modified Web Visits")

# Adding titles and labels
plt.title("Web Visits Over Time (By Product Type and Total)", fontsize=16)
plt.xlabel("Datetime", fontsize=14)
plt.ylabel("Web Visits", fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
