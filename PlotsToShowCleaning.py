"""
Created on Sat Mar 22 17:50:42 2025

@author: nicho
"""

import pandas as pd
import matplotlib.pyplot as plt

"""
This script compares total web visit trends before and after outlier removal.

Key Step:

1. **Compare Original vs. Outlier-Removed Data**:
   - Loads original and cleaned datasets.
   - Aggregates total web visits by timestamp.
   - Plots both time series to visualize the impact of removing outliers.
"""

# Load the dataset
data = pd.read_csv('web_data_with_product_types')
#data = pd.read_csv('web_data_outliers_removed.csv')
#data = pd.read_csv('web_data_outlier_removed_incl_neg_peaks.csv')

# Convert 'datetime' to a proper datetime object
data['datetime'] = pd.to_datetime(data['datetime'])


# Compute total visits for web and app
total_web_visits = data['visits_web'].sum()
total_app_visits = data['visits_app'].sum()
grand_total_visits = total_web_visits + total_app_visits


# ------------------------------------
# Figure 1: Plot Total Web Visits (Original vs Outliers Removed)
# ------------------------------------
# Group by datetime and sum web visits for both datasets
data_outliers_removed = pd.read_csv('web_data_outlier_removed.csv')
total_visits_original = data.groupby('datetime')['visits_web'].sum()
total_visits_outliers_removed = data_outliers_removed.groupby('datetime')['visits_web'].sum()

# Create a new DataFrame for easy plotting
comparison_df = pd.DataFrame({
    'Original': total_visits_original,
    'Outliers_Removed': total_visits_outliers_removed
})

# Plot the total web visits comparison
plt.figure(figsize=(12, 6))

plt.plot(comparison_df.index, comparison_df['Original'], label='Original Data', color='blue', linestyle='-')
plt.plot(comparison_df.index, comparison_df['Outliers_Removed'], label='Outliers Removed', color='red', linestyle='--')

# Add labels, title, and legend
plt.xlabel('Datetime')
plt.ylabel('Total Web Visits')
plt.title('Comparison of Total Web Visits (Original vs Outliers Removed)')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xlim(pd.Timestamp("2023-10-23 20:00:00"),pd.Timestamp("2023-10-23 22:00:00"))
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
