import pandas as pd
import matplotlib.pyplot as plt

"""
This script analyzes and visualizes web and app visit data over time by product type.

Key Steps:
1. **Load and Prepare Data**:
   - Loads the dataset from a CSV file and converts the 'datetime' column to a datetime object.
   - Displays dataset info and sample rows for verification.

2. **Compute Visit Statistics**:
   - Calculates total web visits, total app visits, and the grand total (web + app).
   - Prints the results for a quick overview.

3. **Visualize Web Visits**:
   - Groups and pivots data to show web visits for each product type and total visits over time.
   - Creates a line plot for web visits by product type and total visits.

4. **Visualize App Visits**:
   - Groups and pivots data to show app visits for each product type and total visits over time.
   - Creates a line plot for app visits by product type and total visits.

The script provides insights into visit trends by product type and highlights overall traffic patterns for both web and app platforms.
"""


# Load the dataset
data = pd.read_csv('web_data_with_product_types.csv')
#data = pd.read_csv('web_data_outliers_removed.csv')
#data = pd.read_csv('web_data_outlier_removed_incl_neg_peaks.csv')

summary_stats1 = data['visits_app'].describe()
summary_stats2 = data['visits_web'].describe()

print(summary_stats1)
print(summary_stats2)

# Convert 'datetime' to a proper datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

print(data.info())
print(data.head())


# Compute total visits for web and app
total_web_visits = data['visits_web'].sum()
total_app_visits = data['visits_app'].sum()
grand_total_visits = total_web_visits + total_app_visits

# Print the results
print(f"Total Web Visits: {total_web_visits}")
print(f"Total App Visits: {total_app_visits}")
print(f"Grand Total Visits (Web + App): {grand_total_visits}")

# ------------------------------------
# Figure 1: Plot Web Visits
# ------------------------------------

# Group by datetime and product_type for web visits
category_data_web = data.groupby(['datetime', 'product_type'])['visits_web'].sum().reset_index()

# Pivot the data to make product types columns and datetime the index
pivoted_data_web = category_data_web.pivot(index='datetime', columns='product_type', values='visits_web').fillna(0)

# Add a 'Total_Visits' column for web visits
pivoted_data_web['Total_Visits'] = pivoted_data_web.sum(axis=1)

# Create the plot for web visits
plt.figure(figsize=(12, 6))

# Plot each product type
for category in pivoted_data_web.columns[:-1]:  # Exclude 'Total_Visits'
    plt.plot(pivoted_data_web.index, pivoted_data_web[category], label=category)

# Plot total visits
plt.plot(pivoted_data_web.index, pivoted_data_web['Total_Visits'], label='Total_Visits', linewidth=2, linestyle='--', color='black')

# Add labels, title, and legend
plt.xlabel('Datetime')
plt.ylabel('Visits (Web)')
plt.title('Website Visits by Product Type Over Time')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Move legend outside the plot
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# ------------------------------------
# Figure 2: Plot App Visits
# ------------------------------------

# Group by datetime and product_type for app visits
category_data_app = data.groupby(['datetime', 'product_type'])['visits_app'].sum().reset_index()

# Pivot the data to make product types columns and datetime the index
pivoted_data_app = category_data_app.pivot(index='datetime', columns='product_type', values='visits_app').fillna(0)

# Add a 'Total_Visits' column for app visits
pivoted_data_app['Total_Visits'] = pivoted_data_app.sum(axis=1)

# Create the plot for app visits
plt.figure(figsize=(12, 6))

# Plot each product type
for category in pivoted_data_app.columns[:-1]:  # Exclude 'Total_Visits'
    plt.plot(pivoted_data_app.index, pivoted_data_app[category], label=category)

# Plot total visits
plt.plot(pivoted_data_app.index, pivoted_data_app['Total_Visits'], label='Total_Visits', linewidth=2, linestyle='--', color='black')

# Add labels, title, and legend
plt.xlabel('Datetime')
plt.ylabel('Visits (App)')
plt.title('App Visits by Product Type Over Time')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Move legend outside the plot
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
