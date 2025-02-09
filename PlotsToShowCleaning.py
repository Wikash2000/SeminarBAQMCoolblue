# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:11:23 2025

@author: 531725ns
"""

import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv("C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/web_data_with_product_types")
datacl = pd.read_csv("C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/web_data_cleaned_full.csv")
# Convert 'datetime' to a proper datetime object
data['datetime'] = pd.to_datetime(data['datetime'])
datacl['datetime'] = pd.to_datetime(datacl['datetime'])
# ------------------------------------
# Figure 1: Plot Total Web Visits (for the two-week interval)
# ------------------------------------

# Group by datetime only for total web visits (no product_type grouping)
sumdata = data.groupby(['datetime'])[['visits_app', 'visits_web']].agg('sum').reset_index()

# Create the plot for total web visits
plt.figure(figsize=(15, 8))

# Plot total web visits
plt.plot(sumdata['datetime'], sumdata['visits_web'], label='Total Web Visits', color='black')
# Add labels, title, and legend
plt.xlabel('Datetime')
plt.ylabel('Total Visits (Web)')
plt.title('Total Web Visits Over Time ')
plt.grid(True)
plt.tight_layout()
plt.xlim(pd.Timestamp("2023-11-19 18:00:00"),pd.Timestamp("2023-11-19 23:00:00"))

# Create the plot for cleaned total web visits
plt.figure(figsize=(15, 8))
# Plot cleaned total web visits
plt.plot(datacl['datetime'], datacl['visits_web'], label='Total Web Visits Cleaned', color='black')
# Add labels, title, and legend
plt.xlabel('Datetime')
plt.ylabel('Total Visits (Web)')
plt.title('Cleaned Total Web Visits Over Time')
plt.grid(True)
plt.tight_layout()
plt.xlim(pd.Timestamp("2023-11-19 18:00:00"),pd.Timestamp("2023-11-19 23:00:00"))
# ------------------------------------
# Figure 2: Plot Total App Visits (for the two-week interval)
# ------------------------------------

# Create the plot for total app visits
plt.figure(figsize=(15, 8))

# Plot total web visits
plt.plot(sumdata['datetime'], sumdata['visits_app'], label='Total App Visits', color='black')
# Add labels, title, and legend
plt.xlabel('Datetime')
plt.ylabel('Total Visits (App)')
plt.title('Total App Visits Over Time ')
plt.grid(True)
plt.tight_layout()
plt.xlim(pd.Timestamp("2023-11-18 18:00:00"),pd.Timestamp("2023-11-19 23:00:00"))

# Create the plot for cleaned total app visits
plt.figure(figsize=(15, 8))
# Plot cleaned total web visits
plt.plot(datacl['datetime'], datacl['visits_app'], label='Total App Visits Cleaned', color='black')
# Add labels, title, and legend
plt.xlabel('Datetime')
plt.ylabel('Total Visits (App)')
plt.title('Cleaned Total App Visits Over Time')
plt.grid(True)
plt.tight_layout()
plt.xlim(pd.Timestamp("2023-11-18 18:00:00"),pd.Timestamp("2023-11-19 23:00:00"))
