# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:17:42 2025

@author: 531725ns
"""

"""
This file finds the ads that are aired at exactly the same time and removes them based on certain conditions. 
The web traffic data at these points is subsequently adjusted.
"""
import pandas as pd
import numpy as np

#This function finds the duplicates and removes certain ones based on a heuristic
def filter_duplicates(data):
    df = data
    grouped_ads = df.groupby(['date', 'time'])

    grp_percent_list = []
    group_list = []
    for (date,time), group in grouped_ads:
        group_grp = group['indexed_gross_rating_point'].sum()
        grp_list =[]
        grp_percent_list = []
        for index, row in group.iterrows():
            grp_list.append(row['indexed_gross_rating_point'])
            grp_percent_list.append(row['indexed_gross_rating_point']/group_grp)
        if len(grp_list) >= 2 and min(grp_percent_list) > 0.1 and min(grp_list) > 2 : #remove only the ones that meet the conditions
            group_list.append(((date,time),group))
            
    return group_list

#this function removes the found duplicates from the commercial airings
def remove_duplicates_from_commercials(data, group_list):
    # Extract the (date, time) pairs from DuplicateList
    to_remove = [item[0] for item in group_list]  # Get all (date, time) pairs
    
    # Filter out rows where (date, time) matches any in to_remove
    cleaned_data = data[~data[['date', 'time']].apply(tuple, axis=1).isin(to_remove)]
    
    return cleaned_data

        
#----------------------------------------------------------------------------------------------------------------------------------
#Modify Web traffic data
#----------------------------------------------------------------------------------------------------------------------------------

data = pd.read_csv('web_data_outlier_removed.csv')
data = data.groupby(['datetime'])[['visits_app', 'visits_web']].agg('sum').reset_index()
data['datetime'] = pd.to_datetime(data['datetime'])
Commercials = pd.read_csv("Web + broadcasting data - Broadcasting data.csv", sep=";")


group_list = filter_duplicates(Commercials)
datetime_list = []
for (date, time), group in group_list:
    datetime_str = f"{date} {time}"
    datetime_obj = pd.to_datetime(datetime_str)
    datetime_list.append(datetime_obj)

for time_spot in datetime_list:
    start_time = time_spot - pd.Timedelta(minutes=10)
    end_time = time_spot - pd.Timedelta(minutes=1)

    reference_data = data[
         (data['datetime'] >= start_time) &
         (data['datetime'] <= end_time)
         ]
    
    mean_visits_web = reference_data['visits_web'].mean()
    variance_visits_web = reference_data['visits_web'].var()
    mean_visits_app = reference_data['visits_app'].mean()
    variance_visits_app = reference_data['visits_app'].var()

    replace_start = time_spot
    replace_end = time_spot + pd.Timedelta(minutes=3)

    replace_indices = data[
        (data['datetime'] >= replace_start) &
        (data['datetime'] <= replace_end) 
        ].index

    random_visits_web = np.random.normal(
        loc=mean_visits_web,
        scale=np.sqrt(variance_visits_web),
        size=len(replace_indices)
    )
    random_visits_app = np.random.normal(
        loc=mean_visits_app,
        scale=np.sqrt(variance_visits_app),
        size=len(replace_indices)
    )

    data.loc[replace_indices, 'visits_web'] = np.clip(random_visits_web, a_min=0, a_max=None)
    data.loc[replace_indices, 'visits_app'] = np.clip(random_visits_app, a_min=0, a_max=None)
    
data.to_csv('C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/web_data_cleaned_full.csv', index=False)

#----------------------------------------------------------------------------------------------------------------------------------
#Modify Commercial data
#----------------------------------------------------------------------------------------------------------------------------------

DuplicateList = filter_duplicates(Commercials)

Cleaned_data = remove_duplicates_from_commercials(Commercials, DuplicateList)

Cleaned_data.to_csv('C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/Commercial_cleaned_full.csv', index=False)
