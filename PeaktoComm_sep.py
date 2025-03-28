# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:55:57 2025

@author: 531725ns
"""

"""
This file takes as input the predicted traffic and the counterfactual traffic and calculates the uplift caused by individual commercials.
For overlapping commercials, uplift is assigned proportionally to GRP.
"""

import pandas as pd

def calculate_uplift(version):
    Commercial = pd.read_csv("Commercial_cleaned_full.csv")
    peaks = pd.read_csv(f'PeakAnalysis_{version}.csv')
    peaks["Datetime"] = pd.to_datetime(peaks["Datetime"])
    
    Commercial['Datetime'] = pd.to_datetime(Commercial['date'] + ' ' + Commercial['time'],format='%m/%d/%Y %I:%M:%S %p') 
    Commercial = Commercial.loc[Commercial.groupby('Datetime')['indexed_gross_rating_point'].idxmax()]
    Commercial = Commercial.sort_values(by='Datetime')
    Commercial['same_program'] = (Commercial['program_before'] == Commercial['program_after']).astype(int)
    Commercial['tag_ons'] = Commercial['spotlength'].apply(lambda x: "0" if x == '15' else "1" if x == '15 + 10' else "2" if x in ['15 + 10 + 5', '15 + 10 + 10'] else "none")
    Commercial['commercial_id'] = range(1, len(Commercial) + 1)

    Merged_data = pd.merge(peaks, Commercial, on='Datetime', how='left')
    Merged_data['commercial'] = Merged_data['Datetime'].isin(Commercial['Datetime']).astype(int)
    Merged_data = Merged_data[['Datetime', 'Actual Prediction', 'CF Prediction', 'commercial', 'indexed_gross_rating_point', 'commercial_id']]
    data = Merged_data.copy()
    # Initialize a list to store results for each commercial
    uplift_results = []
    
    # Variables to track the uplift sum, active commercials, and their viewership
    uplift_sum = 0
    active_commercials = []
    active_viewership = []
    
    # To track if we are in an uplift period
    in_uplift_period = False
    
    # Iterate over the data by timestamp
    for index, row in data.iterrows():
        actual_traffic = row['Actual Prediction']
        counterfactual_traffic = row['CF Prediction']
        uplift = actual_traffic - counterfactual_traffic

        
        if in_uplift_period:
            if uplift > 0.001: #Otherwise the peak is practically over
                uplift_sum += uplift  # Accumulate the uplift
                
                # Track commercials airing during this period
                if row['commercial'] == 1 and row['commercial_id'] not in active_commercials:
                    active_commercials.append(row["commercial_id"])  # Add commercial airing in this period
                    active_viewership.append(row['indexed_gross_rating_point'])  # Track its viewership
            #If not, end the peak        
            else:
                if len(active_commercials) > 0:
                    total_viewership = sum(active_viewership)
                    for i in range(len(active_commercials)):
                        commercial_index = active_commercials[i]
                        viewership = active_viewership[i]
                        if total_viewership > 0:
                            commercial_uplift = uplift_sum * (viewership / total_viewership) #Assign proportional to GRP
                        else:
                            commercial_uplift = uplift_sum
                        # Append the result for the current commercial
                        uplift_results.append({
                            'commercial_id': commercial_index,
                            'uplift': commercial_uplift
                        })
                in_uplift_period = False
                uplift_sum = 0
                active_commercials = []
                active_viewership = []
        
        # If we are not in an uplift period but a commercial airs, start the uplift measuring period
        if row['commercial'] == 1 and not in_uplift_period:
            in_uplift_period = True  # Start a new uplift period
            uplift_sum = uplift  # Initialize uplift sum with current uplift
            active_commercials.append(row["commercial_id"])  # Start tracking this commercial
            active_viewership.append(row['indexed_gross_rating_point'])  # Track the viewership for this commercial
       
   
    # Convert the uplift results to a DataFrame
    uplift_df = pd.DataFrame(uplift_results)
    final_output = pd.merge(uplift_df, Commercial, on='commercial_id', how='left')
    final_output = final_output[['Datetime', 'commercial_id', 'uplift', 'indexed_gross_rating_point','channel','position_in_break', 'program_cat_before', 'program_cat_after','flight_description','same_program','tag_ons']]
    final_output.to_csv(f'SHAPinput_{version}.csv', index=False)

    return uplift_df

#Select the outcome variable
version = "visits_web_scaled" 
#version = "visits_app_scaled"

uplift_df = calculate_uplift(version)
print(uplift_df)
