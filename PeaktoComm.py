# -*- coding: utf-8 -*-
"""
Spyder Editor

This file takes as input raw traffic and counterfactual traffic values and attributes 'peak sizes' to individual commercials.
"""

import pandas as pd

peaks = pd.read_csv('C:/Users/531725ns/OneDrive - Erasmus University Rotterdam/Master/Seminar/PeakAnalysis.csv')
peaks["Datetime"] = pd.to_datetime(peaks["Datetime"])
Commercial = pd.read_csv("C:/Users/531725ns/OneDrive - Erasmus University Rotterdam/Master/Seminar/Web + broadcasting data - Broadcasting data.csv", sep=";")
Commercial['Datetime'] = pd.to_datetime(Commercial['date'] + ' ' + Commercial['time'],format='%m/%d/%Y %I:%M:%S %p') 
Commercial_max = Commercial.loc[Commercial.groupby('Datetime')['indexed_gross_rating_point'].idxmax()]

Merged_data = pd.merge(peaks, Commercial_max, on='Datetime', how='left')

def attribute_peaks_to_comm(df, threshold_c=5, effect_window=3):
    """
    Attributes traffic peaks to commercials based on presence and GRP weighting.

    Parameters:
    - df: Pandas DataFrame with columns:
        ['datetime', 'actual_traffic', 'predicted_traffic', 'commercial', 'commercial_id', 'GRP']
    - threshold_c: The threshold below which the effect is considered to have faded.
    - effect_window: Number of minutes the effect of an ad lasts.

    Returns:
    - DataFrame with assigned impact per commercial.
    """
    
    df = df.sort_values("datetime").reset_index(drop=True)
    df["impact"] = df["actual_traffic"] - df["predicted_traffic"]
    df["assigned_impact"] = 0.0  # Initialize impact assignment
    commercial_impacts = []  # Store results per commercial

    for idx, row in df[df["commercial"] == 1].iterrows():
        commercial_id = row["commercial_id"]
        grp = row["GRP"]
        start_time = row["datetime"]
        
        # Track impact window
        impact_sum = 0
        assigned_rows = []
        
        for i in range(idx, len(df)):
            if abs(df.at[i, "impact"]) < threshold_c:
                break  # Stop summing when the effect fades
            
            assigned_rows.append(i)
            impact_sum += df.at[i, "impact"]

        if impact_sum == 0:
            continue

        # Find other active commercials in this period
        end_time = df.at[assigned_rows[-1], "datetime"]
        active_ads = df[(df["datetime"] >= start_time) & 
                        (df["datetime"] <= end_time + pd.Timedelta(minutes=effect_window)) & 
                        (df["commercial"] == 1)]

        if len(active_ads) == 1:
            # No overlap, full impact goes to this commercial
            df.loc[assigned_rows, "assigned_impact"] += impact_sum
            commercial_impacts.append((commercial_id, impact_sum))
        else:
            # Overlapping commercials: distribute by GRP proportion
            total_grp = active_ads["GRP"].sum()
            for _, ad in active_ads.iterrows():
                share = ad["GRP"] / total_grp
                impact_for_ad = impact_sum * share
                df.loc[assigned_rows, "assigned_impact"] += impact_for_ad
                commercial_impacts.append((ad["commercial_id"], impact_for_ad))

    return df, pd.DataFrame(commercial_impacts, columns=["commercial_id", "impact"])