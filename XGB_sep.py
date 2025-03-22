# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:18:41 2025

@author: 531725ns
"""
"""
This file first evaluates an XGBoost on out of sample prediction of web traffic data and then refits the xgtboost to the full data, keeping 10% validation. 
Next, we make true predictions as well as counterfactual predcitions. These predictions are later used to calcualte uplift in traffic caused by commercials.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#------------------------------------------------------------------------------------------------------------------------
#FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------


def convert_columns_to_str(data, columns):
    for col in columns:
        data[col] = data[col].astype(str)
        
def get_evaluations(actual, prediction):
    metrics = {
      "RMSE": np.sqrt(mean_squared_error(actual, prediction)),
      "MAE": mean_absolute_error(actual, prediction),
      "R² Score": r2_score(actual, prediction)
  }
  
    return pd.DataFrame([metrics]) 

def make_counterfactual(dataset):
    cat_columns_to_set_zero = [
     "channel",
    "channel_lag_1", "channel_lag_2", "channel_lag_3", "channel_lag_4", "channel_lag_5",
    "program_cat_before", "program_cat_after", "position_in_break", "tag_ons",  
    "program_cat_before_lag_1", "program_cat_before_lag_2", "program_cat_before_lag_3", "program_cat_before_lag_4", "program_cat_before_lag_5",
    "program_cat_after_lag_1", "program_cat_after_lag_2", "program_cat_after_lag_3", "program_cat_after_lag_4", "program_cat_after_lag_5",
    "position_in_break_lag_1", "position_in_break_lag_2", "position_in_break_lag_3", "position_in_break_lag_4", "position_in_break_lag_5",
    "tag_ons_lag_1", "tag_ons_lag_2", "tag_ons_lag_3", "tag_ons_lag_4", "tag_ons_lag_5",  
    "flight_description", "flight_description_lag_1", "flight_description_lag_2", 
    "flight_description_lag_3", "flight_description_lag_4", "flight_description_lag_5",
    "same_program", "same_program_lag_1", "same_program_lag_2", "same_program_lag_3", "same_program_lag_4", "same_program_lag_5"
    ]
    
    dummy_columns = [col for col in dataset.columns if any(cat in col for cat in cat_columns_to_set_zero )]

    # Create a counterfactual dataset by setting all dummy variables to 0
    dataset_cf = dataset.copy()
    dataset_cf[dummy_columns] = False
    
    cont_columns_to_set_zero = ["indexed_gross_rating_point",
    "indexed_gross_rating_point_lag_1", "indexed_gross_rating_point_lag_2", "indexed_gross_rating_point_lag_3", "indexed_gross_rating_point_lag_4", "indexed_gross_rating_point_lag_5"]

    dataset_cf[cont_columns_to_set_zero] = 0
    return dataset_cf

def load_data():
    #Load traffic data
    Website = pd.read_csv("web_data_cleaned_full.csv")
    # Ensure 'datetime' column in Websites in datetime format
    Website['datetime'] = pd.to_datetime(Website['datetime'], errors='coerce')
    #Add a column for total traffic
    Website['traffic'] = Website['visits_web'] + Website['visits_app']
    
    #Load Commercial data
    Commercial = pd.read_csv("Commercial_cleaned_full.csv")
    # Combine 'Date' and 'Time' into a single datetime column
    Commercial['datetime'] = pd.to_datetime(Commercial['date'] + ' ' + Commercial['time'],
                                            format='%m/%d/%Y %I:%M:%S %p') 
    Commercial['same_program'] = (Commercial['program_before'] == Commercial['program_after']).astype(int)
    Commercial['tag_ons'] = Commercial['spotlength'].apply(lambda x: "0" if x == '15' else "1" if x == '15 + 10' else "2" if x in ['15 + 10 + 5', '15 + 10 + 10'] else "none")
    # Merge the two dataframes (to get all commercial variables)

    Website1 = Website.copy()
    Website1['commercial'] = Website1['datetime'].isin(Commercial['datetime']).astype(int)
    Commercial_max = Commercial.loc[Commercial.groupby('datetime')['indexed_gross_rating_point'].idxmax()]
    # Now merge the datasets
    Merged_data = pd.merge(Website1, Commercial_max, on='datetime', how='left')
    # Select only the columns you want to keep
    Merged_data = Merged_data[['datetime', 'traffic', 'visits_app', 'visits_web', 'indexed_gross_rating_point','channel','position_in_break', 'program_cat_before', 'program_cat_after', 'flight_description','same_program','tag_ons']]    
    
    #fill na with 0 
    Merged_data = Merged_data.assign(
    channel=Merged_data['channel'].fillna("0"),
    indexed_gross_rating_point=Merged_data['indexed_gross_rating_point'].fillna(0),
    position_in_break=Merged_data['position_in_break'].fillna("0"),
    program_cat_before=Merged_data['program_cat_before'].fillna("0"),
    program_cat_after=Merged_data['program_cat_after'].fillna("0"),
    flight_description=Merged_data['flight_description'].fillna("0"),
    same_program=Merged_data['same_program'].fillna("none"),
    tag_ons=Merged_data['tag_ons'].fillna("none")
    )  

    # filter period without commercial data
    Merged_data = Merged_data[Merged_data['datetime'] >= pd.Timestamp("2023-09-11")]
    data = Merged_data.copy()

    # Extract time-based features
    data["hour"] = data["datetime"].dt.hour
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["month"] = data["datetime"].dt.month
    data["minute"] = data["datetime"].dt.minute
    
    # Periodic time encoding for hour and day (sin/cos for circular encoding)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["minute_sin"] = np.sin(2 * np.pi * data["minute"] / 1440)
    data["minute_cos"] = np.cos(2 * np.pi * data["minute"] / 1440)

    
    # Ensure data is sorted by datetime
    data = data.sort_values("datetime")
      
    #scale data
    data["traffic_scaled"] = data[["traffic"]]*100
    data["visits_app_scaled"] = data[["visits_app"]]*1000
    data["visits_web_scaled"] = data[["visits_web"]]*100

    data["traffic_lag"] = data['traffic_scaled'].rolling(window=60,min_periods=1).mean().shift(1)
    data["visits_web_lag"] = data['visits_web_scaled'].rolling(window=60,min_periods=1).mean().shift(1)
    data["visits_app_lag"] = data['visits_app_scaled'].rolling(window=60,min_periods=1).mean().shift(1)
    

    for lag in range(1, 6):  # Now generating 5 lags (1 to 5)
        data[f"channel_lag_{lag}"] = data["channel"].shift(lag)
        data[f"indexed_gross_rating_point_lag_{lag}"] = data["indexed_gross_rating_point"].shift(lag)
        data[f"program_cat_before_lag_{lag}"] = data["program_cat_before"].shift(lag)
        data[f"program_cat_after_lag_{lag}"] = data["program_cat_after"].shift(lag)
        data[f"position_in_break_lag_{lag}"] = data["position_in_break"].shift(lag)
        data[f"flight_description_lag_{lag}"] = data["flight_description"].shift(lag)
        data[f"same_program_lag_{lag}"] = data["same_program"].shift(lag)
        data[f"tag_ons_lag_{lag}"] = data["tag_ons"].shift(lag)
        
    # List of columns to convert (including original and lagged variables)
    categorical_columns = [
    "hour", "day_of_week", "month",    "channel",
    "channel_lag_1", "channel_lag_2", "channel_lag_3", "channel_lag_4", "channel_lag_5",
    "program_cat_before", "program_cat_after", "position_in_break", "tag_ons",  
    "program_cat_before_lag_1", "program_cat_before_lag_2", "program_cat_before_lag_3", "program_cat_before_lag_4", "program_cat_before_lag_5",
    "program_cat_after_lag_1", "program_cat_after_lag_2", "program_cat_after_lag_3", "program_cat_after_lag_4", "program_cat_after_lag_5",
    "position_in_break_lag_1", "position_in_break_lag_2", "position_in_break_lag_3", "position_in_break_lag_4", "position_in_break_lag_5",
    "tag_ons_lag_1", "tag_ons_lag_2", "tag_ons_lag_3", "tag_ons_lag_4", "tag_ons_lag_5",  
    "flight_description", "flight_description_lag_1", "flight_description_lag_2", 
    "flight_description_lag_3", "flight_description_lag_4", "flight_description_lag_5",
    "same_program", "same_program_lag_1", "same_program_lag_2", "same_program_lag_3", "same_program_lag_4", "same_program_lag_5"
    ]

    # Apply the function to convert the columns to string
    convert_columns_to_str(data, categorical_columns)

    data = data[61:].reset_index(drop = True)
    data_cf = make_counterfactual(data)
    
    return data, data_cf

#----------------------------------------------------------------------------------------------------------------
#DEFINE DATASETS AND PARAMS
#--------------------------------------------------------------------------------------------------


# Load data
data = load_data()[0]
data_cf = load_data()[1]

#for web data
version = "visits_web_scaled"
lag = "visits_app_lag"

#for app data
# version = "visits_app_scaled"
# lag = "visits_web_lag"

def main(version,lag):

    # Encode categorical variables
    categorical_columns = [
    "hour", "day_of_week", "month",     "channel",
    "channel_lag_1", "channel_lag_2", "channel_lag_3", "channel_lag_4", "channel_lag_5",
    "program_cat_before", "program_cat_after", "position_in_break", "tag_ons", 
    "program_cat_before_lag_1", "program_cat_before_lag_2", "program_cat_before_lag_3", "program_cat_before_lag_4", "program_cat_before_lag_5",
    "program_cat_after_lag_1", "program_cat_after_lag_2", "program_cat_after_lag_3", "program_cat_after_lag_4", "program_cat_after_lag_5",
    "position_in_break_lag_1", "position_in_break_lag_2", "position_in_break_lag_3", "position_in_break_lag_4", "position_in_break_lag_5",
    "tag_ons_lag_1", "tag_ons_lag_2", "tag_ons_lag_3", "tag_ons_lag_4", "tag_ons_lag_5",  
    "flight_description", "flight_description_lag_1", "flight_description_lag_2", 
    "flight_description_lag_3", "flight_description_lag_4", "flight_description_lag_5",
    "same_program", "same_program_lag_1", "same_program_lag_2", "same_program_lag_3", "same_program_lag_4", "same_program_lag_5"
    ]
    
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    data_encoded_cf = make_counterfactual(data_encoded)
    
    # Define features and target
    features = [col for col in data_encoded.columns if col not in ['traffic', 'visits_app', 'visits_web', 'visits_app_scaled', 'visits_web_scaled', 'traffic_scaled','datetime', lag, 'traffic_lag']]
    target = version
    
    #Define sets for OOS evaluation
    train_data, test_set = train_test_split(data_encoded, test_size=0.2, shuffle=False)
    train_set, val_set = train_test_split(train_data,test_size=0.2,shuffle=False)
    test_set_cf = make_counterfactual(test_set)
    
    # Prepare OOS DMatrix for XGBoost
    dtrain = xgb.DMatrix(train_set[features], label=train_set[target])
    dval = xgb.DMatrix(val_set[features], label=val_set[target])
    dtest = xgb.DMatrix(test_set[features], label=test_set[target])
    dtest_cf = xgb.DMatrix(test_set_cf[features], label=test_set_cf[target])
    
    #Define sets for full IS fit
    full_train_data,full_val_data = train_test_split(data_encoded, test_size=0.1, shuffle=False)
    
    # Prepare IS DMatrix for XGBoost
    dtrain_full = xgb.DMatrix(full_train_data[features], label=full_train_data[target])
    dval_full = xgb.DMatrix(full_val_data[features], label=full_val_data[target])
    dfull = xgb.DMatrix(data_encoded[features], label=data_encoded[target])
    dfull_cf = xgb.DMatrix(data_encoded_cf[features], label=data_encoded_cf[target])
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 200
    }
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #OOS evaluation
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Train XGBoost model
    model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dval, 'validation')], early_stopping_rounds=10)
    
    # Make predictions
    predictions = model.predict(dtest)
    predictions_cf = model.predict(dtest_cf)
    
    # Evaluate model
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(test_set[target], predictions)),
        "MAE": mean_absolute_error(test_set[target], predictions),
        "R² Score": r2_score(test_set[target], predictions)
    }
    
    # Print evaluation metrics
    print(pd.DataFrame([metrics]))
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test_set['datetime'], test_set[target], label='Actual')
    plt.plot(test_set['datetime'], predictions, label='Predicted', linestyle='dashed')
    plt.plot(test_set['datetime'], predictions_cf, label='Counterfactual', linestyle='dashed')
    plt.xlabel('Datetime')
    plt.ylabel('Traffic')
    plt.legend()
    plt.title('XGBoost Traffic Prediction')
    plt.xlim(pd.Timestamp("2023-12-22 18:00:00"),pd.Timestamp("2023-12-22 22:00:00"))
    plt.show()
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #IS fit 
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model = xgb.train(params, dtrain_full, num_boost_round=200, evals=[(dval_full, 'validation')], early_stopping_rounds=10)
    
    # Make predictions
    predictions = model.predict(dfull)
    predictions_cf = model.predict(dfull_cf)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data_encoded['datetime'], data_encoded[target], label='Actual')
    plt.plot(data_encoded['datetime'], predictions, label=r'$\hat{y}_t$', linestyle='dashed')
    plt.plot(data_encoded['datetime'], predictions_cf, label=r'$\hat{y}^*_t$', linestyle='dashed')
    # Increase font sizes
    plt.xlabel('Datetime', fontsize=14)  # Increase font size of x-axis label
    plt.ylabel('Traffic', fontsize=14)   # Increase font size of y-axis label
    plt.title('XGBoost Traffic Prediction', fontsize=16)  # Increase font size of title
    plt.legend(fontsize=20) 
    plt.xlim(pd.Timestamp("2023-11-21 20:00:00"),pd.Timestamp("2023-11-21 22:00:00"))
    plt.show()
    
    df = pd.DataFrame({'Datetime': data["datetime"], 'Actual Prediction': predictions, 'CF Prediction': predictions_cf})
    
    # Save to CSV
    df.to_csv(f'PeakAnalysis_{version}.csv', index=False)
    
    
main("visits_web_scaled", "visits_app_lag")
#main("visits_app_scaled", "visits_web_lag")
