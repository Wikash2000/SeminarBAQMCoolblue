# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:15:22 2025

@author: nicho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ProgressBar
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.metrics import RMSE




#------------------------------------------------------------------------------------------------------------------------
#FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------
# Define a custom progress bar callback
class CustomProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        elapsed_time = time.time() - self.start_time
        print(f"Training finished! Total time: {elapsed_time:.2f} seconds")
        
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
  
     cont_columns_to_set_zero = ["indexed_gross_rating_point",
    "indexed_gross_rating_point_lag_1", "indexed_gross_rating_point_lag_2", "indexed_gross_rating_point_lag_3", "indexed_gross_rating_point_lag_4", "indexed_gross_rating_point_lag_5"]

     dataset_cf = dataset.copy(deep=True)
     dataset_cf[cat_columns_to_set_zero] = "0"
     dataset_cf[cont_columns_to_set_zero] = 0
     return dataset_cf

def load_data():
   Website = pd.read_csv("C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/web_data_cleaned_full.csv")
   # Ensure 'datetime' column in Websites in datetime format
   Website['datetime'] = pd.to_datetime(Website['datetime'], errors='coerce')
   #Add a column for total traffic
   Website['traffic'] = Website['visits_web'] + Website['visits_app']
   
   
   Commercial = pd.read_csv("C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/Commercial_cleaned_full.csv")
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
   # Create an integer index for the time series
   data["time_idx"] = range(len(data))
   # Set group ID to 0
   data["group_id"] = 0
       
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
#DEFINE DATASETS
#--------------------------------------------------------------------------------------------------

#load data
data = load_data()[0]
data_cf = load_data()[1]
    
# Split data
train_data, test_set = train_test_split(data, test_size=0.2, shuffle=False)
train_set, val_set = train_test_split(train_data,test_size=0.2,shuffle=False)
test_set_cf = make_counterfactual(test_set)

max_encoder_length = 10  # Use past 10mins for encoding
max_prediction_length = 1  # Predict one step ahead

#for web traffic
version = "visits_web_scaled"
lag = "visits_web_lag"

# #for app traffic
# version = "visits_app_scaled"
# lag = "visits_app_lag"

def main(version,lag):

    true_categoricals = [
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
  
  
    true_continuous = [
        "hour_sin", "hour_cos", "day_sin", "day_cos", "indexed_gross_rating_point",
        "indexed_gross_rating_point_lag_1", "indexed_gross_rating_point_lag_2", "indexed_gross_rating_point_lag_3", "indexed_gross_rating_point_lag_4", "indexed_gross_rating_point_lag_5", lag
    ]
  
    
    # Define the TimeSeriesDataSet for the training data
    train_dataset = TimeSeriesDataSet(
        train_set,
        time_idx="time_idx",
        target=version,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],  # No static categorical features in this case
        time_varying_known_reals=true_continuous,
        time_varying_known_categoricals = true_categoricals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
    )
    
    test_dataset = TimeSeriesDataSet(
        test_set,
        time_idx="time_idx",
        target=version,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_reals=true_continuous,
        time_varying_known_categoricals = true_categoricals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
    )
    
    val_dataset = TimeSeriesDataSet(
        val_set,
        time_idx="time_idx",
        target=version,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_reals=true_continuous,
        time_varying_known_categoricals = true_categoricals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
    )
    
    
    # Define the TimeSeriesDataSet for the test (counterfactual) data
    test_dataset_cf = TimeSeriesDataSet(
        test_set_cf,
        time_idx="time_idx",
        target=version,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=true_continuous,
        time_varying_known_categoricals = true_categoricals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
    )
    
    
    #------------------------------------------------------------------------------------------------------------------------------
    #MODEL SETUP
    #------------------------------------------------------------------------------------------------------------------------
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0, shuffle=False)
    test_dataloader_cf = test_dataset_cf.to_dataloader(train=False, batch_size=64, num_workers=0, shuffle=False)
    test_val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0, shuffle=False)
    
    # Create the progress bar callback
    progress_bar = CustomProgressBar()
    
    
    #------------------------------------------------------------------------------------------------------------------------------------------
    #OOS-Training+Predictions
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # TFT
    # ----------------------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    
    # Define the TFT model
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.03,
        hidden_size=16,  # Model size
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=1,  # Point forecast
        loss=RMSE(),
        log_interval=10,
    )
    
    # Update trainer
    trainer = pl.Trainer(
        max_epochs=30,
        callbacks=[progress_bar, early_stop_callback],
        val_check_interval=1.0,  # Evaluate on validation set every epoch
    )
    trainer.fit(tft, train_dataloader,test_val_dataloader)
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)    
    
    # Make predictions with commercials
    validation_predictions= tft.predict(test_val_dataloader).numpy()
    test_predictions = tft.predict(test_dataloader).numpy()
    cf_predictions = tft.predict(test_dataloader_cf).numpy()
    in_sample_fit =  tft.predict(train_dataloader).numpy()
    
    #Naive benchmark
    #--------------------------------------------- 
    naive_predictions = test_set[version][9:-1]
    naive_MA_predictions = test_set[lag][10:]
    
    #Evaluation
    #---------------------------------------------
    eval_metrics_TFT = get_evaluations(test_set[version][10:], test_predictions) 
    eval_metrics_naive = get_evaluations(test_set[version][10:],naive_predictions)
    eval_metrics_naiveMA = get_evaluations(test_set[version][10:],naive_MA_predictions)
    
    print(eval_metrics_TFT)
    print(eval_metrics_naive)
    print(eval_metrics_naiveMA)
  
    #--------------------------------------------------------------------------------------------------------
    #PLOTTING - OOS 
    #--------------------------------------------------------------------------------------------------------
    
    #Prepare data for plotting
    data["test_predictions"] = np.nan
    data["counterfactual_predictions"] = np.nan
    data["validation_predictions"] = np.nan
    data["in_sample_fit"] = np.nan
    
    # Shift prediction indices by 100
    shifted_indices_train = train_set.index[10:10 + len(train_dataset)]  
    shifted_indices_test = test_set.index[10:10 + len(test_predictions)]  # Adjust for encoder length
    shifted_indices_val = val_set.index[10:10 + len(validation_predictions)] 
    
    data.loc[shifted_indices_train, "in_sample_fit"] = in_sample_fit
    data.loc[shifted_indices_test, "actual_predictions"] = test_predictions
    data.loc[shifted_indices_test, "counterfactual_predictions"] = cf_predictions
    data.loc[shifted_indices_val, "validation_predictions"] = validation_predictions
    
    # Add a column indicating the dataset split
    data["dataset_split"] = "train"
    data.loc[val_set.index, "dataset_split"] = "validation"
    data.loc[test_set.index, "dataset_split"] = "test"
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Plot true traffic
    plt.plot(data["datetime"], data[version], label="True Traffic", color="blue", alpha=0.7)
    
    # Plot actual predictions
    plt.plot(data["datetime"], data["actual_predictions"], label="Actual Predictions", color="green")
    
    plt.plot(data["datetime"], data["in_sample_fit"], label="In-sample fit", color="cyan", linestyle="--")
    
    # Plot counterfactual predictions
    plt.plot(data["datetime"], data["counterfactual_predictions"], label="Counterfactual Predictions", color="red")
    
    plt.plot(data["datetime"], data["validation_predictions"], label = "Validation Predictions", color = "orange", linestyle = ":")
    
    # Highlight the dataset splits
    plt.axvspan(data.loc[train_set.index[0], "datetime"], data.loc[train_set.index[-1], "datetime"],
                color="lightblue", alpha=0.2, label="Train Set")
    plt.axvspan(data.loc[val_set.index[0], "datetime"], data.loc[val_set.index[-1], "datetime"],
                color="lightgreen", alpha=0.2, label="Validation Set")
    plt.axvspan(data.loc[test_set.index[0], "datetime"], data.loc[test_set.index[-1], "datetime"],
                color="lightcoral", alpha=0.2, label="Test Set")
    
    # Set zoom range
    plt.xlim(pd.Timestamp("2023-12-22 18:00:00"),pd.Timestamp("2023-12-22 22:00:00"))
    # Formatting
    #plt.title("Traffic Predictions and Counterfactual Analysis", fontsize=16)
    plt.xlabel("Datetime", fontsize=14)
    plt.ylabel("Traffic", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # #---------------------------------------------------------------------------------------------------------------------------
    # #NAIVE PLOT
    # #-------------------------------------------------------------------------------------------------------------------------
    # Plot actual predictions
    plt.figure(figsize=(15, 8))
    plt.plot(test_set["datetime"][10:], test_set[version][10:], label="true", color="blue")
    
    #plt.plot(test_set["datetime"][10:], naive_predictions, label="naive pred", color="orange")
    #plt.plot(test_set["datetime"][10:], naive_MA_predictions, label="naive pred MA", color="orange")
    plt.plot(test_set["datetime"][10:], test_predictions, label="tft pred", color="orange")
    # plt.plot(test_set["datetime"][10:], cf_predictions, label="cf pred", color="orange)
    # Set zoom range
    plt.xlim(pd.Timestamp("2023-12-22 18:00:00"),pd.Timestamp("2023-12-22 22:00:00"))
    plt.xlabel("Datetime", fontsize=14)
    plt.ylabel("Traffic", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


main("visits_web_scaled", "visits_web_lag")
#main("visits_app_scaled", "visits_app_lag")