# -*- coding: utf-8 -*-
"""
This file loads the cleaned traffic and commercial data and performs an extenisve peak analysis.

@author: 531725ns
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.metrics import SMAPE
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ProgressBar
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from lightning.pytorch.callbacks import EarlyStopping



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
        
def decompose_traffic_series(data):
    ts = data.set_index('datetime')["traffic"]
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=1440, extrapolate_trend='freq')  # You can adjust the period based on your data
    data['trend'] = decomposition.trend.values
    return data

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
    "channel", "commercial", "commercial_lag_1", "commercial_lag_2", "commercial_lag_3", "commercial_lag_4", "commercial_lag_5",
    "channel_lag_1", "channel_lag_2", "channel_lag_3", "channel_lag_4", "channel_lag_5",
    "program_cat_before", "program_cat_after", "position_in_break", "spotlength",
    "program_cat_before_lag_1", "program_cat_before_lag_2", "program_cat_before_lag_3", "program_cat_before_lag_4", "program_cat_before_lag_5",
    "program_cat_after_lag_1", "program_cat_after_lag_2", "program_cat_after_lag_3", "program_cat_after_lag_4", "program_cat_after_lag_5",
    "position_in_break_lag_1", "position_in_break_lag_2", "position_in_break_lag_3", "position_in_break_lag_4", "position_in_break_lag_5",
    "spotlength_lag_1", "spotlength_lag_2", "spotlength_lag_3", "spotlength_lag_4", "spotlength_lag_5"
    ]
    
    cont_columns_to_set_zero = ["indexed_gross_rating_point",
    "indexed_gross_rating_point_lag_1", "indexed_gross_rating_point_lag_2", "indexed_gross_rating_point_lag_3", "indexed_gross_rating_point_lag_4", "indexed_gross_rating_point_lag_5"]

    dataset_cf = dataset.copy(deep=True)
    dataset_cf[cat_columns_to_set_zero] = "0"
    dataset_cf[cont_columns_to_set_zero] = 0
    return dataset_cf

def load_data(version):
    Website = pd.read_csv("C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/web_data_cleaned_full.csv")
    # Ensure 'datetime' column in Websites in datetime format
    Website['datetime'] = pd.to_datetime(Website['datetime'], errors='coerce')
    #Add a column for total traffic
    Website['traffic'] = Website['visits_web'] + Website['visits_app']
    
    
    Commercial = pd.read_csv("C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/Commercial_cleaned_full.csv")
    # Combine 'Date' and 'Time' into a single datetime column
    Commercial['datetime'] = pd.to_datetime(Commercial['date'] + ' ' + Commercial['time'],
                                            format='%m/%d/%Y %I:%M:%S %p') 
    
    # Merge the two dataframes (to get all commercial variables)
    if (version == 1):
        Website1 = Website.copy()
        Website1['commercial'] = Website1['datetime'].isin(Commercial['datetime']).astype(int)
        Commercial_max = Commercial.loc[Commercial.groupby('datetime')['indexed_gross_rating_point'].idxmax()]
        # Now merge the datasets
        Merged_data = pd.merge(Website1, Commercial_max, on='datetime', how='left')
        # Select only the columns you want to keep
        Merged_data = Merged_data[['datetime', 'traffic', 'commercial', 'indexed_gross_rating_point','channel','position_in_break', 'program_cat_before', 'program_cat_after', 'spotlength']]    
        #fill na with 0 
        Merged_data = Merged_data.assign(
        channel=Merged_data['channel'].fillna("0"),
        indexed_gross_rating_point=Merged_data['indexed_gross_rating_point'].fillna(0),
        position_in_break=Merged_data['position_in_break'].fillna("0"),
        program_cat_before=Merged_data['program_cat_before'].fillna("0"),
        program_cat_after=Merged_data['program_cat_after'].fillna("0"),
        spotlength=Merged_data['spotlength'].fillna("0")
        )  

        # filter period without commercial data
        Merged_data = Merged_data[Merged_data['datetime'] >= pd.Timestamp("2023-09-11")]
        data = Merged_data.copy()
    if (version == 2):
        Website2 = Website.groupby(['datetime'])['visits_web'].sum().reset_index() #web visits only
        Website2['commercial'] = Website2['datetime'].isin(Commercial['datetime']).astype(int)
        Merged_data = pd.merge(Website2, Commercial, on='datetime', how='left')
        Merged_data = Merged_data[Merged_data['datetime'] >= '2023/9/11'] # filter 
        data = Website2
    if (version ==3):
        Website3 = Website.groupby(['datetime'])['visits_app'].sum().reset_index() # app visits only
        Website3['commercial'] = Website3['datetime'].isin(Commercial['datetime']).astype(int)
        Merged_data = pd.merge(Website3, Commercial, on='datetime', how='left')
        Merged_data = Merged_data[Merged_data['datetime'] >= '2023/9/11'] # filter 
        data = Website3
    
    # Extract time-based features
    data["hour"] = data["datetime"].dt.hour
    data["minute_of_day"] = data["datetime"].dt.hour * 60 + data["datetime"].dt.minute
    data["day_of_week"] = data["datetime"].dt.dayofweek
    
    # Periodic time encoding for hour and day (sin/cos for circular encoding)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["minute_sin"] = np.sin(2 * np.pi * data["minute_of_day"] / 1440)
    data["minute_cos"] = np.cos(2 * np.pi * data["minute_of_day"] / 1440)
    data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    
    # Ensure data is sorted by datetime
    data = data.sort_values("datetime")
    # Create an integer index for the time series
    data["time_idx"] = range(len(data))
    # Set group ID to 0
    data["group_id"] = 0

    for lag in range(1, 6):  # Now generating 5 lags (1 to 5)
        data[f"commercial_lag_{lag}"] = data["commercial"].shift(lag)
        data[f"channel_lag_{lag}"] = data["channel"].shift(lag)
        data[f"indexed_gross_rating_point_lag_{lag}"] = data["indexed_gross_rating_point"].shift(lag)
        data[f"program_cat_before_lag_{lag}"] = data["program_cat_before"].shift(lag)
        data[f"program_cat_after_lag_{lag}"] = data["program_cat_after"].shift(lag)
        data[f"position_in_break_lag_{lag}"] = data["position_in_break"].shift(lag)
        data[f"spotlength_lag_{lag}"] = data["spotlength"].shift(lag)

    data["traffic_lag"] = data['traffic'].rolling(window=60,min_periods=1).mean().shift(1)

    # List of columns to convert (including original and lagged variables)
    categorical_columns = [
        "hour", "day_of_week", "minute_of_day", "channel", "commercial","commercial_lag_1", "commercial_lag_2", "commercial_lag_3", "commercial_lag_4", "commercial_lag_5",
        "channel_lag_1", "channel_lag_2", "channel_lag_3", "channel_lag_4", "channel_lag_5",
        "program_cat_before", "program_cat_after", "position_in_break", "spotlength",
        "program_cat_before_lag_1", "program_cat_before_lag_2", "program_cat_before_lag_3", "program_cat_before_lag_4", "program_cat_before_lag_5",
        "program_cat_after_lag_1", "program_cat_after_lag_2", "program_cat_after_lag_3", "program_cat_after_lag_4", "program_cat_after_lag_5",
        "position_in_break_lag_1", "position_in_break_lag_2", "position_in_break_lag_3", "position_in_break_lag_4", "position_in_break_lag_5",
        "spotlength_lag_1", "spotlength_lag_2", "spotlength_lag_3", "spotlength_lag_4", "spotlength_lag_5"
    ]

    # Apply the function to convert the columns to string
    convert_columns_to_str(data, categorical_columns)

    data = data[61:].reset_index(drop = True)
    
    data_cf = make_counterfactual(data)
    
    # # Make counterfactual predictions (commercials = 0)
    # data["commercial_counterfactual"] = "0"
    # data["counterfactual_channel"] = "0"
    # data["program_cat_before_counterfactual"] = "0"
    # data["program_cat_after_counterfactual"] = "0"
    # data["position_in_break_counterfactual"] = "0"
    # data["spotlength_counterfactual"] = "0"
    # data["counterfactual_indexed_gross_rating_point"] = 0

    # for lag in range(1, 6):  # Now generating 5 lags (1 to 5)
    #     data[f"commercial_counterfactual_lag_{lag}"] = "0"
    #     data[f"counterfactual_channel_lag_{lag}"] = "0"
    #     data[f"program_cat_before_counterfactual_lag_{lag}"] = "0"
    #     data[f"program_cat_after_counterfactual_lag_{lag}"] = "0"
    #     data[f"position_in_break_counterfactual_lag_{lag}"] = "0"
    #     data[f"spotlength_counterfactual_lag_{lag}"] = "0"
    #     data[f"counterfactual_indexed_gross_rating_point_lag_{lag}"] = 0 
        
    return data, data_cf



#----------------------------------------------------------------------------------------------------------------
#DEFINE DATASETS
#--------------------------------------------------------------------------------------------------

#load data
data = load_data(1)[0]
data_cf = load_data(1)[1]
scaler = MinMaxScaler(feature_range=(10, 20))
data["traffic_scaled"] = scaler.fit_transform(data[["traffic"]])
data_cf["traffic_scaled"] = scaler.fit_transform(data_cf[["traffic"]])
    
# Split data
train_data, test_set = train_test_split(data, test_size=0.2, shuffle=False)
train_set, val_set = train_test_split(train_data,test_size=0.2,shuffle=False)
test_set_cf = make_counterfactual(test_set)

#HIERONDER TEST SIZE AANPASSEN voor Otte
full_train_data,full_val_data = train_test_split(data, test_size=0.1, shuffle=False)
# Define the TimeSeriesDataSet
max_encoder_length = 10  # Use past 10mins for encoding
max_prediction_length = 1  # Predict one step ahead
true_categoricals = [
    "hour", "day_of_week", "commercial", "minute_of_day", 
    "commercial_lag_1", "commercial_lag_2", "commercial_lag_3", "commercial_lag_4", "commercial_lag_5", 
    "channel_lag_1", "channel_lag_2", "channel_lag_3", "channel_lag_4", "channel_lag_5", 
    "program_cat_before", "program_cat_after", "position_in_break", "spotlength", 
    "program_cat_before_lag_1", "program_cat_before_lag_2", "program_cat_before_lag_3", "program_cat_before_lag_4", "program_cat_before_lag_5", 
    "program_cat_after_lag_1", "program_cat_after_lag_2", "program_cat_after_lag_3", "program_cat_after_lag_4", "program_cat_after_lag_5", 
    "position_in_break_lag_1", "position_in_break_lag_2", "position_in_break_lag_3", "position_in_break_lag_4", "position_in_break_lag_5", 
    "spotlength_lag_1", "spotlength_lag_2", "spotlength_lag_3", "spotlength_lag_4", "spotlength_lag_5"
]

# cf_categoricals = [
#     "hour", "day_of_week", "commercial_counterfactual", "minute_of_day", 
#     "commercial_counterfactual_lag_1", "commercial_counterfactual_lag_2", "commercial_counterfactual_lag_3", "commercial_counterfactual_lag_4", "commercial_counterfactual_lag_5", 
#     "counterfactual_channel_lag_1", "counterfactual_channel_lag_2", "counterfactual_channel_lag_3", "counterfactual_channel_lag_4", "counterfactual_channel_lag_5",  
#     "counterfactual_channel", "program_cat_before_counterfactual", "program_cat_after_counterfactual", "position_in_break_counterfactual", "spotlength_counterfactual", 
#     "program_cat_before_counterfactual_lag_1", "program_cat_before_counterfactual_lag_2", "program_cat_before_counterfactual_lag_3", "program_cat_before_counterfactual_lag_4", "program_cat_before_counterfactual_lag_5", 
#     "program_cat_after_counterfactual_lag_1", "program_cat_after_counterfactual_lag_2", "program_cat_after_counterfactual_lag_3", "program_cat_after_counterfactual_lag_4", "program_cat_after_counterfactual_lag_5", 
#     "position_in_break_counterfactual_lag_1", "position_in_break_counterfactual_lag_2", "position_in_break_counterfactual_lag_3", "position_in_break_counterfactual_lag_4", "position_in_break_counterfactual_lag_5", 
#     "spotlength_counterfactual_lag_1", "spotlength_counterfactual_lag_2", "spotlength_counterfactual_lag_3", "spotlength_counterfactual_lag_4", "spotlength_counterfactual_lag_5"
# ]

true_continuous = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "minute_sin", "minute_cos", "indexed_gross_rating_point",
    "indexed_gross_rating_point_lag_1", "indexed_gross_rating_point_lag_2", "indexed_gross_rating_point_lag_3", "indexed_gross_rating_point_lag_4", "indexed_gross_rating_point_lag_5", "traffic_lag"
]

# cf_continuous = [
#     "hour_sin", "hour_cos", "day_sin", "day_cos", "minute_sin", "minute_cos", "counterfactual_indexed_gross_rating_point", 
#     "counterfactual_indexed_gross_rating_point_lag_1", "counterfactual_indexed_gross_rating_point_lag_2", 
#     "counterfactual_indexed_gross_rating_point_lag_3", "counterfactual_indexed_gross_rating_point_lag_4", "counterfactual_indexed_gross_rating_point_lag_5", "traffic_lag"
# ]

full_dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="traffic_scaled",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # No static categorical features in this case
    time_varying_known_reals=true_continuous,
    time_varying_known_categoricals = true_categoricals,
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
)


full_train_dataset = TimeSeriesDataSet(
    full_train_data,
    time_idx="time_idx",
    target="traffic_scaled",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # No static categorical features in this case
    time_varying_known_reals=true_continuous,
    time_varying_known_categoricals = true_categoricals,
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
)

full_val_dataset = TimeSeriesDataSet(
    full_val_data,
    time_idx="time_idx",
    target="traffic_scaled",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # No static categorical features in this case
    time_varying_known_reals=true_continuous,
    time_varying_known_categoricals = true_categoricals,
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
)


# Define the TimeSeriesDataSet for the training data
train_dataset = TimeSeriesDataSet(
    train_set,
    time_idx="time_idx",
    target="traffic_scaled",
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
    target="traffic_scaled",
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
    target="traffic_scaled",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    time_varying_known_reals=true_continuous,
    time_varying_known_categoricals = true_categoricals,
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
)

full_dataset_cf = TimeSeriesDataSet(
    data_cf,
    time_idx="time_idx",
    target="traffic_scaled",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=true_continuous ,
    time_varying_known_categoricals = true_categoricals,
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
)


# Define the TimeSeriesDataSet for the test (counterfactual) data
test_dataset_cf = TimeSeriesDataSet(
    test_set_cf,
    time_idx="time_idx",
    target="traffic_scaled",
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
# Create dataloaders
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
test_dataloader = test_dataset.to_dataloader(train = False, batch_size = 64, num_workers=0)
test_dataloader_cf = test_dataset_cf.to_dataloader(train = False, batch_size = 64, num_workers=0)
test_val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)


full_train_dataloader = full_train_dataset.to_dataloader(train=True, batch_size=64, num_workers = 0)
full_val_dataloader = full_val_dataset.to_dataloader(train=False, batch_size=64, num_workers = 0)

full_dataloader = full_dataset.to_dataloader(train=False, batch_size=64, num_workers = 0)
full_dataloader_cf = full_dataset_cf.to_dataloader(train=False, batch_size=64, num_workers = 0)


# Create the progress bar callback
progress_bar = CustomProgressBar()
#------------------------------------------------------------------------------------------------------------------------------------------
#OOS-Training+Predictions
#-------------------------------------------------------------------------------------------------------------------------------------------

#TFT
#----------------------------------------
# early_stop_callback = EarlyStopping(
#     monitor="val_loss", patience=3, verbose=True, mode="min"
# )


# # Define the TFT model
# tft = TemporalFusionTransformer.from_dataset(
#     train_dataset,
#     learning_rate=0.03,
#     hidden_size=16,  # Model size
#     attention_head_size=4,
#     dropout=0.1,
#     hidden_continuous_size=8,
#     output_size=7,  # Quantiles to predict
#     loss=QuantileLoss(),
#     log_interval=10,
# )

# # Update trainer
# trainer = pl.Trainer(
#     max_epochs=30,
#     callbacks=[progress_bar, early_stop_callback],
#     val_check_interval=1.0,  # Evaluate on validation set every epoch
# )
# trainer.fit(tft, train_dataloader,test_val_dataloader)

# # Make predictions with commercials
# validation_predictions = scaler.inverse_transform(tft.predict(test_val_dataloader))
# test_predictions = scaler.inverse_transform(tft.predict(test_dataloader))
# cf_predictions = scaler.inverse_transform(tft.predict(test_dataloader_cf))
# in_sample_fit =  scaler.inverse_transform(tft.predict(train_dataloader))

# #Naive benchmark
# #--------------------------------------------- 
# naive_predictions = test_set['traffic'][9:-1]

# #Evaluation
# #---------------------------------------------
# eval_metrics_TFT = get_evaluations(test_set['traffic'][10:], test_predictions) 
# eval_metrics_naive = get_evaluations(test_set['traffic'][10:],test_set['traffic'][9:-1])

# full_set_fit = scaler.inverse_transform(tft.predict(full_dataset))
# cf_full_set_fit = scaler.inverse_transform(tft.predict(full_dataset_cf))


#----------------------------------------------------------------------------------------------------------------------------------------
#IS-Training+Predictions
#---------------------------------------------------------------------------------------------------------------------------------------
early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=3, min_delta=0.001,verbose=True, mode="min"
)


# Define the TFT model
tft = TemporalFusionTransformer.from_dataset(
    full_train_dataset,
    learning_rate=0.03,
    hidden_size=16,  # Model size
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # Quantiles to predict
    loss=QuantileLoss(),
    log_interval=10,
)

# Update trainer
trainer = pl.Trainer(
    max_epochs=30,
    callbacks=[progress_bar, early_stop_callback],
    val_check_interval=1.0, 
    
)

trainer.fit(tft, full_train_dataloader, full_val_dataloader)

tft =TemporalFusionTransformer.load_from_checkpoint(r"c:\users\nicho\onedrive - erasmus university rotterdam\master\seminar\lightning_logs\version_203\checkpoints\epoch=5-step=13602.ckpt")


full_set_fit = scaler.inverse_transform(tft.predict(full_dataset))
cf_full_set_fit = scaler.inverse_transform(tft.predict(full_dataset_cf))

best_model_path = trainer.checkpoint_callback.best_model_path

# for col in true_categoricals:
#     full_categories = set(data[col].unique())
#     train_categories = set(full_train_data[col].unique())
    
#     print(f"Column: {col}")
#     print(f"Categories in full dataset but not in training: {full_categories - train_categories}")
#     print(f"Categories in training but not in full dataset: {train_categories - full_categories}")
#     print("-" * 50)
   



#---------------------------------------------------------------------------------------------------------------------------------------
#PLOTTING - IS
#--------------------------------------------------------------------------------------------------------------------------------------

shifted_indices_full = data.index[10:10 + len(full_set_fit)] 
data.loc[shifted_indices_full, "full_in_sample_fit"] = full_set_fit
data.loc[shifted_indices_full, "cf_in_sample_fit"] = cf_full_set_fit

plt.figure(figsize=(15, 8))
plt.plot(data["datetime"],data["full_in_sample_fit"], color = "orange",label="Actual Predictions",linestyle = ":")
plt.plot(data["datetime"],data["cf_in_sample_fit"], color = "red",label="CF Predictions", linestyle = ":")
plt.plot(data["datetime"],data["traffic"], color = "blue", label="Actual Traffic",linestyle = ":")

plt.xlabel("Datetime", fontsize=14)
plt.ylabel("Traffic", fontsize=14)
plt.legend(fontsize=12)

plt.xlim(pd.Timestamp("2023-11-22 18:00:00"),pd.Timestamp("2023-11-22 23:00:00"))

df = pd.DataFrame({'Datetime': data["datetime"][10:],'Actual Prediction': full_set_fit[:,0], 'CF Prediction': cf_full_set_fit[:,0]})

# Save to CSV
df.to_csv('C:/Users/nicho/OneDrive - Erasmus University Rotterdam/Master/Seminar/PeakAnalysis.csv', index=False)

#--------------------------------------------------------------------------------------------------------
#PLOTTING - OOS 
#--------------------------------------------------------------------------------------------------------

# Prepare data for plotting
# data["test_predictions"] = np.nan
# data["counterfactual_predictions"] = np.nan
# data["validation_predictions"] = np.nan
# data["in_sample_fit"] = np.nan

# # Shift prediction indices by 100
# shifted_indices_train = train_set.index[10:10 + len(train_dataset)]  
# shifted_indices_test = test_set.index[10:10 + len(test_predictions)]  # Adjust for encoder length
# shifted_indices_val = val_set.index[10:10 + len(validation_predictions)] 

# data.loc[shifted_indices_train, "in_sample_fit"] = in_sample_fit
# data.loc[shifted_indices_test, "actual_predictions"] = test_predictions
# data.loc[shifted_indices_test, "counterfactual_predictions"] = cf_predictions
# data.loc[shifted_indices_val, "validation_predictions"] = validation_predictions

# # Add a column indicating the dataset split
# data["dataset_split"] = "train"
# data.loc[val_set.index, "dataset_split"] = "validation"
# data.loc[test_set.index, "dataset_split"] = "test"

# # Plotting
# plt.figure(figsize=(15, 8))

# # Plot true traffic
# plt.plot(data["datetime"], data["traffic"], label="True Traffic", color="blue", alpha=0.7)

# # Plot actual predictions
# plt.plot(data["datetime"], data["actual_predictions"], label="Actual Predictions", color="green", linestyle="--")

# plt.plot(data["datetime"], data["in_sample_fit"], label="In-sample fit", color="cyan", linestyle="--")

# # Plot counterfactual predictions
# plt.plot(data["datetime"], data["counterfactual_predictions"], label="Counterfactual Predictions", color="red", linestyle=":")

# plt.plot(data["datetime"], data["validation_predictions"], label = "Validation Predictions", color = "orange", linestyle = ":")

# # Highlight the dataset splits
# plt.axvspan(data.loc[train_set.index[0], "datetime"], data.loc[train_set.index[-1], "datetime"],
#             color="lightblue", alpha=0.2, label="Train Set")
# plt.axvspan(data.loc[val_set.index[0], "datetime"], data.loc[val_set.index[-1], "datetime"],
#             color="lightgreen", alpha=0.2, label="Validation Set")
# plt.axvspan(data.loc[test_set.index[0], "datetime"], data.loc[test_set.index[-1], "datetime"],
#             color="lightcoral", alpha=0.2, label="Test Set")

# # Set zoom range
# plt.xlim(pd.Timestamp("2023-12-15 18:00:00"),pd.Timestamp("2023-12-15 22:00:00"))
# # Formatting
# #plt.title("Traffic Predictions and Counterfactual Analysis", fontsize=16)
# plt.xlabel("Datetime", fontsize=14)
# plt.ylabel("Traffic", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# #---------------------------------------------------------------------------------------------------------------------------
# #NAIVE PLOT
# #-------------------------------------------------------------------------------------------------------------------------
# # Plot actual predictions
# plt.figure(figsize=(15, 8))
# plt.plot(test_set["datetime"][10:], test_set['traffic'][10:], label="true", color="green", linestyle="--")

# plt.plot(test_set["datetime"][10:], test_set["traffic"][9:-1], label="naive pred", color="cyan", linestyle="--")
# # Set zoom range
# #plt.xlim(pd.Timestamp("2023-12-20 18:00:00"),pd.Timestamp("2023-12-20 22:00:00"))
# plt.xlabel("Datetime", fontsize=14)
# plt.ylabel("Traffic", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

