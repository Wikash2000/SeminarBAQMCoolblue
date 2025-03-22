# Coolblue TV Commercial Optimization  

This repository contains the code used in the Coolblue seminar on TV commercial optimization, conducted on behalf of Erasmus University Rotterdam for the Master's seminar of the Business Analytics & Quantitative Marketing master track. (March 18, 2025)

## **Authors**  
- **Otte Beljaars** – 737354  
- **Wikash Chitoe** – 548239  
- **Nicholas Sassen** – 531725  

## **Data Disclaimer**  
Due to privacy considerations, Coolblue's data is not included in this repository. The files **"Web + broadcasting data - Broadcasting data.csv"** and **"web_data_with_product_types"** must be obtained from a separate source.  

## **Repository Structure**  
Below, we outline the purpose of each file in this repository and provide the correct sequence for running them.  

- ## *OutlierRemoval.py*  
  This file removes outliers present in the **web_data_with_product_types** file.  

- ## *PlotsToShowCleaning.py*  
  This file generates plots comparing the data before and after cleaning, highlighting the differences.  

- ## *initialAnalysisWebData.py*  
  This file provides visualizations of the web data, which are also included in the report. 

- ## *initialAnalysisCommercialData*  
  Plots histograms describing the distribution and daily volume of commercial data.
 
- ## *STA_Analysis.py*  
  This file performs seasonal-trend decomposition on web and app visit data to uncover trends and patterns. It also visualizes the distribution of TV commercials across specific time intervals.

- ## *DoubleAdDetection.py*  
  Detects ads that were aired at the same time and flags them accordingly.  

- ## *Peaktocomm_sep.py*  
  Calculates the **uplift in web or app traffic** by comparing actual traffic to a counterfactual prediction. It attributes the uplift to specific commercials based on their viewership.  

- ## *XGB_sep.py*  
  Implements an XGBoost model to predict web traffic following TV commercials. (one of the two methods that is tested)

- ## *tft_sep.py*  
  Implements a Temporal Fusion Transformer (TFT) model for forecasting of web traffic after TV commercials. (second method that is tested)

- ## *Attribution.py*  
  Uses XGBoost to analyze the key characteristics of TV commercials that impact peak website visits. It also generates **Partial Dependence Plots** and **Feature Importance** visualizations. (works on both the XGB_sep.py and tft_sep.py files as different forecast models)  
