# Predictive-Analytics-for-AGVs : Battery Failure Prediction Using Machine Learning Techniques
This repository contains the code from one of my projects on Automated Guided Vehicles. It involved developing a sytem to predict potential battery failures based on SOC (State of Charge) drop rates and predicts with a 78% accuracy for unseen data, premptively alerting stakeholders of potential battery failures.

# Author: Oluwatosin Adewale

## Project Overview
Early detection of battery failures can significantly enhance operational efficiency and safety, particularly in the automotive, energy storage, and consumer electronics sectors.The project aims to use SOC (State of Charge) data to predict the likelihood of battery failures, thereby reducing downtime and maintenance costs. This README will provide a thorough walkthrough of the project's structure, methodologies employed, key findings, and insights derived from the analysis.

### Objective: 
Develop a model that accurately predicts battery failures, reducing downtime and maintenance costs.

## Data Collection and Description
The data was sourced from operational logs of battery usage in AGVs, ensuring high reliability due to the controlled collection environment.

### Description 
The dataset covers a detailed 20-day period, recording minute-by-minute SoC readings from multiple batteries, totaling over 50,000 records. Each record includes the battery's carrier ID, timestamp, and SoC percentage, among other operational parameters like charging status, e.t.c.

## Data Preprocessing and Initial Exploration 
### Cleaning and Preprocessing 
The data preprocessing involved:
1. Removing Unnecessary Columns: Columns unrelated to the analysis, such as unnamed or redundant fields, were dropped.
2. Handling Missing Values: Missing values in critical fields like SoC were imputed using forward fill, ensuring no interruption in time series data.
3. Timestamp Parsing: All timestamps were converted to datetime objects to facilitate time-based analysis and feature engineering.
   
### Initial Exploration 
My initial exploration of the dataset revealed a well-maintained dataset with minimal missing values and a consistent data entry format. While the date appeared relatively clean, I later discovered a vital column containing information on whether or not a battery actually failed was missing and there was no wayto gather that information in the alloted time. Aside that, preliminary statistics provided insights into the general health of the batteries, with SoC levels varying widely, indicating diverse usage patterns and battery conditions which would later be explored to account for the missing definitive column. An additional dataset containing a small record of failed and active batteries was later provided and the final model performance was benchmarked against that.

## Feature Engineering 
Given that there was initially no definitive column stating whether or not the batteries ultimately failed except the column that supposedly contained the timestamp of all recorded faults, feature engineering for this project was guided by a hypothesis based on the observations from the data and domain knowledge that rapid changes in SoC are indicative of potential battery failure. Additional Features such as "SOC decline rate" - Calculated as the percentage change in SoC over rolling windows of 24 hours to capture daily volatility, and "Critical Soc" - A binary feature indicating whether the SoC dropped below 20%, considered a critical threshold for battery operation were also developed to better capture those dynamics.
Both these features proved crucial in improving the model sensitivity to potential failures, significantly improving the predictive accuracy when included in the machine learning models 

## Methodology and model Development 

## Model Evaluation and Selection 

## Explainable AI(XAI) and Model Interpretation 

## Results and future advancements
