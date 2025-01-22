# Predictive-Analytics-for-AGVs : Battery Failure Prediction Using Machine Learning Techniques
This repository contains the code from one of my projects on Automated Guided Vehicles. It involved developing a system to predict potential battery failures based on SOC (State of Charge) drop rates and predicts with a 78% accuracy for unseen data, premptively alerting stakeholders of potential battery failures.

## Project Overview
Early detection of battery failures can significantly enhance operational efficiency and safety, particularly in the automotive, energy storage, and consumer electronics sectors.The project aims to use SOC (State of Charge) data to predict the likelihood of battery failures, thereby reducing downtime and maintenance costs. This README will provide a thorough walkthrough of the project's structure, methodologies employed, key findings, and insights derived from the analysis.

### Objective: 
Develop a model that accurately predicts battery failures, reducing downtime and maintenance costs.

### Key Technologies and Libraries
1. Base Libraries:
**Pandas** - For data manipulation and organization,
**NumPy** - For numerical computations and handling arrays,
**Matplotlib** - For creating visualizations and plots.
**Seaborn** - For statistical data visualization and enhancing Matplotlib plots.

3. Machine Learning Frameworks:
**scikit-learn** - For implementing a machine learning models and utilities, including:
   - SimpleImputer: For handling missing data.
   - StandardScaler: For feature scaling.
   - t-SNE (manifold.TSNE): For dimensionality reduction and visualization.
   - Train-Test Split and Cross-Validation: For model evaluation and splitting datasets.
   - MLPClassifier: For implementing multi-layer perceptron models.
   - RandomForestClassifier and GradientBoostingClassifier: For ensemble methods.
   - GridSearchCV: For hyperparameter tuning.
   - Evaluation Metrics: For performance measurement, including accuracy, ROC-AUC, confusion matrix, and classification report.
   - plot_tree: For visualizing decision trees

5. Deep Learning:
**Keras** - For building and training Neural Networks, specifically:
   - Sequential API: For constructing layer-by-layer models.
   - Dense Layers: For fully connected layers in the neural network.
   - Dropout and BatchNormalization: For regularization and improving training stability.
   - Optimizers (Adam): For efficient weight optimization.
   - Callbacks (EarlyStopping): To prevent overfitting during training.
   - Regularization (l1_l2): For penalizing large weights in the network.

4. Model Interpretability:
**LIME (Local Interpretable Model-Agnostic Explanations)** - For model interpretability and explaining individual predictions.

## Data Collection and Description
The data was sourced from operational logs of battery usage in AGVs, ensuring high reliability due to the controlled collection environment.

### Description 
The dataset covers a detailed 20-day period, recording minute-by-minute SoC readings from multiple batteries, totaling over 50,000 records. Each record includes the battery's carrier ID, timestamp, and SoC percentage, among other operational parameters like charging status, e.t.c.

## Data Preprocessing and Initial Exploration 
### Cleaning and Preprocessing 
The data preprocessing involved:
1. Removing Unnecessary Columns: Columns unrelated to the analysis, such as unnamed or redundant fields, were dropped.
2. Handling Missing Values: Missing values in critical fields like SoC were imputed using forward fill, ensuring no interruption in time series data.
3. Timestamp Parsing: All timestamps were converted to date-time objects to aid time-based analysis and feature engineering.
   
### Initial Exploration 
My initial exploration of the dataset revealed a well-maintained dataset with minimal missing values and a consistent data entry format. While the date appeared relatively clean, I later discovered a vital column containing information on whether or not a battery actually failed was missing and there was no way to gather that information in the alloted time. Aside that, preliminary statistics provided insights into the general health of the batteries, with SoC levels varying widely, indicating diverse usage patterns and battery conditions which would later be explored to account for the missing definitive column. An additional dataset containing a small record of failed and active batteries was later provided and the final model performance was benchmarked against that.

## Feature Engineering 
Given that there was initially no definitive column stating whether or not the batteries ultimately failed except the column that supposedly contained the timestamp of all recorded faults, feature engineering for this project was guided by a hypothesis based on the observations from the data and domain knowledge that rapid changes in SoC are indicative of potential battery failure. Additional Features such as "SOC decline rate" - Calculated as the percentage change in SoC over rolling windows of 24 hours to capture daily volatility, and "Critical Soc" - A binary feature indicating whether the SoC dropped below 20%, considered a critical threshold for battery operation were also developed to better capture those dynamics.
Both these features proved crucial in improving the model sensitivity to potential failures, significantly improving the predictive accuracy when included in the machine learning models 

## Methodology and model Development 
The model development process started with a selection of several machine learning algorithms suitable for binary classification problems. Given our goal to predict battery failure effectively, I opted for Gradient Boosting Machines (GBM) due to their robustness in handling diverse data types and complexities. Initially, simpler models like Logistic Regression and Random Forest were also tested to establish baselines for performance comparison.

Some of the challenges I encountered during the development phase included overfitting and initial underperformance on validation datasets. They were mitigated by introducing regularization techniques, adjusting model parameters, and employing ensemble methods to improve generalization. The integration of feature engineering outcomes significantly enriched the model's learning context, enabling a more nuanced understanding and detection of failure patterns.

## Model Evaluation and Selection 
For model evaluation, I used metrics such as—accuracy, precision, recall, F1-score, and ROC AUC—to assess each model's performance comprehensively. The GBM model emerged as the top performer, striking a balance between detecting actual failures (high recall) and maintaining a low false positive rate (high precision). This model achieved an accuracy of **78%**, with a notable improvement in handling edge cases compared to the earlier models.

To select the best model, I compared the metrics across all models tested, considering the trade-offs between detecting as many true positives as possible while minimizing false alarms. The GBM's better ROC AUC score highlighted its effectiveness in distinguishing between failing and non-failing batteries under different threshold settings.

## Explainable AI(XAI) and Model Interpretation 
To enhance trust and transparency in the model's predictions, I applied Explainable AI (XAI) techniques, specifically using LIME (Local Interpretable Model-agnostic Explanations) to illustrate how each feature influenced individual predictions. This was particularly insightful for stakeholders, as it provided a clear understanding of why certain batteries were flagged as likely to fail, based on real-time SoC behavior and historical trends.

It bolstered confidence in the model's decisions and offered actionable insights, such as identifying which battery features contributed most significantly to predicted failures. This knowledge is invaluable for preventive maintenance strategies and further refining the model's accuracy.

## Results and future advancements
The final model not only met but exceeded our initial expectations by accurately predicting battery failures, enabling preemptive action to reduce operational disruptions. There are a couple of ways the model could be advanced including:

1. Integrating real-time data feeds to enable dynamic prediction and continuous model learning.
2. Exploring additional predictive features such as temperature or voltage fluctuations, which could further enhance the model's sensitivity and specificity.
3. Developing a deployment strategy for embedding this model within existing operational infrastructures to automate failure detection processes.
