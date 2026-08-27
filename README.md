# week4-logistics-predictive-modeling
Predictive modeling and optimization of logistics delivery time using Python, machine learning, cross-validation, feature importance, and scenario-based optimization.


## Project Overview

This project focuses on predicting delivery time in a logistics system using machine learning and using predictive insights to improve operational performance.

A simulated logistics dataset containing 5,000 shipment records was created using Python. The dataset includes operational and environmental features such as distance, package weight, traffic index, weather index, vehicle age, warehouse processing time, driver experience, number of stops, fuel price, order priority, and vehicle capacity. The target variable is delivery time in hours.

## Objectives

- Predict delivery time for logistics shipments.
- Perform data quality checks and exploratory analysis.
- Calculate important logistics KPIs.
- Compare multiple regression models.
- Evaluate models using MAE, RMSE, and R².
- Perform five-fold cross-validation.
- Tune Random Forest hyperparameters using GridSearchCV.
- Identify important factors affecting delivery time.
- Develop an operational optimization scenario.

## Methodology

The project follows an end-to-end machine learning workflow:

1. Simulate the logistics dataset.
2. Check missing values and duplicate records.
3. Perform exploratory data analysis.
4. Calculate logistics KPIs.
5. Split the dataset into training and testing sets using an 80/20 split.
6. Train Linear Regression, Decision Tree Regression, and Random Forest Regression models.
7. Evaluate model performance using MAE, RMSE, and R².
8. Perform five-fold cross-validation.
9. Tune Random Forest hyperparameters using GridSearchCV.
10. Analyze feature importance.
11. Perform a scenario-based optimization analysis.
12. Develop operational recommendations.

## Models Used

### Linear Regression

Used as a transparent baseline model for predicting the continuous delivery-time target.

### Decision Tree Regression

Used to capture non-linear relationships and interactions between logistics variables.

### Random Forest Regression

Used for non-linear modeling, feature-importance analysis, and operational scenario analysis.

## Model Results

Linear Regression achieved the strongest held-out test performance:

- MAE: 1.9329 hours
- RMSE: 2.4243 hours
- R²: 0.7319

Five-fold cross-validation produced:

- Mean RMSE: 2.7180 hours
- Standard Deviation: 0.0766 hours

Random Forest hyperparameter tuning selected:

- n_estimators = 75
- max_depth = 15
- min_samples_split = 5

## Feature Importance

The analysis identified the following important factors:

1. Distance
2. Traffic Index
3. Warehouse Processing Time
4. Weather Index
5. Vehicle Age
6. Driver Experience
7. Number of Stops

These factors were used to develop practical logistics recommendations.

## Optimization Scenario

A baseline operating scenario was compared with an improved scenario.

- Baseline predicted delivery time: 21.76 hours
- Optimized predicted delivery time: 14.30 hours
- Estimated time saved: 7.46 hours
- Estimated improvement: 34.26%

The optimization scenario focuses on reducing traffic exposure, decreasing warehouse processing time, reducing unnecessary stops, improving vehicle allocation, and assigning experienced drivers to suitable routes.

## Visualizations

The project includes visual analysis for:

- Delivery Time Distribution
- Distance vs Delivery Time
- Traffic Index vs Delivery Time
- Actual vs Predicted Delivery Time
- Residual Analysis
- Feature Importance

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## How to Run

Install the required Python libraries:

```bash
pip install -r requirements.txt
