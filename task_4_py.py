# ================================================================
# WEEK 4 TASK
# PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS SYSTEMS
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ================================================================
# 1. PROJECT SETUP
# ================================================================

OUTPUT_FOLDER = "Week4_Logistics_Results"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("=" * 70)
print("WEEK 4 - PREDICTIVE MODELING AND OPTIMIZATION")
print("=" * 70)


# ================================================================
# 2. DATA SIMULATION
# ================================================================

print("\n[1] CREATING LOGISTICS DATASET...")

np.random.seed(42)

n = 5000

data = pd.DataFrame({

    "shipment_id":
        range(1, n + 1),

    "distance_km":
        np.random.uniform(5, 500, n),

    "package_weight_kg":
        np.random.uniform(0.5, 50, n),

    "traffic_index":
        np.random.uniform(1, 10, n),

    "weather_index":
        np.random.uniform(1, 10, n),

    "vehicle_age_years":
        np.random.uniform(0, 15, n),

    "warehouse_processing_hours":
        np.random.uniform(0.5, 8, n),

    "driver_experience_years":
        np.random.uniform(0.5, 20, n),

    "number_of_stops":
        np.random.randint(1, 15, n),

    "fuel_price_per_litre":
        np.random.uniform(85, 110, n),

    "order_priority":
        np.random.randint(1, 4, n),

    "vehicle_capacity_kg":
        np.random.choice(
            [500, 1000, 1500, 2000, 3000],
            n
        )
})


# ================================================================
# 3. CREATE TARGET VARIABLE
# ================================================================

noise = np.random.normal(0, 2.5, n)

data["delivery_time_hours"] = (

    1.5

    + 0.018 * data["distance_km"]

    + 0.025 * data["package_weight_kg"]

    + 0.75 * data["traffic_index"]

    + 0.45 * data["weather_index"]

    + 0.20 * data["vehicle_age_years"]

    + 0.55 * data["warehouse_processing_hours"]

    + 0.18 * data["number_of_stops"]

    - 0.10 * data["driver_experience_years"]

    - 0.35 * data["order_priority"]

    + noise
)

data["delivery_time_hours"] = data[
    "delivery_time_hours"
].clip(lower=1)


print("\nDataset created successfully!")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Records:")
print(data.head().to_string(index=False))


# ================================================================
# 4. DATA QUALITY CHECK
# ================================================================

print("\n" + "=" * 70)
print("[2] DATA QUALITY CHECK")
print("=" * 70)

missing_values = data.isnull().sum().sum()

duplicate_rows = data.duplicated().sum()

print("\nTotal Missing Values:",
      missing_values)

print("Total Duplicate Rows:",
      duplicate_rows)

print("\nDataset Information:")
print(data.info())


# ================================================================
# 5. DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 70)
print("[3] DESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    data.describe().round(2).to_string()
)


# ================================================================
# 6. LOGISTICS KPI ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("[4] LOGISTICS KPI ANALYSIS")
print("=" * 70)

average_delivery = data[
    "delivery_time_hours"
].mean()

median_delivery = data[
    "delivery_time_hours"
].median()

minimum_delivery = data[
    "delivery_time_hours"
].min()

maximum_delivery = data[
    "delivery_time_hours"
].max()

average_distance = data[
    "distance_km"
].mean()

average_weight = data[
    "package_weight_kg"
].mean()

on_time_threshold = 24

on_time_percentage = (

    data["delivery_time_hours"]
    .le(on_time_threshold)
    .mean()
    * 100
)


print(
    f"\nAverage Delivery Time : "
    f"{average_delivery:.2f} hours"
)

print(
    f"Median Delivery Time  : "
    f"{median_delivery:.2f} hours"
)

print(
    f"Minimum Delivery Time : "
    f"{minimum_delivery:.2f} hours"
)

print(
    f"Maximum Delivery Time : "
    f"{maximum_delivery:.2f} hours"
)

print(
    f"Average Distance      : "
    f"{average_distance:.2f} km"
)

print(
    f"Average Package Weight: "
    f"{average_weight:.2f} kg"
)

print(
    f"On-Time Delivery Rate : "
    f"{on_time_percentage:.2f}%"
)


# ================================================================
# 7. CORRELATION ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("[5] CORRELATION ANALYSIS")
print("=" * 70)

numeric_data = data.select_dtypes(
    include=np.number
)

correlation = numeric_data.corr()[
    "delivery_time_hours"
].sort_values(
    ascending=False
)

print(
    "\nCorrelation with Delivery Time:"
)

print(
    correlation.round(3).to_string()
)


# ================================================================
# 8. VISUALIZATION - DELIVERY TIME DISTRIBUTION
# ================================================================

plt.figure(figsize=(9, 6))

plt.hist(
    data["delivery_time_hours"],
    bins=40,
    edgecolor="black"
)

plt.title(
    "Distribution of Delivery Time"
)

plt.xlabel(
    "Delivery Time (Hours)"
)

plt.ylabel(
    "Number of Shipments"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_delivery_time_distribution.png"
    ),
    dpi=300
)

plt.show()


# ================================================================
# 9. VISUALIZATION - DISTANCE VS DELIVERY TIME
# ================================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    data["distance_km"],
    data["delivery_time_hours"],
    alpha=0.35
)

plt.title(
    "Distance vs Delivery Time"
)

plt.xlabel(
    "Distance (km)"
)

plt.ylabel(
    "Delivery Time (hours)"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_distance_vs_delivery_time.png"
    ),
    dpi=300
)

plt.show()


# ================================================================
# 10. VISUALIZATION - TRAFFIC VS DELIVERY TIME
# ================================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    data["traffic_index"],
    data["delivery_time_hours"],
    alpha=0.35
)

plt.title(
    "Traffic Index vs Delivery Time"
)

plt.xlabel(
    "Traffic Index"
)

plt.ylabel(
    "Delivery Time (hours)"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_traffic_vs_delivery_time.png"
    ),
    dpi=300
)

plt.show()


# ================================================================
# 11. PREPARE DATA FOR MACHINE LEARNING
# ================================================================

print("\n" + "=" * 70)
print("[6] PREPARING DATA FOR MACHINE LEARNING")
print("=" * 70)

X = data.drop(
    columns=[
        "shipment_id",
        "delivery_time_hours"
    ]
)

y = data[
    "delivery_time_hours"
]


# ================================================================
# 12. TRAIN TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)

print(
    "\nTraining Records:",
    len(X_train)
)

print(
    "Testing Records :",
    len(X_test)
)

print(
    "Number of Features:",
    X.shape[1]
)


# ================================================================
# 13. MODEL 1 - LINEAR REGRESSION
# ================================================================

print("\n" + "=" * 70)
print("[7] LINEAR REGRESSION")
print("=" * 70)

linear_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LinearRegression()
    )
])


linear_model.fit(
    X_train,
    y_train
)


linear_predictions = linear_model.predict(
    X_test
)


linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)


linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)


linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print(
    f"\nMAE  : {linear_mae:.4f}"
)

print(
    f"RMSE : {linear_rmse:.4f}"
)

print(
    f"R²   : {linear_r2:.4f}"
)


# ================================================================
# 14. MODEL 2 - DECISION TREE
# ================================================================

print("\n" + "=" * 70)
print("[8] DECISION TREE REGRESSOR")
print("=" * 70)

tree_model = DecisionTreeRegressor(

    max_depth=10,

    min_samples_split=10,

    random_state=42
)


tree_model.fit(
    X_train,
    y_train
)


tree_predictions = tree_model.predict(
    X_test
)


tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)


tree_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tree_predictions
    )
)


tree_r2 = r2_score(
    y_test,
    tree_predictions
)


print(
    f"\nMAE  : {tree_mae:.4f}"
)

print(
    f"RMSE : {tree_rmse:.4f}"
)

print(
    f"R²   : {tree_r2:.4f}"
)


# ================================================================
# 15. MODEL 3 - RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("[9] RANDOM FOREST REGRESSOR")
print("=" * 70)

random_forest = RandomForestRegressor(

    n_estimators=100,

    max_depth=15,

    min_samples_split=5,

    random_state=42,

    n_jobs=1
)


random_forest.fit(
    X_train,
    y_train
)


rf_predictions = random_forest.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)


rf_r2 = r2_score(
    y_test,
    rf_predictions
)


print(
    f"\nMAE  : {rf_mae:.4f}"
)

print(
    f"RMSE : {rf_rmse:.4f}"
)

print(
    f"R²   : {rf_r2:.4f}"
)


# ================================================================
# 16. MODEL COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("[10] MODEL COMPARISON")
print("=" * 70)

results = pd.DataFrame({

    "Model": [

        "Linear Regression",

        "Decision Tree",

        "Random Forest"
    ],

    "MAE": [

        linear_mae,

        tree_mae,

        rf_mae
    ],

    "RMSE": [

        linear_rmse,

        tree_rmse,

        rf_rmse
    ],

    "R2": [

        linear_r2,

        tree_r2,

        rf_r2
    ]
})


results = results.sort_values(
    by="RMSE"
).reset_index(drop=True)


print(
    "\n" +
    results.round(4).to_string(
        index=False
    )
)


results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_comparison.csv"
    ),
    index=False
)


# ================================================================
# 17. CROSS VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("[11] 5-FOLD CROSS VALIDATION")
print("=" * 70)

cv_model = RandomForestRegressor(

    n_estimators=75,

    max_depth=15,

    random_state=42,

    n_jobs=1
)


cv_scores = cross_val_score(

    cv_model,

    X,

    y,

    cv=5,

    scoring="neg_root_mean_squared_error",

    n_jobs=1
)


cv_rmse_scores = -cv_scores


print(
    "\nCross-Validation RMSE Scores:"
)


for i, score in enumerate(
    cv_rmse_scores,
    start=1
):

    print(
        f"Fold {i}: {score:.4f}"
    )


mean_cv_rmse = cv_rmse_scores.mean()

std_cv_rmse = cv_rmse_scores.std()


print(
    f"\nMean CV RMSE: "
    f"{mean_cv_rmse:.4f}"
)

print(
    f"CV RMSE Standard Deviation: "
    f"{std_cv_rmse:.4f}"
)


# ================================================================
# 18. HYPERPARAMETER TUNING
# ================================================================

print("\n" + "=" * 70)
print("[12] HYPERPARAMETER TUNING")
print("=" * 70)


param_grid = {

    "n_estimators": [
        50,
        75
    ],

    "max_depth": [
        10,
        15
    ],

    "min_samples_split": [
        2,
        5
    ]
}


grid_model = GridSearchCV(

    RandomForestRegressor(

        random_state=42,

        n_jobs=1
    ),

    param_grid=param_grid,

    cv=3,

    scoring="neg_root_mean_squared_error",

    n_jobs=1
)


print(
    "\nRunning GridSearchCV..."
)

print(
    "Please wait..."
)


grid_model.fit(
    X_train,
    y_train
)


best_model = grid_model.best_estimator_


print(
    "\nBest Parameters:"
)

print(
    grid_model.best_params_
)


best_cv_rmse = -grid_model.best_score_


print(
    f"\nBest CV RMSE: "
    f"{best_cv_rmse:.4f}"
)


# ================================================================
# 19. FINAL TUNED MODEL
# ================================================================

print("\n" + "=" * 70)
print("[13] FINAL TUNED RANDOM FOREST MODEL")
print("=" * 70)


tuned_predictions = best_model.predict(
    X_test
)


tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)


tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_predictions
    )
)


tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)


print(
    f"\nFinal MAE  : "
    f"{tuned_mae:.4f} hours"
)

print(
    f"Final RMSE : "
    f"{tuned_rmse:.4f} hours"
)

print(
    f"Final R²   : "
    f"{tuned_r2:.4f}"
)


# ================================================================
# 20. ACTUAL VS PREDICTED GRAPH
# ================================================================

plt.figure(figsize=(9, 6))


plt.scatter(

    y_test,

    tuned_predictions,

    alpha=0.4
)


minimum = min(

    y_test.min(),

    tuned_predictions.min()
)


maximum = max(

    y_test.max(),

    tuned_predictions.max()
)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    linestyle="--"
)


plt.title(
    "Actual vs Predicted Delivery Time"
)

plt.xlabel(
    "Actual Delivery Time (hours)"
)

plt.ylabel(
    "Predicted Delivery Time (hours)"
)

plt.grid(alpha=0.3)

plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "04_actual_vs_predicted.png"
    ),

    dpi=300
)


plt.show()


# ================================================================
# 21. RESIDUAL ANALYSIS
# ================================================================

residuals = (
    y_test -
    tuned_predictions
)


plt.figure(figsize=(9, 6))


plt.scatter(

    tuned_predictions,

    residuals,

    alpha=0.4
)


plt.axhline(

    y=0,

    linestyle="--"
)


plt.title(
    "Residual Analysis"
)

plt.xlabel(
    "Predicted Delivery Time (hours)"
)

plt.ylabel(
    "Residual"
)

plt.grid(alpha=0.3)

plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "05_residual_analysis.png"
    ),

    dpi=300
)


plt.show()


# ================================================================
# 22. FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 70)
print("[14] FEATURE IMPORTANCE")
print("=" * 70)


importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        best_model.feature_importances_
})


importance = importance.sort_values(

    by="Importance",

    ascending=False
)


print(
    "\nFeature Importance:"
)

print(

    importance.round(4).to_string(
        index=False
    )
)


importance.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "feature_importance.csv"
    ),

    index=False
)


# ================================================================
# 23. FEATURE IMPORTANCE GRAPH
# ================================================================

top_features = importance.head(10)


plt.figure(figsize=(10, 6))


plt.barh(

    top_features["Feature"][::-1],

    top_features["Importance"][::-1]
)


plt.title(
    "Top 10 Features Affecting Delivery Time"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "06_feature_importance.png"
    ),

    dpi=300
)


plt.show()


# ================================================================
# 24. LOGISTICS OPTIMIZATION SCENARIO
# ================================================================

print("\n" + "=" * 70)
print("[15] LOGISTICS OPTIMIZATION SCENARIO")
print("=" * 70)


baseline = pd.DataFrame({

    "distance_km": [250],

    "package_weight_kg": [20],

    "traffic_index": [8],

    "weather_index": [7],

    "vehicle_age_years": [10],

    "warehouse_processing_hours": [6],

    "driver_experience_years": [3],

    "number_of_stops": [12],

    "fuel_price_per_litre": [100],

    "order_priority": [2],

    "vehicle_capacity_kg": [1000]
})


optimized = baseline.copy()


# Optimization actions

optimized["traffic_index"] = 5

optimized["weather_index"] = 5

optimized["vehicle_age_years"] = 5

optimized["warehouse_processing_hours"] = 3

optimized["driver_experience_years"] = 8

optimized["number_of_stops"] = 7

optimized["vehicle_capacity_kg"] = 1500


baseline_prediction = best_model.predict(

    baseline
)[0]


optimized_prediction = best_model.predict(

    optimized
)[0]


time_saved = (

    baseline_prediction -

    optimized_prediction
)


percentage_reduction = (

    time_saved /

    baseline_prediction
) * 100


print(
    f"\nBaseline Predicted Delivery Time:"
    f" {baseline_prediction:.2f} hours"
)


print(
    f"Optimized Predicted Delivery Time:"
    f" {optimized_prediction:.2f} hours"
)


print(
    f"Estimated Time Saved:"
    f" {time_saved:.2f} hours"
)


print(
    f"Estimated Improvement:"
    f" {percentage_reduction:.2f}%"
)


# ================================================================
# 25. OPTIMIZATION SUMMARY
# ================================================================

optimization_summary = pd.DataFrame({

    "Metric": [

        "Baseline Delivery Time",

        "Optimized Delivery Time",

        "Estimated Time Saved",

        "Estimated Reduction (%)"
    ],

    "Value": [

        baseline_prediction,

        optimized_prediction,

        time_saved,

        percentage_reduction
    ]
})


print(
    "\nOptimization Summary:"
)

print(

    optimization_summary.round(2).to_string(
        index=False
    )
)


optimization_summary.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "optimization_summary.csv"
    ),

    index=False
)


# ================================================================
# 26. SAMPLE PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("[16] SAMPLE PREDICTIONS")
print("=" * 70)


prediction_sample = pd.DataFrame({

    "Actual_Delivery_Time":
        y_test.iloc[:15].values,

    "Predicted_Delivery_Time":
        tuned_predictions[:15]
})


prediction_sample["Absolute_Error"] = (

    abs(

        prediction_sample[
            "Actual_Delivery_Time"
        ]

        -

        prediction_sample[
            "Predicted_Delivery_Time"
        ]
    )
)


print(

    prediction_sample.round(2).to_string(
        index=False
    )
)


prediction_sample.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "sample_predictions.csv"
    ),

    index=False
)


# ================================================================
# 27. FINAL RESULTS FILE
# ================================================================

final_results = pd.DataFrame({

    "Metric": [

        "Dataset Records",

        "Dataset Features",

        "Average Delivery Time",

        "Median Delivery Time",

        "On-Time Delivery Rate (%)",

        "Linear Regression MAE",

        "Linear Regression RMSE",

        "Linear Regression R2",

        "Decision Tree MAE",

        "Decision Tree RMSE",

        "Decision Tree R2",

        "Random Forest MAE",

        "Random Forest RMSE",

        "Random Forest R2",

        "Tuned Random Forest MAE",

        "Tuned Random Forest RMSE",

        "Tuned Random Forest R2",

        "Mean 5-Fold CV RMSE",

        "CV RMSE Standard Deviation",

        "Baseline Predicted Time",

        "Optimized Predicted Time",

        "Estimated Time Saved",

        "Optimization Reduction (%)"
    ],

    "Value": [

        len(data),

        X.shape[1],

        average_delivery,

        median_delivery,

        on_time_percentage,

        linear_mae,

        linear_rmse,

        linear_r2,

        tree_mae,

        tree_rmse,

        tree_r2,

        rf_mae,

        rf_rmse,

        rf_r2,

        tuned_mae,

        tuned_rmse,

        tuned_r2,

        mean_cv_rmse,

        std_cv_rmse,

        baseline_prediction,

        optimized_prediction,

        time_saved,

        percentage_reduction
    ]
})


final_results.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "final_results.csv"
    ),

    index=False
)


# ================================================================
# 28. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FINAL ANALYSIS SUMMARY")
print("=" * 70)


print(
    "\nBest Model:"
)

print(
    "Tuned Random Forest Regressor"
)


print(
    f"\nFinal MAE:"
    f" {tuned_mae:.4f} hours"
)


print(
    f"Final RMSE:"
    f" {tuned_rmse:.4f} hours"
)


print(
    f"Final R²:"
    f" {tuned_r2:.4f}"
)


print(
    f"\nMean 5-Fold CV RMSE:"
    f" {mean_cv_rmse:.4f}"
)


print(
    f"\nBaseline Predicted Time:"
    f" {baseline_prediction:.2f} hours"
)


print(
    f"Optimized Predicted Time:"
    f" {optimized_prediction:.2f} hours"
)


print(
    f"Estimated Time Saved:"
    f" {time_saved:.2f} hours"
)


print(
    f"Estimated Improvement:"
    f" {percentage_reduction:.2f}%"
)


print(
    "\nTop 5 Important Features:"
)


for index, row in importance.head(5).iterrows():

    print(

        f"- {row['Feature']}: "
        f"{row['Importance']:.4f}"
    )


# ================================================================
# 29. COMPLETION MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("ALL ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 70)


print(
    "\nAll output files and graphs are saved in:"
)

print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)


print("\nFiles generated:")

print(
    "1. 01_delivery_time_distribution.png"
)

print(
    "2. 02_distance_vs_delivery_time.png"
)

print(
    "3. 03_traffic_vs_delivery_time.png"
)

print(
    "4. 04_actual_vs_predicted.png"
)

print(
    "5. 05_residual_analysis.png"
)

print(
    "6. 06_feature_importance.png"
)

print(
    "7. model_comparison.csv"
)

print(
    "8. feature_importance.csv"
)

print(
    "9. optimization_summary.csv"
)

print(
    "10. sample_predictions.csv"
)

print(
    "11. final_results.csv"
)


print("\nTake screenshots of the important console outputs")
print("and all six graphs for the final DOC report.")

print("\nWEEK 4 TASK COMPLETED.")
