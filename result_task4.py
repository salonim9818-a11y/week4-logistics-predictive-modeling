Python 3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.

====== RESTART: C:/Users/SACHIN KUMAR/OneDrive/Desktop/task_3/task_4_py.py =====
======================================================================
WEEK 4 - PREDICTIVE MODELING AND OPTIMIZATION
======================================================================

[1] CREATING LOGISTICS DATASET...

Dataset created successfully!

Dataset Shape:
(5000, 13)

First 5 Records:
 shipment_id  distance_km  package_weight_kg  traffic_index  weather_index  vehicle_age_years  warehouse_processing_hours  driver_experience_years  number_of_stops  fuel_price_per_litre  order_priority  vehicle_capacity_kg  delivery_time_hours
           1   190.397359          19.984958       4.362767       5.497032          10.949975                    1.192781                12.943819                9            105.004223               3                 1000            18.853986
           2   475.603582          23.935065       3.996209       7.720721           2.767680                    0.955653                 9.456203                6             93.781335               2                 1500            20.904388
           3   367.337001          42.800096       2.585385       6.064001           5.199595                    5.031440                19.307721                8             97.126953               2                  500            16.921006
           4   301.335950          17.330217       6.465400       1.749723           9.949210                    7.745872                 4.770080                3             89.698943               2                 2000            17.690791
           5    82.229227          43.547659       5.289617       2.670222           7.231340                    4.270410                11.963200                8             97.142460               3                 1000            10.759738

======================================================================
[2] DATA QUALITY CHECK
======================================================================

Total Missing Values: 0
Total Duplicate Rows: 0

Dataset Information:
<class 'pandas.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 13 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   shipment_id                 5000 non-null   int64  
 1   distance_km                 5000 non-null   float64
 2   package_weight_kg           5000 non-null   float64
 3   traffic_index               5000 non-null   float64
 4   weather_index               5000 non-null   float64
 5   vehicle_age_years           5000 non-null   float64
 6   warehouse_processing_hours  5000 non-null   float64
 7   driver_experience_years     5000 non-null   float64
 8   number_of_stops             5000 non-null   int32  
 9   fuel_price_per_litre        5000 non-null   float64
 10  order_priority              5000 non-null   int32  
 11  vehicle_capacity_kg         5000 non-null   int64  
 12  delivery_time_hours         5000 non-null   float64
dtypes: float64(9), int32(2), int64(2)
memory usage: 468.9 KB
None

======================================================================
[3] DESCRIPTIVE STATISTICS
======================================================================
       shipment_id  distance_km  package_weight_kg  traffic_index  weather_index  vehicle_age_years  warehouse_processing_hours  driver_experience_years  number_of_stops  fuel_price_per_litre  order_priority  vehicle_capacity_kg  delivery_time_hours
count      5000.00      5000.00            5000.00        5000.00        5000.00            5000.00                     5000.00                  5000.00          5000.00               5000.00         5000.00              5000.00              5000.00
mean       2500.50       250.93              24.83           5.51           5.57               7.63                        4.18                    10.26             7.45                 97.57            1.98              1594.20                16.77
std        1443.52       143.37              14.14           2.62           2.59               4.33                        2.13                     5.62             4.01                  7.24            0.81               851.51                 4.62
min           1.00         5.01               0.50           1.00           1.00               0.00                        0.50                     0.50             1.00                 85.01            1.00               500.00                 2.10
25%        1250.75       125.71              12.73           3.22           3.36               3.88                        2.35                     5.33             4.00                 91.41            1.00              1000.00                13.60
50%        2500.50       252.50              24.56           5.47           5.61               7.70                        4.18                    10.29             7.00                 97.48            2.00              1500.00                16.78
75%        3750.25       375.31              36.80           7.85           7.76              11.37                        5.99                    15.09            11.00                103.89            3.00              2000.00                19.88
max        5000.00       499.86              49.98          10.00          10.00              15.00                        8.00                    20.00            14.00                109.99            3.00              3000.00                32.85

======================================================================
[4] LOGISTICS KPI ANALYSIS
======================================================================

Average Delivery Time : 16.77 hours
Median Delivery Time  : 16.78 hours
Minimum Delivery Time : 2.10 hours
Maximum Delivery Time : 32.85 hours
Average Distance      : 250.93 km
Average Package Weight: 24.83 kg
On-Time Delivery Rate : 94.06%

======================================================================
[5] CORRELATION ANALYSIS
======================================================================

Correlation with Delivery Time:
delivery_time_hours           1.000
distance_km                   0.546
traffic_index                 0.422
weather_index                 0.279
warehouse_processing_hours    0.265
number_of_stops               0.173
vehicle_age_years             0.166
package_weight_kg             0.076
vehicle_capacity_kg          -0.001
fuel_price_per_litre         -0.004
shipment_id                  -0.012
order_priority               -0.062
driver_experience_years      -0.137

======================================================================
[6] PREPARING DATA FOR MACHINE LEARNING
======================================================================

Training Records: 4000
Testing Records : 1000
Number of Features: 11

======================================================================
[7] LINEAR REGRESSION
======================================================================

MAE  : 1.9329
RMSE : 2.4243
R²   : 0.7319

======================================================================
[8] DECISION TREE REGRESSOR
======================================================================

MAE  : 2.7452
RMSE : 3.4439
R²   : 0.4590

======================================================================
[9] RANDOM FOREST REGRESSOR
======================================================================

MAE  : 2.1024
RMSE : 2.6332
R²   : 0.6837

======================================================================
[10] MODEL COMPARISON
======================================================================

            Model    MAE   RMSE     R2
Linear Regression 1.9329 2.4243 0.7319
    Random Forest 2.1024 2.6332 0.6837
    Decision Tree 2.7452 3.4439 0.4590

======================================================================
[11] 5-FOLD CROSS VALIDATION
======================================================================

Cross-Validation RMSE Scores:
Fold 1: 2.7342
Fold 2: 2.5916
Fold 3: 2.7753
Fold 4: 2.6797
Fold 5: 2.8092

Mean CV RMSE: 2.7180
CV RMSE Standard Deviation: 0.0766

======================================================================
[12] HYPERPARAMETER TUNING
======================================================================

Running GridSearchCV...
Please wait...

Best Parameters:
{'max_depth': 15, 'min_samples_split': 5, 'n_estimators': 75}

Best CV RMSE: 2.7574

======================================================================
[13] FINAL TUNED RANDOM FOREST MODEL
======================================================================

Final MAE  : 2.0896 hours
Final RMSE : 2.6263 hours
Final R²   : 0.6854

======================================================================
[14] FEATURE IMPORTANCE
======================================================================

Feature Importance:
                   Feature  Importance
               distance_km      0.3495
             traffic_index      0.2110
warehouse_processing_hours      0.1099
             weather_index      0.1040
         vehicle_age_years      0.0625
   driver_experience_years      0.0437
           number_of_stops      0.0375
         package_weight_kg      0.0359
      fuel_price_per_litre      0.0279
       vehicle_capacity_kg      0.0101
            order_priority      0.0079

======================================================================
[15] LOGISTICS OPTIMIZATION SCENARIO
======================================================================

Baseline Predicted Delivery Time: 21.76 hours
Optimized Predicted Delivery Time: 14.30 hours
Estimated Time Saved: 7.46 hours
Estimated Improvement: 34.26%

Optimization Summary:
                 Metric  Value
 Baseline Delivery Time  21.76
Optimized Delivery Time  14.30
   Estimated Time Saved   7.46
Estimated Reduction (%)  34.26

======================================================================
[16] SAMPLE PREDICTIONS
======================================================================
 Actual_Delivery_Time  Predicted_Delivery_Time  Absolute_Error
                17.90                    19.92            2.01
                 8.87                    13.80            4.92
                15.66                    15.43            0.23
                18.30                    14.55            3.76
                24.91                    21.35            3.56
                12.85                    12.34            0.51
                10.31                     9.61            0.70
                12.88                    19.68            6.80
                13.49                    15.05            1.56
                13.45                    14.63            1.18
                17.46                    16.94            0.52
                15.67                    20.36            4.69
                16.60                    14.45            2.15
                10.25                    10.24            0.02
                17.89                    16.68            1.21

======================================================================
FINAL ANALYSIS SUMMARY
======================================================================

Best Model:
Tuned Random Forest Regressor

Final MAE: 2.0896 hours
Final RMSE: 2.6263 hours
Final R²: 0.6854

Mean 5-Fold CV RMSE: 2.7180

Baseline Predicted Time: 21.76 hours
Optimized Predicted Time: 14.30 hours
Estimated Time Saved: 7.46 hours
Estimated Improvement: 34.26%

Top 5 Important Features:
- distance_km: 0.3495
- traffic_index: 0.2110
- warehouse_processing_hours: 0.1099
- weather_index: 0.1040
- vehicle_age_years: 0.0625

======================================================================
ALL ANALYSIS COMPLETED SUCCESSFULLY!
======================================================================

All output files and graphs are saved in:
C:\Users\SACHIN KUMAR\OneDrive\Desktop\task_3\Week4_Logistics_Results

Files generated:
1. 01_delivery_time_distribution.png
2. 02_distance_vs_delivery_time.png
3. 03_traffic_vs_delivery_time.png
4. 04_actual_vs_predicted.png
5. 05_residual_analysis.png
6. 06_feature_importance.png
7. model_comparison.csv
8. feature_importance.csv
9. optimization_summary.csv
10. sample_predictions.csv
11. final_results.csv

Take screenshots of the important console outputs
and all six graphs for the final DOC report.

WEEK 4 TASK COMPLETED.
