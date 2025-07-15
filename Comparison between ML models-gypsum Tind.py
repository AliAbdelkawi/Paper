# -*- coding: utf-8 -*-

"""
Gypsum induction time prediction: Comparison between machine learning models
 Author: Ali A. Abdelkawi
         PhD Candidate
         University of Minnesota, Twin Cities

  Email: abdel259@umn.edu

  Reference:
   [1] (under review) A. A. Abdelkawi, J. L. Lindsey, and N. C. Wright, “Prediction of gypsum induction time to inform scaling kinetics using machine learning and Smoluchowski theory”.

"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the data
file_name = "Database.xlsx"
data = pd.read_excel(file_name, skiprows=0)
# Defining Features and target
X = data[['Temperature_C', 'salt concnetration_M', '1/SI2','ca_act']]
y = data['log induction time_min']
# Clean up column names 
data.columns = data.columns.str.replace(r'\[|\]| ', '', regex=True)
# Define the models
models = {
    'Dec.T': DecisionTreeRegressor(),
    'Rand.For': RandomForestRegressor(),
    'Grad.Bos': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(),
}

# Initialize DataFrame to store results
results = pd.DataFrame(columns=['Random_State', 'Model', 'R²_CV', 'MSE_CV', 'MAE_CV', 'R²_Test', 'MSE_Test', 'MAE_Test'])
# Loop over random states
for rndm in range(1, 101):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rndm)   
    # Define cross-validation method
    kf = KFold(n_splits=10, shuffle=True, random_state=rndm)    
    # Evaluate each model using cross-validation
    for name, model in models.items():
        # Cross-Validation
        cv_results_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2').mean()
        cv_results_mse = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
        cv_results_mae = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error').mean()

        # Train the model on the entire training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Test set evaluation
        r2_test = r2_score(y_test, y_pred)
        mse_test = mean_squared_error(y_test, y_pred)
        mae_test = mean_absolute_error(y_test, y_pred)
        
        # Store the results
        results = results.append({
            'Random_State': rndm,
            'Model': name,
            'R²_CV': cv_results_r2,
            'MSE_CV': cv_results_mse,
            'MAE_CV': cv_results_mae,
            'R²_Test': r2_test,
            'MSE_Test': mse_test,
            'MAE_Test': mae_test
        }, ignore_index=True)

# Display or save the results
print(results)
results.to_csv('model_performance_across_random_states.csv', index=False)

# Example of plotting the results for R² on the test set for the models
plt.figure(figsize=(10, 6))
for model in models.keys():
    model_results = results[results['Model'] == model]
    plt.plot(model_results['Random_State'], model_results['R²_Test'], label=model)
plt.title("R² Score Across Different Random States (Test Set)")
plt.xlabel("Random State")
plt.ylabel("R² Score")
plt.legend()
plt.show()
