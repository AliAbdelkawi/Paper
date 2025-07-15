
"""
Gypsum induciton time prediction: Integrated machine learning-Smoluchowski model
 Author: Ali A. Abdelkawi
         PhD Candidate
         University of Minnesota, Twin Cities

  Email: abdel259@umn.edu

  Reference:
   [1] (under review) A. A. Abdelkawi, J. L. Lindsey, and N. C. Wright, “Prediction of gypsum induction time to inform scaling kinetics using machine learning and Smoluchowski theory”.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import shap
import joblib
from sklearn.inspection import PartialDependenceDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable
#---------------Functions----------------------------------------------
#depenedce plot function
def dependence_plot(feature, xlabel):
    shap_vals = shap_values[:, X_test.columns.get_loc(feature)].values  # Convert to array    
    plt.figure(figsize=(6, 5), dpi=300)
    plt.scatter(X_test[feature], shap_vals, color="#ffaf08", alpha=0.8, edgecolor='k', linewidth=0.3)   
    # Horizontal baseline
    plt.axhline(0, linestyle='--', color='gray', linewidth=1)
    # Formatting
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', direction='in', length=5, width=1)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("SHAP value", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.show()
 #--------------------------Bivariate dependence plot
def BDP(features,xlabel,ylabel) :   
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = PartialDependenceDisplay.from_estimator(
        best_model,
        X,
        features,
        kind='average',
        grid_resolution=10,
        ax=ax,
        contour_kw={
            "cmap": "coolwarm_r",  # red = low, blue = high
            "linewidths": 0
        }
    )    
    # Access the contour object
    contour = disp.contours_[0, 0]
    for c in contour.collections:
        c.set_edgecolor("face")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', direction='in', length=5, width=1, labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', direction='in', length=5, width=1)
    # Add a separate axis for colorbar (well outside)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.3)  # Increase pad to push it farther
    
    # Create colormap
    cbar = fig.colorbar(contour.collections[0], cax=cax)
    cbar.set_label("Partial dependence", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Layout and display
    plt.tight_layout()
    plt.show()

#--------Bayesian optimization fnctions-----------------
param_space_gb = {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(0.01, 0.3, prior='uniform'),
    'max_depth': Integer(3, 12),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'subsample': Real(0.5, 1.0, prior='uniform'),
}

#--Start the code-----------------------------------------------------------------------
file_name = "Database.xlsx" # Reading data
data = pd.read_excel(file_name, skiprows=0)

# Defining Features and target
X = data[['Temperature_C', 'salt concnetration_M', '1/SI2','Interfacial energy', 'ca_act']]#,'Interfacial energy','Cs','ionicstrength'
y = data['log induction time_min']
# Clean up column names
data.columns = data.columns.str.replace(r'\[|\]| ', '', regex=True)
groups=data['Label']

# Initialize DataFrame to store results
results = pd.DataFrame(columns=['Random_State', 'Model',  'R²_Test', 'RMSE_Test', 'MAE_Test'])
rndm=10 #can be any random state 

# Group-based Train-Test Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

# Grouped Splitting into train and test sets
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_groups = groups.iloc[train_idx]
#Group-based Cross-Validation for Bayesian Optimization
group_kf = GroupKFold(n_splits=10)
# Set up Bayesian Optimization
opt_gb = BayesSearchCV(
    GradientBoostingRegressor(),
    search_spaces=param_space_gb,
    n_iter=50,  # Number of iterations
    cv=group_kf.split(X_train, y_train, groups=train_groups),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
# Perform the search
opt_gb.fit(X_train, y_train)


# Print the best parameters and score
print("Best parameters found: ", opt_gb.best_params_)
print("Best score found: ", -opt_gb.best_score_)

# Train the best model on the entire training set
best_model = opt_gb.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_pred_train=best_model.predict(X_train)
joblib.dump(best_model, "ML_Smol_model.pkl")# Save the best model.
# Test set evaluation
r2_test = r2_score(y_test, y_pred)
Rmse_test = np.sqrt(np.mean(np.square(y_test - y_pred)))
mae_test = mean_absolute_error(y_test, y_pred)


# Store the results
results = results.append({
    'Random_State': rndm,
    'Model': 'ML+smol',
    'R²_Test': r2_test,
    'RMSE_Test': Rmse_test,
    'MAE_Test': mae_test
}, ignore_index=True)
# Display or save the results
print(results)

# --Post processing and plotting-------------------------------------

# Scatter plot for predictions vs true values
x_line = np.linspace(min(y_test), max(y_test), 100)
plt.figure(figsize=(8, 8))
plt.plot(x_line, x_line, color='black', linestyle='--', label='1:1 Line')
plt.scatter(y_test, y_pred, color='blue', label='Test Data', alpha=0.6)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('1:1 Plot of Predictions vs True Values (Test Set)')
plt.legend()
plt.grid(True)
plt.show()
#SHAP explainer's plots
# Initialize SHAP Explainer for the trained model
explainer = shap.Explainer(best_model, X_train)
# Calculate SHAP values for the test set
shap_values = explainer(X_test,check_additivity=False)
# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test)
#  plot SHAP values for a single prediction
shap.plots.waterfall(shap_values[5])  # change the number 5 to any number you would like
# Partial and Bivariate dependence plots-----------------------------------------
dependence_plot("Interfacial energy", "Interfacial energy [mJ/m$^2$]")
dependence_plot("Temperature_C","Temperatrue[$^o$C]")
dependence_plot("1/SI2", "1/SI$^2$")
dependence_plot("ca_act","Ca$^{2+}$ activity")
dependence_plot("salt concnetration_M", "NaCl [M]")
# BDP to asses interaction between features
BDP([('1/SI2', 'ca_act')], "1/SI$^2$", "Ca$^{2+}$ activity")  
BDP([('1/SI2', 'salt concnetration_M')], "1/SI$^2$", "NaCl [M]")  
BDP([('1/SI2', 'Temperature_C')], "1/SI$^2$", "Temperature [$^o$C]")  
BDP([('1/SI2', 'Interfacial energy')], "1/SI$^2$", "Interfacial energy [mJ/m$^2$]")  
#----------------------------------end of code--------------------------------------------
