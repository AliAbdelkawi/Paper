
"""
Gypsum induciton time prediction-Smoluchowski model
 Author: Ali A. Abdelkawi
         PhD Candidate
         University of Minnesota, Twin Cities

  Email: abdel259@umn.edu

  Reference:
   [1] (under review) A. A. Abdelkawi, J. L. Lindsey, and N. C. Wright, “Prediction of gypsum induction time to inform scaling kinetics using machine learning and Smoluchowski theory”.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib
# Constants
Na = 6.022E23  # molecules/mole AVogadro's number
Kb = 1.38E-23  # J/K Boltzman constant
VM = 0.00007469  # m^3/mole gypsum molecular volume
vm = VM / Na
M = (32 / (3 * (2.3**3))) * (np.pi * vm**2) / Kb**3
# Read data from Excel file
file_name = "Database.xlsx"
data = pd.read_excel(file_name, skiprows=0, sheet_name='Smol')
data = pd.DataFrame(data)

# Extract relevant features and target variable
labels = data['label']  #  this column contains group labels
Temperature=[]
NACL=[]
# Prepare a DataFrame to store results
results = []
Exp_log_T_ind={}
mod_log_T_ind={}
_SI={}
# Define the model function
def Smol_model(var1_sub, C0_sub, del_0):
    return (2 / (K1 * C0_sub)) * (M * (var1_sub**3) * (del_0**3)-1 )

# Define the residual function for least squares
def residuals(params, var1_sub, C0_sub, T_ind_sub):
    del_0 = params[0]  # Extract parameter from array
    Tind_predicted_sub = Smol_model(var1_sub, C0_sub, del_0)
    return np.log(Tind_predicted_sub) - np.log(T_ind_sub)

def fit_and_evaluate_polynomial_model(degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict and evaluate
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Print results
    print(f"Polynomial Degree: {degree}")
    print("R² Score:", r2)
    print("Mean Squared Error:", mse)
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)
    print("\n")
    return model, r2, mse,y_pred

# Fit the model for each subgroup
for label in data['label'].unique():
    # Filter data for the current subgroup
    subgroup_data = data[data['label'] == label]
    act_ca_sub = subgroup_data['ca_act']
    T_sub = subgroup_data['Temp'] + 273
    Temperature.append(float(subgroup_data['Temp'].unique())+273)
    NACL.append(float(subgroup_data['Bcknd_conc'].unique()))
    SI_sub = subgroup_data['SI']
    _SI_sub=subgroup_data['1/SI2']
    C0_sub =act_ca_sub* Na * 1000# Here Iam introducing the activity instead of conc
    T_ind_sub = subgroup_data['Tind_mean'] * 60
    log_Tind_sub=np.log10(T_ind_sub)
    Nacl_sub = subgroup_data['Bcknd_conc']
    K1 = (10**-25)# Assuming the aggregation rate constant is constant across the different conditions (value chosen after trials)
    #--------------variables conc-----------------------   
    var1_sub = 1/(T_sub*SI_sub)
    combine_sub = (var1_sub, C0_sub)

    # Initial guesses and bounds
    initial_guesses = [5]  # Initial guess for del_0
    bounds = ([0], [np.inf])  # Setting bounds correctly

    # Perform least squares fitting
    result = least_squares(
        residuals,
        initial_guesses,
        bounds=bounds,
        args=(var1_sub, C0_sub, T_ind_sub),
        xtol=1e-40,
        max_nfev=10000000
    )

    # Extract optimized parameters
    del_0_fit = result.x[0]

    # Calculate predicted Tind_mean using the fitted parameters
    Tind_predicted_sub = Smol_model(var1_sub, C0_sub, del_0_fit)
    log_Tind_predicted_sub=np.log10(Tind_predicted_sub)
    
    Exp_log_T_ind[label]=log_Tind_sub
    mod_log_T_ind[label]=log_Tind_predicted_sub
    # Calculate R², MSE, and MAE
    r2_log = r2_score(log_Tind_sub, log_Tind_predicted_sub)
    Rmspe_log = np.sqrt(np.mean(np.square((log_Tind_sub - log_Tind_predicted_sub) / log_Tind_sub))) * 100
    mae_log = mean_absolute_error(log_Tind_sub, log_Tind_predicted_sub)
    
    r2 = r2_score(T_ind_sub, Tind_predicted_sub)
    Rmspe = np.sqrt(np.mean(np.square((T_ind_sub - Tind_predicted_sub) / T_ind_sub))) * 100
    mae = mean_absolute_error(T_ind_sub, Tind_predicted_sub)

    # Store the results
    results.append({
        'Label': label,
        'Fitted del0': del_0_fit,
        'R²': r2,
        'RMSPE': Rmspe,
        'MAE': mae,
        'R² log': r2_log,
        'RMSPE log': Rmspe_log,
        'MAE log': mae_log
        
    })
    plt.figure(figsize=(12, 8))   
    plt.scatter(_SI_sub, log_Tind_sub, alpha=0.9, edgecolors='k', label=f'Original (Label {label})')
    plt.scatter(_SI_sub, log_Tind_predicted_sub, alpha=0.9, edgecolors='k', label=f'Fitted (Label {label})')
    plt.xlabel('1/SI$^2$')
    plt.ylabel('log Tind')
    plt.title('Original vs Fitted Tind_mean (log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()
# Convert results to DataFrame
results_df = pd.DataFrame(results)
#------------------------------------------------------------------
del_0_final=results_df['Fitted del0']*1000

X = df = pd.DataFrame({
    'Temperature': Temperature,
    'NaCl': NACL
})
y = del_0_final
# Fit and evaluate third degree polynomial models of interfacial energy 
model_degree_3, r2_degree_3, mse_degree_3,y_3 = fit_and_evaluate_polynomial_model(degree=3)
fitted_results=[]
joblib.dump(model_degree_3, "Interfacial_energy_model.pkl")
# Get the induction time from the fitted interfacial energy
log_model=[]
log_exp=[]
for label in data['label'].unique():
    # Filter data for the current subgroup
    subgroup_data = data[data['label'] == label]    
    act_ca_sub = subgroup_data['ca_act']
    T_sub = subgroup_data['Temp'] + 273
    SI_sub = subgroup_data['SI']
    _SI_sub=subgroup_data['1/SI2']
    C0_sub =act_ca_sub* Na * 1000  
    T_ind_sub = subgroup_data['Tind_mean'] * 60
    log_Tind_sub=np.log10(T_ind_sub)
    Nacl_sub = subgroup_data['Bcknd_conc']
    K1 = (10**-25)
    ##using the fitted model
    X_sub = pd.DataFrame({
    'Temperature': [T_sub.iloc[0]],  # or specify a single value explicitly
    'NaCl': [Nacl_sub.iloc[0]]       # or specify a single value explicitly
})
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly_sub = poly.fit_transform(X_sub)
    IE_model = joblib.load("Interfacial_energy_model.pkl")
    del_0_empirical=IE_model.predict(X_poly_sub)/1000    
    #--------------variables conc-----------------------
    var2=1/(T_sub*SI_sub)
    T_ind_fit=(2 / (K1 * C0_sub)) * (M * (var2**3) * (del_0_empirical**3)-1 )
    log_Tind_fit=np.log10(T_ind_fit)
    
    mod_log_T_ind[label]=log_Tind_fit
    Exp_log_T_ind[label]=log_Tind_sub
    _SI[label]=_SI_sub
    for i in range(len(log_Tind_fit)):
        log_model.append(log_Tind_fit.iloc[i])   
        log_exp.append(log_Tind_sub.iloc[i])
    
    # Calculate R², MSE, and MAE
    r2 = r2_score(T_ind_sub, T_ind_fit)
    mse = mean_squared_error(T_ind_sub, T_ind_fit)
    mae = mean_absolute_error(T_ind_sub, T_ind_fit)
    # Calculate R², MSE, and MAE Log based
    r2_log = r2_score(log_Tind_sub, log_Tind_fit)
    Rmspe_log = np.sqrt(np.mean(np.square((log_Tind_sub - log_Tind_fit) / log_Tind_sub))) * 100
    mae_log = mean_absolute_error(log_Tind_sub, log_Tind_fit)
    
    # Store the results
    fitted_results.append({
        'Label': label,
        'R²': r2,
        'RMSPE': Rmspe,
        'MAE': mae,
        'R² log': r2_log,
        'RMSPE log': Rmspe_log,
        'MAE log': mae_log
        
    })

    plt.figure(figsize=(12, 8))   
    plt.scatter(_SI_sub, log_Tind_sub, alpha=0.9, edgecolors='k', label=f'Original (Label {label})')
    plt.scatter(_SI_sub, log_Tind_fit, alpha=0.9, edgecolors='k', label=f'predicted (Label {label})')    
    plt.xlabel('1/SI$^2$')
    plt.ylabel('log Tind')
    plt.title('Original vs Predicted Tind_mean (log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()
# Convert results to DataFrame
fitted_results_df = pd.DataFrame(fitted_results)
print('Smol_R2=',r2_score(log_exp,log_model))
print('Smol_MAE=',mean_absolute_error(log_exp,log_model))
print('Smol_RMSE=',np.sqrt(mean_squared_error(log_exp, log_model))) 
Exp_log_T_ind= pd.DataFrame(Exp_log_T_ind)
mod_log_T_ind=pd.DataFrame(mod_log_T_ind)

_SI=pd.DataFrame(_SI)
  
#-----------------------------------------------------------------


#--------------variables conc-----------------------

