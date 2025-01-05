# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:26:11 2024

@author: informatikalipi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,median_absolute_error
from IPython import get_ipython
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import sklearn.metrics as metrics
import csv
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.iolib.summary2 import summary_col
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'
from pandas import DataFrame,Series
 
#%%load online monitoring maninjau data
df_maninjau = pd.read_excel('Maninjau_New_Dataset2.xlsx')  #CopyManinjau_New_Dataset2
df_maninjau.columns
df_maninjau.shape
df_maninjau.head()
df_maninjau.info()
df_maninjau.dtypes
df_maninjau.describe()
df_maninjau.describe().T
df_maninjau.describe(include='all')
df_maninjau.isnull().sum()

#count value domg
Total_Count_domg_multi = (df_maninjau.iloc[:, 552:613] > 20).sum() #domg depth 0-60
print(Total_Count_domg_multi) #output: 107

#rata-rata
df_maninjau['avg_WT'] = df_maninjau.iloc[:, 3:64].mean(axis=1)
df_maninjau['avg_Salinity'] = df_maninjau.iloc[:, 64:125].mean(axis=1)
df_maninjau['avg_Conductivity'] = df_maninjau.iloc[:, 125:186].mean(axis=1)
df_maninjau['avg_EC25'] = df_maninjau.iloc[:, 186:247].mean(axis=1)
df_maninjau['avg_Density'] = df_maninjau.iloc[:, 247:308].mean(axis=1)
df_maninjau['avg_SigmaT'] = df_maninjau.iloc[:, 308:369].mean(axis=1)
df_maninjau['avg_CHl-Flu'] = df_maninjau.iloc[:, 369:430].mean(axis=1) 
df_maninjau['avg_CHl-a'] = df_maninjau.iloc[:, 430:491].mean(axis=1) 
df_maninjau['avg_Turbidity'] = df_maninjau.iloc[:, 491:552].mean(axis=1) 
df_maninjau['avg_domg'] = df_maninjau.iloc[:, 552:613].mean(axis=1) 
df_maninjau['avg_Batt'] = df_maninjau.iloc[:, 613:674].mean(axis=1) 

df_maninjau[['avg_WT','avg_Salinity','avg_Conductivity','avg_EC25',
             'avg_Density','avg_SigmaT','avg_CHl-Flu','avg_CHl-a',
             'avg_Turbidity','avg_domg','avg_Batt']].describe()

df_maninjau[['avg_EC25', 'avg_SigmaT','avg_CHl-Flu',
             'avg_Turbidity','avg_domg','avg_Batt']].describe()

#%%feature selection
# apply feature matrix
df_feature = df_maninjau.iloc[:,3:62*9-6] #df iloc[:, 3:613]
df_feature.to_csv('df_feature_dataset4.csv')
df_feature.columns
df_feature.shape

#variables: water temperature,salinity,conductivity,EC25,density,SigmaT,Chl-flu
#Chl-a ,Turbidity,Bathymetry

df_targetDO = df_maninjau.iloc[:,62*9-6:62*10-7] #   [:, 614:675]
df_targetDO.columns

#select highly correlated features: pearson's correlation
def correlation(dataset,threshold):
    col_corr= set() #set all features
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            #absolute coeffisien value
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

#select output features 
corr_high = correlation(df_targetDO, 0.8)
df_DO_new = df_targetDO.drop(corr_high,axis=1)
df_DO_new.shape
df_DO_new.to_csv('df_DO4.csv') 
#DOmg_l_Depth1,DOmg_l_Depth2,DOmg_l_Depth21= 0.8, 0.9=(53,9)

#select predictor features
corr_high = correlation(df_feature, 0.8)
df_feat_new = df_feature.drop(corr_high,axis=1)
df_feat_new.shape #(53, 71) = 0.8 (53, 40) = 0.7  (53,172)=0.9
df_feat_new.T
df_feat_new.to_csv('featureResult_result.csv')

#recursive 
df_MLR = df_maninjau[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth1","Chl-Flu_Depth2"]]

#features selection 
df_XGB = df_maninjau[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth13","Turb-M_Depth38","WaterTemp_depth0","Chl-Flu_Depth2","EC25_Depth2","Turb-M_Depth27","Turb-M_Depth33","Chl-Flu_Depth0",
"Turb-M_Depth56","Turb-M_Depth29","Turb-M_Depth26","Chl-Flu_Depth41","Turb-M_Depth35","Turb-M_Depth59","Salinity_Depth31","Turb-M_Depth39","Chl-Flu_Depth19","Chl-Flu_Depth37",
"Turb-M_Depth47","Chl-Flu_Depth1","Chl-Flu_Depth38","Turb-M_Depth51","Turb-M_Depth49","Salinity_Depth1","Turb-M_Depth54","Turb-M_Depth53","Turb-M_Depth24","Turb-M_Depth36",
"Salinity_Depth37","Turb-M_Depth22","Turb-M_Depth52","Chl-Flu_Depth50","Turb-M_Depth37","Turb-M_Depth23","Turb-M_Depth46","Turb-M_Depth41","Turb-M_Depth32","Turb-M_Depth43",
"Turb-M_Depth45","Turb-M_Depth31","Turb-M_Depth42","Chl-Flu_Depth58","Turb-M_Depth58","Turb-M_Depth55","Salinity_Depth26","Chl-Flu_Depth36","Turb-M_Depth25","Turb-M_Depth34",
"EC25_Depth1","Chl-Flu_Depth53","Turb-M_Depth44","Chl-Flu_Depth12"]] #54

df_SVR = df_maninjau[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth13","WaterTemp_depth0","Chl-Flu_Depth2","Turb-M_Depth59",
                      "Turb-M_Depth39","Turb-M_Depth47","Chl-Flu_Depth1","Turb-M_Depth49","Turb-M_Depth36",
                      "Turb-M_Depth48","Salinity_Depth37","Turb-M_Depth37","Turb-M_Depth23","Turb-M_Depth41",
                      "Turb-M_Depth43","Turb-M_Depth45","Turb-M_Depth31","Turb-M_Depth58","Turb-M_Depth44","Chl-Flu_Depth22","Chl-Flu_Depth16",
                      "Turb-M_Depth60","Chl-Flu_Depth14","Turb-M_Depth28"]] #26

df_RF = df_maninjau[["WaterTemp_depth1","Salinity_Depth0","Turb-M_Depth38","WaterTemp_depth0","Chl-Flu_Depth2","Turb-M_Depth27","Turb-M_Depth33","Chl-Flu_Depth0",
                     "Chl-Flu_Depth41","Turb-M_Depth35","Salinity_Depth31","Turb-M_Depth39","Chl-Flu_Depth37","Turb-M_Depth47","Chl-Flu_Depth1",
                     "Chl-Flu_Depth38","Turb-M_Depth49","Salinity_Depth1","Turb-M_Depth54","Turb-M_Depth53","Turb-M_Depth24","Turb-M_Depth36","Turb-M_Depth48",
                     "Turb-M_Depth57","Chl-Flu_Depth49","Salinity_Depth37","Turb-M_Depth22","Turb-M_Depth52","Turb-M_Depth37","Turb-M_Depth23",
                     "Turb-M_Depth46","Turb-M_Depth41","Turb-M_Depth32","Turb-M_Depth43","Turb-M_Depth45","Turb-M_Depth31","Turb-M_Depth42",
                     "Chl-Flu_Depth58","Salinity_Depth26","Turb-M_Depth34","EC25_Depth1","Chl-Flu_Depth53","Chl-Flu_Depth12","Chl-Flu_Depth22",
                     "Chl-Flu_Depth16","Chl-Flu_Depth48","Salinity_Depth30","Turb-M_Depth60","Chl-Flu_Depth42","Chl-Flu_Depth14","Chl-Flu_Depth34","Turb-M_Depth28"]]

#sorted df3: recirsive 2x
df_poly = df_maninjau[["Salinity_Depth0","WaterTemp_depth0","Turb-M_Depth38","Chl-Flu_Depth19","Turb-M_Depth37",
                       "Turb-M_Depth60","Turb-M_Depth36","Chl-Flu_Depth53","Turb-M_Depth40",
                       "Chl-Flu_Depth34","Turb-M_Depth24","Chl-Flu_Depth37","Chl-Flu_Depth38","EC25_Depth2"]]

df_poly2 = df_maninjau[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth1","Chl-Flu_Depth2"]]

#%%normalisation data
from sklearn.preprocessing import MinMaxScaler

# Membuat objek MinMaxScaler untuk fitur-fitur
scaler_x = MinMaxScaler()

# Menggunakan scaler_X untuk mengubah skala fitur-fitur
x_scaled = scaler_x.fit_transform(df_feat_new)
x_scaled = pd.DataFrame(x_scaled)

# Membuat objek MinMaxScaler untuk target (opsional, jika diperlukan)
scaler_y = MinMaxScaler()

# Menggunakan scaler_y untuk mengubah skala target
y_scaled = scaler_y.fit_transform(df_DO_new)
y_scaled = pd.DataFrame(y_scaled)

#scale utk feature MLR
x_scaled_mlr = scaler_x.fit_transform(df_MLR)
x_scaled_mlr = pd.DataFrame(x_scaled_mlr)

#scale untuk feature XGB
x_scaled_XGb = scaler_x.fit_transform(df_XGB)
x_scaled_XGb = pd.DataFrame(x_scaled_XGb)

#scale untuk feature SVR
x_scaled_SVR = scaler_x.fit_transform(df_SVR)
x_scaled_SVR = pd.DataFrame(x_scaled_SVR)

#scale untuk feature SVR
x_scaled_RF = scaler_x.fit_transform(df_RF)
x_scaled_RF = pd.DataFrame(x_scaled_RF)

#scale untuk poly
x_scaled_poly = scaler_x.fit_transform(df_poly2)
x_scaled_poly = pd.DataFrame(x_scaled_poly)

# Create a list to store the results for each feature scenario
results_list = []

#%% split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.15, random_state=42) #0.15

#%% MLR															
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as metrics

x_train, x_test, y_train, y_test = train_test_split(x_scaled_mlr, y_scaled, test_size=0.15, random_state=42) #0.15
reg = LinearRegression()
m_reg=reg.fit(x_train, y_train)
y_pred_multi=m_reg.predict(x_test)

test_r2_score = r2_score(y_test, y_pred_multi, multioutput='raw_values')
test_r2_score

mse_rf = mean_squared_error(y_test, y_pred_multi, multioutput='raw_values')
mse_rf

mae_rf = mean_absolute_error(y_test, y_pred_multi, multioutput='raw_values')
mae_rf

mape_rf = mean_absolute_percentage_error(y_test, y_pred_multi, multioutput='raw_values')
mape_rf

print(f"intercept: {m_reg.intercept_}")
print(f"coefficients: {m_reg.coef_}")

y_test2=np.array(y_test)
DOmg_l_Depth0 = 'blue'
DOmg_l_Depth2= 'black'
DOmg_l_Depth21= 'orange'

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test2[:, 0], y_pred_multi[:, 0], c=DOmg_l_Depth0, s=100)
plt.scatter(y_test2[:, 0], y_pred_multi[:, 1], c=DOmg_l_Depth2, s=100)
plt.scatter(y_test2[:, 0], y_pred_multi[:, 2], c=DOmg_l_Depth21, s=100)
#plt.scatter(y_test,y_pred_multi, c=colors, cmap='viridis', alpha=0.7, label='RF')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red', linestyle='--')
plt.xlabel('True Value')
plt.title('Plot of RF')
plt.tight_layout()
plt.show()

#%% #Evaluate model
evaluation_results = []

mse_rf = mean_squared_error(y_test, y_pred_multi)
rmse_rf = np.sqrt(mse_rf)
stdev_rf = np.std(y_pred_multi)
evaluation_results.append({
    'Model': 'mlr',
    'MSE': mse_rf,
    #'RMSE': rmse_rf,
    'MAE': mae_rf,
    'MAPE': mape_rf,
    #'Standard Deviation': stdev_rf,
    'Best R-squared Score': test_r2_score
})

# Create a DataFrame from the evaluation_results
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df) #0.21

#MLR_MTR
from sklearn.inspection import permutation_importance
result_MLR_MTR = permutation_importance(m_reg, x_test, y_test, n_repeats=4, random_state=42)
importance_MLR_MTR = result_MLR_MTR.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
#output 4 karena input feature: 4
importance_normalized_MLR_MTR = 100.0 * (importance_MLR_MTR / importance_MLR_MTR.sum())

#%%Extreme boost
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
x_train, x_test, y_train, y_test = train_test_split(x_scaled_XGb, y_scaled, test_size=0.15, random_state=42) #0.15

multi_reg = MultiOutputRegressor(estimator = XGBRegressor())
multi_reg.fit(x_train, y_train)
multi_reg_pred_GBM = multi_reg.predict(x_test)
test_r2_score_GBM = r2_score(y_test, multi_reg_pred_GBM, multioutput='raw_values')
test_r2_score_GBM

mse_GBM = mean_squared_error(y_test, multi_reg_pred_GBM, multioutput='raw_values')
mse_GBM 

mae_rf = mean_absolute_error(y_test, multi_reg_pred_GBM, multioutput='raw_values')
mae_rf

mape_rf = mean_absolute_percentage_error(y_test, multi_reg_pred_GBM, multioutput='raw_values')
mape_rf

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test2[:, 0], multi_reg_pred_GBM[:, 0], c=DOmg_l_Depth0, s=160)
plt.scatter(y_test2[:, 0], multi_reg_pred_GBM[:, 1], c=DOmg_l_Depth2, s=160)
plt.scatter(y_test2[:, 0], multi_reg_pred_GBM[:, 2], c=DOmg_l_Depth21, s=160)
#plt.scatter(y_test,multi_reg_pred_GBM, color='blue', label='RF')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red', linestyle='--')
plt.xlabel('True Value')
plt.title('Plot of GBM-XGBoost Regression')
plt.legend()
plt.tight_layout()
plt.show()

evaluation_results = []

# Evaluate model
mse_GBM = mean_squared_error(y_test, multi_reg_pred_GBM)
rmse_GBM = np.sqrt(mse_GBM)
stdev_GBM = np.std(multi_reg_pred_GBM)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_GBM,
    'RMSE': rmse_GBM,
    'Standard Deviation': stdev_GBM,
    'Best R-squared Score': test_r2_score_GBM.max()
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

#GBM_MTR
from sklearn.inspection import permutation_importance
result_GBM_MTR = permutation_importance(multi_reg, x_test, y_test, n_repeats=4, random_state=42)
importance_GBM_MTR = result_GBM_MTR.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_GBM_MTR = 100.0 * (importance_GBM_MTR / importance_GBM_MTR.sum())

#%%SVR
x_train, x_test, y_train, y_test = train_test_split(x_scaled_SVR, y_scaled, test_size=0.15, random_state=42) #0.15
from sklearn.svm import SVR
svr=MultiOutputRegressor(SVR(kernel='rbf')) #,epsilon=1.0
svr.fit(x_train,y_train)
pred2=svr.predict(x_test)
test_r2_score_SVR = r2_score(y_test, pred2, multioutput='raw_values')
test_r2_score_SVR

mse_SVR = mean_squared_error(y_test, pred2, multioutput='raw_values')
mse_SVR

mae_rf = mean_absolute_error(y_test, pred2, multioutput='raw_values')
mae_rf

mape_rf = mean_absolute_percentage_error(y_test, pred2, multioutput='raw_values')
mape_rf

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test2[:, 0], pred2[:, 0], c=DOmg_l_Depth0, s=160)
plt.scatter(y_test2[:, 0], pred2[:, 1], c=DOmg_l_Depth2, s=160)
plt.scatter(y_test2[:, 0], pred2[:, 2], c=DOmg_l_Depth21, s=160)
#plt.scatter(y_test2[:, 0], pred2[:, 1], c=DOmg_l_Depth21, s=160)

#plt.scatter(y_test,pred2)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Plot of SVR')
plt.show()

evaluation_results = []

# Evaluate model
mse_SVR = mean_squared_error(y_test, pred2)
rmse_SVR = np.sqrt(mse_SVR)
stdev_SVR = np.std(pred2)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_SVR,
    'RMSE': rmse_SVR,
    'Standard Deviation': stdev_SVR,
    'Best R-squared Score': test_r2_score_SVR.max() #untuk dapat 3 target, hapus max()
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

#SVR_MTR
from sklearn.inspection import permutation_importance
result_SVR_MTR = permutation_importance(svr, x_test, y_test, n_repeats=4, random_state=42)
importance_SVR_MTR = result_SVR_MTR.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_SVR_MTR = 100.0 * (importance_SVR_MTR / importance_SVR_MTR.sum())

#%%poly recursive
x_train, x_test, y_train, y_test = train_test_split(x_scaled_poly, y_scaled, test_size=0.15, random_state=42) #0.15
poly = PolynomialFeatures(degree=2,include_bias=False)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)

#build regression model
lr = LinearRegression()
lr.fit(x_train_trans, y_train)
score_train_MTR=lr.score(x_train_trans,y_train)
score_test_MTR=lr.score(x_test_trans,y_test)
y_pred2 = lr.predict(x_test_trans)
test_r2_score_poly = r2_score(y_test, y_pred2, multioutput='raw_values') #multioutput target DO
test_r2_score_poly

mse_poly = mean_squared_error(y_test, y_pred2, multioutput='raw_values')
mse_poly 

mae_rf = mean_absolute_error(y_test, y_pred2, multioutput='raw_values')
mae_rf

mape_rf = mean_absolute_percentage_error(y_test, y_pred2, multioutput='raw_values')
mape_rf

print(f"intercept: {lr.intercept_}")
print(f"coefficients: {lr.coef_}")

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test2[:, 0], y_pred2[:, 0], c=DOmg_l_Depth0, s=160)
plt.scatter(y_test2[:, 1], y_pred2[:, 1], c=DOmg_l_Depth2, s=160)
plt.scatter(y_test2[:, 2], y_pred2[:, 2], c=DOmg_l_Depth21, s=160)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Plot of poly')
plt.show()

#poly_feature: expect 47 features
#from sklearn.inspection import permutation_importance
#result_poly_MTR = permutation_importance(lr, x_test, y_test, n_repeats=4, random_state=42)
#importance_poly_MTR = result_poly_MTR.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
#importance_normalized_poly_MTR = 100.0 * (importance_poly_MTR / importance_poly_MTR.sum())

def get_permuted_score(idx,xtrain,xtest,ytrain,ytest):
    xtrain_permuted = xtrain.copy()
    xtrain_permuted.iloc[:, idx] = np.random.permutation(xtrain_permuted.iloc[:, idx])   # permute one column
    xtest_permuted = xtest.copy()
    xtest_permuted.iloc[:, idx] = np.random.permutation(xtest_permuted.iloc[:, idx])   # permute one column

    x_train_trans = poly.fit_transform(xtrain_permuted)
    x_test_trans  = poly.fit_transform(xtest_permuted)

    ## build regression model
    lr = LinearRegression()
    lr.fit(x_train_trans, y_train)

    ## score
    sc2_train = lr.score(x_train_trans,ytrain)
    sc2_test  = lr.score(x_test_trans,ytest)

    return sc2_train,sc2_test

FI_permuted_MTR = np.zeros((2,4))
for idx in range(4):
    FI_permuted_MTR[:,idx] = get_permuted_score(idx,x_train,x_test,y_train,y_test)

evaluation_results = []

# Evaluate model
mse_poly = mean_squared_error(y_test, y_pred2)
rmse_poly = np.sqrt(mse_rf)
stdev_poly = np.std(y_pred2)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_poly,
    'RMSE': rmse_poly,
    'Standard Deviation': stdev_poly,
    'Best R-squared Score': test_r2_score_poly.max()
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

## permuation importance:71
importance_poly_MTR1 = ["18.8297432102276","-54.7559215260545","11.6292786667537","4.41874846345599","-1.26247839069637",
                                   "-21.0399802118204","3.33420578586151","-21.0696833632066","-16.5167479164459","6.52727571657649",
                                   "15.8340270206523","1.28748201994023","-0.0738267161515447","-60.1434469179295","-2.29585755532811",
                                   "-1.76222900954392","0.0949983175512758","3.47822802130269","6.06827197126974","28.3072137038998",
                                   "9.88834416386705","13.7331373814412","3.70913784728064","23.4357035784873","24.0424128784333",
                                   "-4.12300873026789","-7.5905041825519","-5.32981246585229","6.89464932393676","0.671357366364137",
                                   "16.4265363815876","3.03834937758694","-6.9785293289361","-11.0277214199168","13.2340692980597",
                                   "6.73012087616066","-16.2138786622093","8.68737732972962","1.49482143976015","9.60852560863858",
                                   "5.66258764789117","9.53859991235821","6.28144937468983","-11.4560917744869","5.27144062093686",
                                   "-1.33734836357499","10.9576930370462","12.3053240392676","11.6647301152028","-16.6614844026605",
                                   "15.5600086706153","0.323926984030773","-0.423423878537412","5.48441819816532","7.22079079211825",
                                   "-11.1907439915967","-1.4617973820844","0.38866065204194","1.56696669921742","5.65677320389814",
                                   "4.98734530763173","7.24577992936045","0.575557527160308","-1.54237306813307","4.62886372795744",
                                   "3.44003538931392","3.28161683481803","-2.96262280595339","-0.927713915967914",
                                   "3.02521647168795","11.6754250956718"]

importance_normalized_poly_MTR1 = np.array([float(x) for x in importance_poly_MTR1])

## permuation importance: 47
importance_poly_MTR2 = ["-1.42144105291385","0.717574040596585","-0.215186133021021","-0.44481348449151","-2.19468027250312",
                             "2.46440912050683","6.38676466273507","0.0878616519784584","1.65000572766561","6.99395568354126",
                             "0.793055081108828","0.186085293543866","0.24528343715363","3.01176321605311","2.11066476932946",
                             "1.115114857407","3.84721868402782","0.097891981861173","2.33678376522609","4.08418531857987",
                             "-0.714474153810072","3.06879595834389","0.751231714405885","3.00801640740402","1.99029149025165",
                             "6.25770690019367","4.86558541506149","3.07682063794469","-1.89493590823099","2.90205399122083",
                             "0.087537307602402","6.37699624526546","-1.86556701192732","2.97137655757241","2.37529175397398",
                             "6.43567755141704","2.7282707098423","6.61035683872875","0.745155975385016","3.05362311809462",
                             "5.96905647108998","-1.93256049141098","-0.111880748306843","2.61668222982667","3.26582196893735",
                             "-3.54949134163774","9.06006406437666"]

importance_normalized_poly_MTR2 = np.array([float(x) for x in importance_poly_MTR2])

importance_poly_MTR3 = ["19.0145872217911","33.6079926584955","34.7932772076281","12.5841429120853"]
importance_normalized_poly_MTR3 = np.array([float(x) for x in importance_poly_MTR3])

#%%RF recursive
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(x_scaled_RF, y_scaled, test_size=0.15, random_state=42) #0.15
regressor_RF2 = MultiOutputRegressor(RandomForestRegressor(n_estimators = 100,max_depth=3,random_state=42))
regressor_RF2.fit(x_train, y_train)
y_pred_RF3 = regressor_RF2.predict(x_test)
test_r2_score_RF3 = r2_score(y_test, y_pred_RF3, multioutput='raw_values')
test_r2_score_RF3

mse_rf3 = mean_squared_error(y_test, y_pred_RF3, multioutput='raw_values')
mse_rf3

mae_rf3 = mean_absolute_error(y_test, y_pred_RF3, multioutput='raw_values')
mae_rf3

mape_rf3 = mean_absolute_percentage_error(y_test, y_pred_RF3, multioutput='raw_values')
mape_rf3


#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test2[:, 0], y_pred_RF3[:, 0], c=DOmg_l_Depth0, s=160)
plt.scatter(y_test2[:, 0], y_pred_RF3[:, 1], c=DOmg_l_Depth2, s=160)
plt.scatter(y_test2[:, 0], y_pred_RF3[:, 2], c=DOmg_l_Depth21, s=160)

#plt.scatter(y_test,y_pred_RF2, color='blue', label='RF')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red', linestyle='--')
plt.xlabel('True Value')
plt.title('Plot of RF')
plt.legend()
plt.tight_layout()
plt.show()

evaluation_results = []

# Evaluate model, karena ada 3 targets
mse_rf3 = mean_squared_error(y_test, y_pred_RF3)
rmse_rf3 = np.sqrt(mse_rf)
stdev_rf3 = np.std(y_pred_RF3)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_rf3,
    'RMSE': rmse_rf3,
    'Standard Deviation': stdev_rf3,
    'Best R-squared Score': test_r2_score_RF3.max()
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

from sklearn.inspection import permutation_importance
result_RF2_MTR = permutation_importance(regressor_RF2, x_test, y_test, n_repeats=4, random_state=42)
importance_RF2_MTR = result_RF2_MTR.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_RF2_MTR = 100.0 * (importance_RF2_MTR / importance_RF2_MTR.sum())  

#%%RF
 
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.15, random_state=42) #0.15
regressor_RF = MultiOutputRegressor(RandomForestRegressor(n_estimators = 100,max_depth=3,random_state=42))
regressor_RF.fit(x_train, y_train)
y_pred_RF2 = regressor_RF.predict(x_test)
test_r2_score_RF = r2_score(y_test, y_pred_RF2, multioutput='raw_values')
test_r2_score_RF

#RF_MTR
from sklearn.inspection import permutation_importance
result_RF_MTR = permutation_importance(regressor_RF, x_test, y_test, n_repeats=4, random_state=42)
importance_RF_MTR = result_RF_MTR.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_RF_MTR = 100.0 * (importance_RF_MTR / importance_RF_MTR.sum())

#%% plot 5
fig, axs = plt.subplots(1, 5, figsize=(30, 10))

#MLR
#axs[0].scatter(y_test,y_pred_multi)
axs[0].scatter(y_test2[:, 0], y_pred_multi[:, 0], c=DOmg_l_Depth0, s=160, label='DOmg_l_Depth0')
axs[0].scatter(y_test2[:, 1], y_pred_multi[:, 1], c=DOmg_l_Depth2, s=160, label='DOmg_l_Depth2')
axs[0].scatter(y_test2[:, 2], y_pred_multi[:, 2], c=DOmg_l_Depth21, s=160, label='DOmg_l_Depth21')
axs[0].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[0].set_title('Plot of MLR', size=30)
axs[0].set_xlabel('Observed DO (mg/L)', size=30)
axs[0].set_ylabel('Predicted DO (mg/L)', size=30)
axs[0].tick_params(axis='both', labelsize=30)

#Poly
#axs[1].scatter(y_test,y_pred)
axs[1].scatter(y_test2[:, 0], y_pred2[:, 0], c=DOmg_l_Depth0, s=160, label='DOmg_l_Depth0')
axs[1].scatter(y_test2[:, 1], y_pred2[:, 1], c=DOmg_l_Depth2, s=160, label='DOmg_l_Depth2')
axs[1].scatter(y_test2[:, 2], y_pred2[:, 2], c=DOmg_l_Depth21, s=160, label='DOmg_l_Depth21')
axs[1].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[1].set_title('Plot of Polynomial Regression', size=30)
axs[1].set_xlabel('Observed DO (mg/L)', size=30)
axs[1].set_ylabel('Predicted DO (mg/L)', size=30)
axs[1].tick_params(axis='both', labelsize=30)

#SVR
#axs[2].scatter(y_test,pred2)
axs[2].scatter(y_test2[:, 0], pred2[:, 0], c=DOmg_l_Depth0, s=160, label='DOmg_l_Depth0')
axs[2].scatter(y_test2[:, 1], pred2[:, 1], c=DOmg_l_Depth2, s=160, label='DOmg_l_Depth2')
axs[2].scatter(y_test2[:, 2], pred2[:, 2], c=DOmg_l_Depth21, s=160, label='DOmg_l_Depth21')
axs[2].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[2].set_title('Plot of SVR', size=30)
axs[2].set_xlabel('Observed DO (mg/L)', size=30)
axs[2].set_ylabel('Predicted DO (mg/L)', size=30)
axs[2].tick_params(axis='both', labelsize=30)

#RF
axs[3].scatter(y_test2[:, 0], y_pred_RF3[:, 0], c=DOmg_l_Depth0, s=160, label='DOmg_l_Depth0')
axs[3].scatter(y_test2[:, 1], y_pred_RF3[:, 1], c=DOmg_l_Depth2, s=160, label='DOmg_l_Depth2')
axs[3].scatter(y_test2[:, 2], y_pred_RF3[:, 2], c=DOmg_l_Depth21, s=160, label='DOmg_l_Depth21')
#axs[4].scatter(y_test,y_pred_RF2)
axs[3].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[3].set_title('Plot of RF', size=30)
axs[3].set_xlabel('Observed DO (mg/L)', size=30)
axs[3].set_ylabel('Predicted DO (mg/L)', size=30)
axs[3].set_xlabel('Observed DO (mg/L)', size=30)
axs[3].tick_params(axis='both', labelsize=30)

#XGBoost-GBM
#axs[3].scatter(y_test,multi_reg_pred_GBM)
axs[4].scatter(y_test2[:, 0], multi_reg_pred_GBM[:, 0], c=DOmg_l_Depth0, s=160, label='DOmg_l_Depth0')
axs[4].scatter(y_test2[:, 1], multi_reg_pred_GBM[:, 1], c=DOmg_l_Depth2, s=160, label='DOmg_l_Depth2')
axs[4].scatter(y_test2[:, 2], multi_reg_pred_GBM[:, 2], c=DOmg_l_Depth21, s=160, label='DOmg_l_Depth21')
axs[4].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[4].set_title('Plot of XGBoost Regression', size=30)
axs[4].set_xlabel('Observed DO (mg/L)', size=30)
axs[4].set_ylabel('Predicted DO (mg/L)', size=30)
axs[4].tick_params(axis='both', labelsize=30)
axs[4].legend()


plt.legend(fontsize="25")
plt.tight_layout()
plt.show()

#%%feature importance

#df_fi2['SVR_MTR']=100*df_fi2['SVR_MTR']/df_fi2['SVR_MTR'].sum()
#fig, ax = plt.subplots()
#ax.legend()

#plot
#ind = np.arange(len(filtered_df))
#width = 0.2

#ax = filtered_df.plot(kind='barh', width=1.0)
#ax.barh(ind, filtered_df['values_MLR'], width, color='blue', label='Values_MLR')
#ax.barh(ind + width, filtered_df['values_XG'], width, color='green', label='Values_XG')
#ax.barh(ind + width + width, filtered_df['value_SVR'], width, color='red', label='Values_SVR')
#ax.barh(ind + width + width + width, filtered_df['value_RF'], width, color='orange', label='Values_RF')
#ax.barh(ind + width + width + width + width, filtered_df['value_Poly'], width, color='black', label='values_Poly')
#ax.set(yticks=ind + width + width , yticklabels=filtered_df.feature, ylim=[4*width - 1, len(filtered_df)])



#%% Plot colormaps feature importance MTR (finish)

mlr_temp=np.zeros(np.shape(importance_RF_MTR)) #yang masih 71
XG_temp=np.zeros(np.shape(importance_RF_MTR))
SVR_temp=np.zeros(np.shape(importance_RF_MTR))
RF_temp=np.zeros(np.shape(importance_RF_MTR))
poly_temp=np.zeros(np.shape(importance_RF_MTR))

data3 = {
  'Features': df_feat_new.columns,
    'MLR_MTR': mlr_temp,
    'XG_MTR' : XG_temp,
    'SVR_MTR': SVR_temp,
    'RF_MTR' : RF_temp,
    'Polynomial_MTR':poly_temp, #,
}

#create dataframe
df_fi = pd.DataFrame(data3,index=df_feat_new.columns)

df_fi.loc[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth1","Chl-Flu_Depth2"],['MLR_MTR']]=np.reshape(importance_normalized_MLR_MTR,(4,1))

df_fi.loc[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth13","Turb-M_Depth38","WaterTemp_depth0","Chl-Flu_Depth2","EC25_Depth2","Turb-M_Depth27","Turb-M_Depth33","Chl-Flu_Depth0",
"Turb-M_Depth56","Turb-M_Depth29","Turb-M_Depth26","Chl-Flu_Depth41","Turb-M_Depth35","Turb-M_Depth59","Salinity_Depth31","Turb-M_Depth39","Chl-Flu_Depth19","Chl-Flu_Depth37",
"Turb-M_Depth47","Chl-Flu_Depth1","Chl-Flu_Depth38","Turb-M_Depth51","Turb-M_Depth49","Salinity_Depth1","Turb-M_Depth54","Turb-M_Depth53","Turb-M_Depth24","Turb-M_Depth36",
"Salinity_Depth37","Turb-M_Depth22","Turb-M_Depth52","Chl-Flu_Depth50","Turb-M_Depth37","Turb-M_Depth23","Turb-M_Depth46","Turb-M_Depth41","Turb-M_Depth32","Turb-M_Depth43",
"Turb-M_Depth45","Turb-M_Depth31","Turb-M_Depth42","Chl-Flu_Depth58","Turb-M_Depth58","Turb-M_Depth55","Salinity_Depth26","Chl-Flu_Depth36","Turb-M_Depth25","Turb-M_Depth34",
"EC25_Depth1","Chl-Flu_Depth53","Turb-M_Depth44","Chl-Flu_Depth12"],['XG_MTR']]=np.reshape(importance_normalized_GBM_MTR,(54,1))

df_fi.loc[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth13","WaterTemp_depth0","Chl-Flu_Depth2","Turb-M_Depth59",
                      "Turb-M_Depth39","Turb-M_Depth47","Chl-Flu_Depth1","Turb-M_Depth49","Turb-M_Depth36",
                      "Turb-M_Depth48","Salinity_Depth37","Turb-M_Depth37","Turb-M_Depth23","Turb-M_Depth41",
                      "Turb-M_Depth43","Turb-M_Depth45","Turb-M_Depth31","Turb-M_Depth58","Turb-M_Depth44","Chl-Flu_Depth22","Chl-Flu_Depth16",
                      "Turb-M_Depth60","Chl-Flu_Depth14","Turb-M_Depth28"],['SVR_MTR']]=np.reshape(importance_normalized_SVR_MTR,(26,1))

df_fi.loc[["WaterTemp_depth1","Salinity_Depth0","Turb-M_Depth38","WaterTemp_depth0","Chl-Flu_Depth2","Turb-M_Depth27","Turb-M_Depth33","Chl-Flu_Depth0",
                     "Chl-Flu_Depth41","Chl-Flu_Depth41","Turb-M_Depth35","Salinity_Depth31","Turb-M_Depth39","Chl-Flu_Depth37","Turb-M_Depth47","Chl-Flu_Depth1",
                     "Chl-Flu_Depth38","Turb-M_Depth49","Salinity_Depth1","Turb-M_Depth54","Turb-M_Depth53","Turb-M_Depth24","Turb-M_Depth36","Turb-M_Depth48",
                     "Turb-M_Depth57","Chl-Flu_Depth49","Salinity_Depth37","Turb-M_Depth22","Turb-M_Depth52","Turb-M_Depth37","Turb-M_Depth23",
                     "Turb-M_Depth46","Turb-M_Depth41","Turb-M_Depth32","Turb-M_Depth43","Turb-M_Depth45","Turb-M_Depth31","Turb-M_Depth42",
                     "Chl-Flu_Depth58","Salinity_Depth26","Turb-M_Depth34","EC25_Depth1","Chl-Flu_Depth53","Chl-Flu_Depth12","Chl-Flu_Depth22",
                     "Chl-Flu_Depth16","Chl-Flu_Depth48","Salinity_Depth30","Turb-M_Depth60","Chl-Flu_Depth42","Chl-Flu_Depth14","Chl-Flu_Depth34","Turb-M_Depth28"],['RF_MTR']]=np.reshape(importance_normalized_RF2_MTR,(53,1))

df_fi.loc[["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth1","Chl-Flu_Depth2"],['Polynomial_MTR']]=np.reshape(importance_normalized_poly_MTR3,(4,1))

#sorting
sorted_df2 = df_fi.sort_values(by=['MLR_MTR', 'XG_MTR','SVR_MTR','RF_MTR','Polynomial_MTR'], 
                              ascending=[False, False, False, False, False])

sorted_df2.to_csv('sorted_df_finish.csv') 

#diubah ke nilai 0

filtered_df2=sorted_df2.iloc[:4]
#filter
#filtered_df = sorted_df2[(sorted_df2['XG_MTR'] > 0.72) | (sorted_df2['SVR_MTR'] > 0.72) | (sorted_df2['RF_MTR'] > 0.72)]

import seaborn as sns
sns.heatmap(filtered_df2, annot=False)

ax = filtered_df2.plot(kind='barh', width=0.9,figsize=(20, 14),fontsize="40")
ax.legend(prop={'size': 50})
plt.rcParams.update({
    'font.size': 50,          # General font size
    'axes.titlesize': 17,     # Title font size
    'axes.labelsize': 50,  
    'xtick.labelsize': 50,    # X-tick font size
    'ytick.labelsize': 50     # Y-tick font size
})

#%% LSTM-MTR
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import tensorflow.keras.backend as K

#non-recursive
#inputs=df_feat_new #seluruh 71 inputs

#for recursive input dari RF
inputs=df_feat_new[["WaterTemp_depth1","Salinity_Depth0","Turb-M_Depth38","WaterTemp_depth0","Chl-Flu_Depth2","Turb-M_Depth27","Turb-M_Depth33","Chl-Flu_Depth0",
                     "Chl-Flu_Depth41","Chl-Flu_Depth41","Turb-M_Depth35","Salinity_Depth31","Turb-M_Depth39","Chl-Flu_Depth37","Turb-M_Depth47","Chl-Flu_Depth1",
                     "Chl-Flu_Depth38","Turb-M_Depth49","Salinity_Depth1","Turb-M_Depth54","Turb-M_Depth53","Turb-M_Depth24","Turb-M_Depth36","Turb-M_Depth48",
                     "Turb-M_Depth57","Chl-Flu_Depth49","Salinity_Depth37","Turb-M_Depth22","Turb-M_Depth52","Turb-M_Depth37","Turb-M_Depth23",
                     "Turb-M_Depth46","Turb-M_Depth41","Turb-M_Depth32","Turb-M_Depth43","Turb-M_Depth45","Turb-M_Depth31","Turb-M_Depth42",
                     "Chl-Flu_Depth58","Salinity_Depth26","Turb-M_Depth34","EC25_Depth1","Chl-Flu_Depth53","Chl-Flu_Depth12","Chl-Flu_Depth22",
                     "Chl-Flu_Depth16","Chl-Flu_Depth48","Salinity_Depth30","Turb-M_Depth60","Chl-Flu_Depth42","Chl-Flu_Depth14","Chl-Flu_Depth34","Turb-M_Depth28"]]
output=df_DO_new #'DOmg_l_Depth0', 'DOmg_l_Depth2', 'DOmg_l_Depth21

# Normalize
scaler = MinMaxScaler()
inputs_normalized = scaler.fit_transform(inputs)

X = np.reshape(inputs_normalized, (inputs_normalized.shape[0], 1, inputs_normalized.shape[1]))
y = output.values

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define input shape dynamically based on X_train
input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

print("Input Shape:", input_shape) #features: 549

# Create the LSTM model
input_layer = Input(shape=input_shape)  # Dynamically includes all features
lstm_layer = LSTM(50, return_sequences=False)(input_layer)
output1 = Dense(1, activation='linear', name='output1')(lstm_layer)
output2 = Dense(1, activation='linear', name='output2')(lstm_layer)
output3 = Dense(1, activation='linear', name='output3')(lstm_layer)

model = Model(inputs=input_layer, outputs=[output1, output2, output3])
model.compile(
    optimizer='adam',
    loss='mae',
    metrics={
        'output1': ['mae'], 
        'output2': ['mae'],  
        'output3': ['mae']
    }
)

#%%for r2 metrics
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - ss_res / (ss_tot + K.epsilon())
    return r2

model = Model(inputs=input_layer, outputs=[output1, output2, output3])
model.compile(
    optimizer='adam',
    loss='mape',  # Use Mean Absolute Percentage Error for loss
    metrics={
        'output1': [r2_metric], 
        'output2': [r2_metric],  
        'output3': [r2_metric]
    }
)
#%%

# Train the model
history = model.fit(
    X_train,
    [y_train[:, 0], y_train[:, 1],y_train[:, 2]],
    validation_data=(X_test, [y_test[:, 0], y_test[:, 1],y_test[:, 2]]),
    epochs=50,
    batch_size=32
)

## Use MSE as the loss function for all outputs
model.summary()

#%% SHAP MTR - MLR
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
shap.initjs()


#inputs = x_scaled_mlr #["WaterTemp_depth1","Salinity_Depth0","Chl-Flu_Depth1","Chl-Flu_Depth2"]
#outputs = y_scaled

x_train, x_test, y_train, y_test = train_test_split(x_scaled_mlr, y_scaled, test_size=0.15, random_state=42) #0.15
reg = LinearRegression()
m_reg=reg.fit(x_train, y_train)
y_pred_multi=m_reg.predict(x_test)

masker = shap.maskers.Independent(x_train)
explainer = shap.Explainer(m_reg, masker=masker)

shap_values = explainer(x_test)
shap_values_output1 = shap_values[..., 0]
shap_values_output2 = shap_values[..., 1]
shap_values_output3 = shap_values[..., 2]

shap.summary_plot(shap_values_output1, x_test)
shap.summary_plot(shap_values_output2, x_test)
shap.summary_plot(shap_values_output3, x_test)

# Compute SHAP values for the 3 output
#shap_values_output1 = explainer([x_test], output_indices=[0])
#shap_values_output2 = explainer([x_test], output_indices=[1])
#shap_values_output3 = explainer([x_test], output_indices=[2])


