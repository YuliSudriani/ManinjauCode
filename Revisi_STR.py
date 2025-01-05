# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:51:31 2024

@author: informatikalipi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,median_absolute_error
from IPython import get_ipython
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
#import statsmodels.api as sm
#import sklearn.metrics as metrics
import csv
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from statsmodels.iolib.summary2 import summary_col
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'
from pandas import DataFrame,Series

#%% data pre-processing

df_maninjau = pd.read_csv('D:/2nd_ManinjauLAke/Data_imputation/STRDatasets.csv')  #CopyManinjau_New_Dataset2
df_maninjau.columns
df_maninjau.shape
df_maninjau.head()
df_maninjau.info()
df_maninjau.dtypes
df_maninjau.describe()
df_maninjau.describe().T
df_maninjau.describe(include='all')
df_maninjau.isnull().sum()

#delete columm which no have value/just outliers
df_maninjau.drop(['id', 'sitename','t13','t14','t17','t18','t19','t20','t23','t24','t29','t30',
                   't31','t32', 't33','t34','t35','t36','t37','t38','t39','t40'
                   ,'t41','t42','wl_temp','wl','ph_temp','ph'
                   ,'ntu','ntu_temp','voltage','checkstring VALUES', 'rf',
                   'ws', 'wd','do_temp', 'do', 'domg_temp'],axis=1,inplace=True) 

#select period of time
df_maninjau['datelog'] = pd.to_datetime(df_maninjau['datelog'])
start_date = '2017-03-22 08:10:14' #40000-56400 data
end_date = '2017-06-12 11:20:16'

#start_date = '2017-03-01 08:10:14' #40000-56400 data
#end_date = '2018-03-01 11:20:16'
filterdata = (df_maninjau['datelog'] > start_date) & (df_maninjau['datelog'] <= end_date)
df_maninjau2 = df_maninjau.loc[filterdata]

#cek Missing data for do,T0,T1,T2,ws from selected data time
percent_missing= df_maninjau2.isnull().sum() * 100 / len(df_maninjau2)
missing_value_df = pd.DataFrame({'column_name': df_maninjau2.columns, 'percent_missing': percent_missing})
missing_value_df

#sub plot to check values
df_datelog  = pd.DataFrame(df_maninjau2, columns= ["datelog"])
df_plot  = pd.DataFrame(df_maninjau2, columns= ["domg"])
df_plot2 = pd.DataFrame(df_maninjau2, columns= ["t0"]) 
df_plot3 = pd.DataFrame(df_maninjau2, columns= ["t1"])
fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8,15))
ax1.plot(df_datelog,df_plot)
ax2.plot(df_datelog,df_plot2)
ax3.plot(df_datelog,df_plot3)
ax1.title.set_text('domg')
ax2.title.set_text('t0') 
ax3.title.set_text('t1')

#count value with domg value more than 20 as extreme value or 0 value
Total_Count_domg = (df_maninjau2['domg'] > 20).sum()
print(Total_Count_domg) #output: 107

Total_Count_domg0 = (df_maninjau2['domg'] < 0.5).sum()
print(Total_Count_domg0) #output 43

#%%feature selection
# apply feature matrix
df_featureRealtime= df_maninjau2.drop(["domg","datelog"],axis=1) #
df_featureRealtime.shape
df_featureRealtime.head()
df_featureRealtime.columns

df_outputRealtime = df_maninjau2[["domg"]]

#using pearson correlation heatmap
import seaborn as sns

df_maninjau2.columns
df_maninjau3= df_maninjau2.drop(["datelog","domg"],axis=1) #
#plt.figure(figsize=(20,20))
cor=df_maninjau3.corr()
f, ax = plt.subplots(figsize=(40, 20))
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r, annot_kws={"size":19})
cbar = ax.collections[0].colorbar
ax.set_xticklabels(ax.get_xticklabels(), fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.show()

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

corr_high = correlation(df_featureRealtime, 0.8)
df_feat_realTime = df_featureRealtime.drop(corr_high,axis=1)
df_feat_realTime.columns
df_feat_realTime.shape #(16400,12)
df_feat_realTime .T
df_feat_realTime.to_csv('data_Str_corr.csv') 

#%%normalisation data
from sklearn.preprocessing import MinMaxScaler
# Membuat objek MinMaxScaler untuk fitur-fitur
scaler_x = MinMaxScaler()

# Menggunakan scaler_X untuk mengubah skala fitur-fitur
x_scaled_realTime = scaler_x.fit_transform(df_feat_realTime)
x_scaled_realTime = pd.DataFrame(x_scaled_realTime)

# Membuat objek MinMaxScaler untuk target (opsional, jika diperlukan)
scaler_y = MinMaxScaler()

# Menggunakan scaler_y untuk mengubah skala target
y_scaled_realTime = scaler_y.fit_transform(df_outputRealtime)
y_scaled_realTime = pd.DataFrame(y_scaled_realTime)

# Membuat objek MinMaxScaler untuk target (opsional, jika diperlukan)
scaler_y = MinMaxScaler()

# Create a list to store the results for each feature scenario
results_list = []

#%%split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled_realTime, y_scaled_realTime, test_size=0.15, random_state=42) #0.15

#%% Regression Models
#MLR
from sklearn.linear_model import LinearRegression
reg_realTime = LinearRegression()
m_reg_realTime=reg_realTime.fit(x_train, y_train)
y_pred_realTime=m_reg_realTime.predict(x_test)
test_r2_score_MLRrealTime = r2_score(y_test, y_pred_realTime)
test_r2_score_MLRrealTime

evaluation_results = []

#Evaluate model
mse_MLRrealTime = mean_squared_error(y_test, y_pred_realTime)
rmse_MLR_real = np.sqrt(mse_MLRrealTime)
stdev_MLR_real = np.std(y_pred_realTime)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_MLRrealTime,
    'RMSE': rmse_MLR_real,
    'Standard Deviation': stdev_MLR_real,
    'Best R-squared Score': test_r2_score_MLRrealTime
})

# Create a DataFrame from the evaluation_results
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred_realTime)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
plt.xlabel('Observed DO (mg/L)', size=20)
plt.ylabel('Predicted DO (mg/L)', size=20)
plt.title('Actual vs Predicted for DO (mg/L) on surface waters (0 m) for Polynomial Regression', size=20)
plt.show()

#boxplot
residuals1 = y_test - y_pred_realTime
# Create a boxplot of the residuals
plt.boxplot(residuals1)
plt.title('Boxplot of Regression Residuals')
plt.xlabel('Residuals')
plt.ylabel('Value')
plt.show()

print(f"intercept: {m_reg_realTime.intercept_}")
print(f"coefficients: {m_reg_realTime.coef_}")

#%%feature importance
from sklearn.inspection import permutation_importance

#MLR
result_MLR_real = permutation_importance(m_reg_realTime, x_test, y_test, n_repeats=4, random_state=42)
importance_MLR = result_MLR_real.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_MLR = 100.0 * (importance_MLR / importance_MLR.sum())

#%% XGBoost
from xgboost import XGBRegressor
regressor_XGBoost_realTime=XGBRegressor()
regressor_XGBoost_realTime2=regressor_XGBoost_realTime.fit(x_train,y_train)
y_pred_XGBoost_RealTime=regressor_XGBoost_realTime2.predict(x_test)
test_r2_score_GBMRealtime = r2_score(y_test, y_pred_XGBoost_RealTime)
test_r2_score_GBMRealtime

evaluation_results = []

# Evaluate model
mse_GBM_realTime = mean_squared_error(y_test, y_pred_XGBoost_RealTime)
rmse_GBM_realTime = np.sqrt(mse_GBM_realTime)
stdev_GBM_realTime = np.std(y_pred_XGBoost_RealTime)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_GBM_realTime,
    'RMSE': rmse_GBM_realTime,
    'Standard Deviation': stdev_GBM_realTime,
    'Best R-squared Score': test_r2_score_GBMRealtime
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred_XGBoost_RealTime)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
plt.xlabel('Observed DO (mg/L)', size=20)
plt.ylabel('Predicted DO (mg/L)', size=20)
plt.title('Actual vs Predicted for DO (mg/L) on surface waters (0 m) for XGBoost Regression', size=20)
plt.show()

#XGBoost
result_XG_real = permutation_importance(regressor_XGBoost_realTime2, x_test, y_test, n_repeats=4, random_state=42)
importance_XG = result_XG_real.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_XG = 100.0 * (importance_XG / importance_XG.sum())

#boxplot
y_pred_XGBoost_RealTime = y_pred_XGBoost_RealTime.reshape((2460, 1))
residuals2 = y_test - y_pred_XGBoost_RealTime

# Create a boxplot of the residuals
plt.boxplot(residuals2)
plt.title('Boxplot of Regression Residuals')
plt.xlabel('Residuals')
plt.ylabel('Value')
plt.show()


#%%SVR
from sklearn.svm import SVR
svr_realTime=SVR(kernel='rbf')
y_reshaped = np.ravel(y_train)
svr_realTime2=svr_realTime.fit(x_train,y_reshaped)
pred_SVR_realTime=svr_realTime2.predict(x_test)
test_r2_score_SVR_realTime = r2_score(y_test, pred_SVR_realTime)
test_r2_score_SVR_realTime

evaluation_results = []

# Evaluate model
mse_SVR_realTime = mean_squared_error(y_test, pred_SVR_realTime)
rmse_SVR_realTime = np.sqrt(mse_SVR_realTime)
stdev_SVR_realTime = np.std(pred_SVR_realTime)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_SVR_realTime,
    'RMSE': rmse_SVR_realTime,
    'Standard Deviation': stdev_SVR_realTime,
    'Best R-squared Score': test_r2_score_SVR_realTime
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test,pred_SVR_realTime)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
plt.xlabel('Observed DO (mg/L)', size=20)
plt.ylabel('Predicted DO (mg/L)', size=20)
plt.title('Actual vs Predicted for DO (mg/L) on surface waters (0 m) for SVR', size=20)
plt.show()

#SVR
result_SVR = permutation_importance(svr_realTime2, x_test, y_test, n_repeats=4, random_state=42)
importance_SVR = result_SVR.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_SVR = 100.0 * (importance_SVR / importance_SVR.sum())

#boxplot
pred_SVR_realTime = pred_SVR_realTime.reshape((2460, 1))
residuals3 = y_test - pred_SVR_realTime

# Create a boxplot of the residuals
plt.boxplot(residuals3)
plt.title('Boxplot of Regression Residuals')
plt.xlabel('Residuals')
plt.ylabel('Value')
plt.show()


#%%random forest
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

regressor_RF_realTime = RandomForestRegressor(n_estimators = 50, random_state = 42)
regressor_RF_realTime=regressor_RF_realTime.fit(x_train, y_train)
y_pred_RF_realTime = regressor_RF_realTime.predict(x_test)
test_r2_score_RF_realTime = r2_score(y_test, y_pred_RF_realTime)
test_r2_score_RF_realTime
         
evaluation_results = []

# Evaluate model
mse_rf_realtime = mean_squared_error(y_test, y_pred_RF_realTime)
rmse_rf_realtime = np.sqrt(mse_rf_realtime)
stdev_rf_realtime = np.std(y_pred_RF_realTime)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_rf_realtime,
    'RMSE': rmse_rf_realtime,
    'Standard Deviation': stdev_rf_realtime,
    'Best R-squared Score': test_r2_score_RF_realTime
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred_RF_realTime)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
plt.xlabel('Observed DO (mg/L)', size=20)
plt.ylabel('Predicted DO (mg/L)', size=20)
plt.title('Actual vs Predicted for DO (mg/L) on surface waters (0 m) for RF', size=20)
plt.show()

#RF
result_RF = permutation_importance(regressor_RF_realTime, x_test, y_test, n_repeats=4, random_state=42)
importance_RF = result_RF.importances_mean

# Normalize feature importances to sum up to 100% and assign column names
importance_normalized_RF = 100.0 * (importance_RF / importance_RF.sum())

#boxplot
y_pred_RF_realTime = y_pred_RF_realTime.reshape((2460, 1))
residuals4 = y_test - y_pred_RF_realTime

# Create a boxplot of the residuals
plt.boxplot(residuals4)
plt.title('Boxplot of Regression Residuals')
plt.xlabel('Residuals')
plt.ylabel('Value')
plt.show()


#%% Poly Regression
from sklearn.inspection import permutation_importance
poly_features = PolynomialFeatures(degree=2,include_bias=False)
xtrain_poly = poly_features.fit_transform(x_train)
xtest_poly  = poly_features.fit_transform(x_test)

## build regression model
reg2 = LinearRegression()
reg2.fit(xtrain_poly,y_train)
score_train=reg2.score(xtrain_poly,y_train)
score_test=reg2.score(xtest_poly,y_test)
y_predict_poly = reg2.predict(xtest_poly)

print(f"intercept: {reg2.intercept_}")
print(f"coefficients: {reg2.coef_}")

## permutation
def get_permuted_score(idx,xtrain,xtest,ytrain,ytest):
    xtrain_permuted = xtrain.copy()
    xtrain_permuted.iloc[:, idx] = np.random.permutation(xtrain_permuted.iloc[:, idx])   # permute one column
    xtest_permuted = xtest.copy()
    xtest_permuted.iloc[:, idx] = np.random.permutation(xtest_permuted.iloc[:, idx])   # permute one column

    xtrain2_poly = poly_features.fit_transform(xtrain_permuted)
    xtest2_poly  = poly_features.fit_transform(xtest_permuted)

    ## build regression model
    reg2 = LinearRegression()
    reg2.fit(xtrain2_poly,ytrain)

    ## score
    sc2_train = reg2.score(xtrain2_poly,ytrain)
    sc2_test  = reg2.score(xtest2_poly,ytest)

    return sc2_train,sc2_test

FI_permuted = np.zeros((2,12))
for idx in range(12):
    FI_permuted[:,idx] = get_permuted_score(idx,x_train,x_test,y_train,y_test)
    
test_r2_score_polyRealTime = r2_score(y_test, y_predict_poly) 
test_r2_score_polyRealTime

evaluation_results = []

# Evaluate model
mse_Poly_realtime = mean_squared_error(y_test, y_predict_poly)
rmse_Poly_realtime = np.sqrt(mse_Poly_realtime)
stdev_Poly_realtime = np.std(y_predict_poly)
evaluation_results.append({
    'Model': 'rf',
    'MSE': mse_Poly_realtime,
    'RMSE': rmse_Poly_realtime,
    'Standard Deviation': stdev_Poly_realtime,
    'Best R-squared Score': test_r2_score_polyRealTime
})

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)

#plot the graphic
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_predict_poly)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
plt.xlabel('Observed DO (mg/L)', size=20)
plt.ylabel('Predicted DO (mg/L)', size=20)
plt.title('Actual vs Predicted for DO (mg/L) on surface waters (0 m) for Polynomial Regression', size=20)
plt.show()

#the calcullation in excel for n_repeats=4
## permuation importance
importance_normalized_poly = ['11.5210307','58.59657748','8.901643997','18.47016554',
                              '1.178824747','	0.1195971219','0.6389020623','0.03261734658',
                              '0.4023843285','-0.3225663838','0.1383335516','	0.3224895081']
importance_normalized_poly = np.array([float(x) for x in importance_normalized_poly])

#%% plot observasi dan prediksi
#plot multiregression
fig, axs = plt.subplots(1, 5, figsize=(30, 10))

#MLR
axs[0].scatter(y_test,y_pred_realTime)
axs[0].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[0].set_title('Plot of MLR', size=30)
axs[0].set_xlabel('Observed DO (mg/L)', size=30)
axs[0].set_ylabel('Predicted DO (mg/L)', size=30)
axs[0].tick_params(axis='both', labelsize=30)

#polynomial
axs[1].scatter(y_test,y_predict_poly)
axs[1].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[1].set_title('Plot of Polynomial Regression', size=30)
axs[1].set_xlabel('Observed DO (mg/L)', size=30)
axs[1].set_ylabel('Predicted DO (mg/L)', size=30)
axs[1].tick_params(axis='both', labelsize=30)

#SVR
axs[2].scatter(y_test,pred_SVR_realTime)
axs[2].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[2].set_title('Plot of SVR', size=30)
axs[2].set_xlabel('Observed DO (mg/L)', size=30)
axs[2].set_ylabel('Predicted DO (mg/L)', size=30)
axs[2].tick_params(axis='both', labelsize=30)

#RF
axs[3].scatter(y_test,y_pred_RF_realTime)
axs[3].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[3].set_title('Plot of RF', size=30)
axs[3].set_xlabel('Observed DO (mg/L)', size=30)
axs[3].set_ylabel('Predicted DO (mg/L)', size=30)
axs[3].tick_params(axis='both', labelsize=30)

#XGBoost
axs[4].scatter(y_test,y_pred_XGBoost_RealTime,label='DOmg_l_Depth2')
axs[4].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='tab:red')
axs[4].set_title('Plot of XGBoost Regression', size=30)
axs[4].set_xlabel('Observed DO (mg/L)', size=30)
axs[4].set_ylabel('Predicted DO (mg/L)', size=30)
axs[4].tick_params(axis='both', labelsize=30)

# Adjust layout and display the plot
plt.legend(fontsize="30")
plt.tight_layout()
plt.show()

#%% evaluation metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as metrics

def regression_error(y_test,y_pred):
    mae=metrics.mean_absolute_error(y_test,y_pred) 
    mse=metrics.mean_squared_error(y_test,y_pred) 
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    r2=metrics.r2_score(y_test,y_pred)
    
    print('MAE: ', round(mae,4))
    print('MSE: ', round(mse,4))
    print('MAPE: ', round(mape,4))
    print('r2: ', round(r2,4))    
    return 

MLR=regression_error(y_test,y_pred_realTime)
poly=regression_error(y_test,y_predict_poly)
SVR=regression_error(y_test,pred_SVR_realTime)
RF=regression_error(y_test,y_pred_RF_realTime)
XGB=regression_error(y_test,y_pred_XGBoost_RealTime)

#%%feature importance
data = {
    'Features': df_feat_realTime.columns,
    'MLR': importance_normalized_MLR,
    'XGBoost_Regression': importance_normalized_XG,
    'SVR': importance_normalized_SVR,
    'RFR':importance_normalized_RF,
    'Polynomial_Regression':importance_normalized_poly,
}

#create dataframe
df_fi = pd.DataFrame(data)


sorted_df = df_fi.sort_values(by=['MLR', 'XGBoost_Regression','SVR','RFR','Polynomial_Regression'], 
                              ascending=[False, False, False, False, False])

#filter
filtered_df = sorted_df[(sorted_df['MLR'] > 0.72) | (sorted_df['XGBoost_Regression'] > 0.72)
              | (sorted_df['SVR'] > 0.72) | (sorted_df['RFR'] > 0.72)
              | (sorted_df['Polynomial_Regression'] > 0.72)]

#plot
#ind = np.arange(len(filtered_df))
#width = 0.2

fig, ax = plt.subplots(figsize=(12, 10))
ax = filtered_df.set_index(data['Features']).plot(kind='barh', width=0.8, fontsize='20')

#ax = filtered_df.plot(kind='barh', width=1.0)
#ax.barh(ind, filtered_df['values_MLR'], width, color='blue', label='Values_MLR')
#ax.barh(ind + width, filtered_df['values_XG'], width, color='green', label='Values_XG')
#ax.barh(ind + width + width, filtered_df['value_SVR'], width, color='red', label='Values_SVR')
#ax.barh(ind + width + width + width, filtered_df['value_RF'], width, color='orange', label='Values_RF')
#ax.barh(ind + width + width + width + width, filtered_df['value_Poly'], width, color='black', label='values_Poly')
#ax.set(yticks=ind + width + width , yticklabels=filtered_df.feature, ylim=[4*width - 1, len(filtered_df)])
ax.legend(prop={'size': 20})

#plot.legend(loc=2, prop={'size': 6})

plt.show()


#%% boxplot

residuals_df = {
    'Model 1': residuals1,
    'Model 2': residuals2,
    'Model 3': residuals3,
    'Model 4': residuals4,
}

fig, ax = plt.subplots()
df = pd.DataFrame(np.random.normal(size=(2460,4)), columns=residuals_df)
df.boxplot(ax=ax, positions=[2,3,4,6], notch=True, bootstrap=5000)
ax.set_xticks(range(10))
ax.set_xticklabels(range(10))
plt.show()

#%%MLR SHAP

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
shap.initjs()


#MLR
from sklearn.linear_model import LinearRegression
#reg_realTime = LinearRegression()
#m_reg_realTime=reg_realTime.fit(x_train, y_train)
masker = shap.maskers.Independent(x_test)

variables_mlr = []
explainer = shap.Explainer(
    m_reg_realTime, masker=masker, feature_names=x_train.columns, algorithm="linear"
)

sa = explainer(x_test)
#plt.title('Feature Importance: SHAP-MLR ', fontsize=20)
plt.rcParams.update({
    'font.size': 50,          # General font size
    'axes.titlesize': 19,     # Title font size
    'axes.labelsize': 50,     # Axis label font size
    'xtick.labelsize': 50,    # X-tick font size
    'ytick.labelsize': 50     # Y-tick font size
})
shap.summary_plot(sa, x_test)

#variable 12= t0', 't1', 't2', 't3', 't15', 't16', 't21', 't22', 't25', 't26', 't27','t28'

#%%XGBoost SHAP
regressor_XGBoost_realTime=XGBRegressor()
regressor_XGBoost_realTime2=regressor_XGBoost_realTime.fit(x_train,y_train)
y_pred_XGBoost_RealTime=regressor_XGBoost_realTime2.predict(x_test)

explainer_xg = shap.Explainer(
    regressor_XGBoost_realTime2, masker=masker, feature_names=x_train.columns
)

sa_xg = explainer_xg(x_test)
#plt.title('Feature Importance: SHAP-XGBoost Regression', fontsize=20)
plt.rcParams.update({
    'font.size': 50,          # General font size
    'axes.titlesize': 17,     # Title font size
    'axes.labelsize': 50,     # Axis label font size
    'xtick.labelsize': 50,    # X-tick font size
    'ytick.labelsize': 50     # Y-tick font size
})
shap.summary_plot(sa_xg, x_test)

#%% polynom SHAP : belum bisa, tanya mbak Nida

explainer_poly = shap.Explainer(
    reg2, masker=masker, feature_names=x_train.columns
)

sa_poly = explainer_poly(x_test)
plt.title('Feature Importance: SHAP-XGBoost Regression', fontsize=20)
plt.rcParams.update({
    'font.size': 50,          # General font size
    'axes.titlesize': 17,     # Title font size
    'axes.labelsize': 50,     # Axis label font size
    'xtick.labelsize': 50,    # X-tick font size
    'ytick.labelsize': 50     # Y-tick font size
})
shap.summary_plot(sa_poly, x_test)


#sources: https://stackoverflow.com/questions/65621290/shap-explainer-constructor-error-asking-for-undocumented-positional-argument







