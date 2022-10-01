#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 18:04:08 2022

@author: Lynn Menchaca

"""

"""
The purpose of this document is to train a machine learning model to predict which team will win

Processing steps:
    - get data ready for machine learning models
    - perform feature selection on the data
    - train multiple machine learning models with the data
    - pick the best model and train with data
    - test model with new nfl week games


Resources:
youtube -> Krish Naik -> Live-Features Selection-Various Techniques To Select Features Day7
https://www.youtube.com/watch?v=k-EpAMjw6AE

youtube -> Krish Naik -> Live Machine Learning playlist
https://www.youtube.com/playlist?list=PLZoTAELRMXVPjaAzURB77Kz0YXxj65tYz

Quarterback = QB
Offense = Off
Defense = Def


#Clean data
# weighted mean of all ranks
# add dummy values
# feature selection for stadium
# feature selection for offense stats
#   make all stats the same scale?
# feature selection for defense stats
# feature selection for quarterback stats
# build models

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression


# Download data file
data_file_path = '/Users/lynnpowell/Documents/DS_Projects/NFL_Analysis/Data_Sets/'
nfl = pd.read_csv(data_file_path+'nfl_cleaned_data.csv', parse_dates=['DateTime'])

print(nfl.columns)

##### Clean Data ########

# Columns to drop:
drop_col = ['Season', 'Week', 'DateTime', 'HomeTeam', 'HomePts',
       'HomeYds', 'HomeTO', 'AwayTeam', 'AwayPts', 'AwayYds', 'AwayTO',
       'QBR_Off_Home', 'QBName_Home','QBR_Off_Away','QBName_Away',
       'QB_Pre_RK_Away', 'Poss/G_Off_Away', 'QB_Pre_RK_Home',
       'Poss/G_Off_Home', 'Poss/G_Def_Home', 'Poss/G_Off_Away', 'Poss/G_Def_Away',
       'Opened', 'Cost', 'Rest_Home', 'Rest_Away']

nfl = nfl.drop(drop_col, axis=1)

# turning capacity from string to number
nfl['Capacity'] = nfl['Capacity'].str.replace(',','')
nfl['Capacity'] = nfl['Capacity'].astype(str).astype(int)

#Cleaning the turf column for all category values match
nfl['Turf'] = nfl['Turf'].str.replace('t','T')

# Rename quarterback rank columns to make clearer
nfl = nfl.rename({'RK_Away':'QB_RK_Away', 'RK_Home':'QB_RK_Home'},axis=1)

#print(nfl.dtypes)
na_col = nfl.columns[nfl.isna().any()].tolist()
#print(nfl[na_col].isnull().sum())

#nfl['H_WinTeam'] = nfl['H_WinTeam'].dropna(axis=1)

nfl = nfl.dropna(subset=['H_WinTeam'], axis=0)
#print(nfl['H_WinTeam'].isnull().sum())


# null values comes from the quarterback ranks
# drop for now maybe later put in dummy values for the rank
# could see if the offense data frame had a rank for the quarterback

na_drop = ['QB_RK_Home','QBR_Home', 'PAA_Home', 'Plays_Home', 'EPA_Home', 'PASS_Home',
       'RUN_Home', 'SACK_Home', 'PEN_Home', 'RAW_Home', 'QB_RK_Away',
       'QBR_Away', 'PAA_Away', 'Plays_Away', 'EPA_Away', 'PASS_Away',
       'RUN_Away', 'SACK_Away', 'PEN_Away', 'RAW_Away']
nfl = nfl.dropna(subset=na_drop, axis=0)
na_col = nfl.columns[nfl.isna().any()].tolist()
print(nfl[na_col].isnull().sum())


# Cleaning the rest columns so it is just the number of days rest between games
nfl['win_rest_delta_days'] = nfl['win_rest_delta'].str.split(' ', expand=True)[0]
nfl = nfl.dropna(subset=['win_rest_delta_days'], axis=0)
nfl = nfl.drop(['win_rest_delta'], axis=1)
print(nfl['win_rest_delta_days'].isna().sum())
nfl['win_rest_delta_days'] = nfl['win_rest_delta_days'].astype(str).astype(int)

# Find ally columns not numeric type
nfl_types = nfl.dtypes

#Converting variables in to dummy values
nfl = pd.get_dummies(nfl)

# --------------------- Feature Selection Methods ---------------------

#### Univariate Selection: ####

X = nfl.drop('H_WinTeam', axis=1)
y = nfl['H_WinTeam']

# Values can't be negative so have to drop the follow columns
# Converting Sacks from negative to positive
drop_neg = ['win_rest_delta_days', 'PAA_Home', 'EPA_Home', 'PASS_Home',
       'RUN_Home', 'PEN_Home','PAA_Away', 'EPA_Away', 
       'PASS_Away', 'RUN_Away', 'PEN_Away']
X = X.drop(drop_neg, axis=1)
# Converting Sacks from negative to positive
X['SACK_Away'] = X['SACK_Away'].abs()
X['SACK_Home'] = X['SACK_Home'].abs()

X_col = X.shape[1]

### Apply SelectKBest Algorithm
### Also refered to as information gain?
ordered_rank_features = SelectKBest(score_func=chi2, k=X_col)
ordered_feature = ordered_rank_features.fit(X,y)

univar_score = pd.DataFrame(ordered_feature.scores_, columns=['Score'])
univar_col = pd.DataFrame(X.columns)

univar_df = pd.concat([univar_col, univar_score], axis=1)
univar_df.columns=['Features','Score']

# For SelectKBest Algorithm the higher the score the higher the feature importance
univar_df['Score'].sort_values(ascending=False)
univar_df = univar_df.nlargest(50, 'Score')


##### Feature Importance: #####
#This method provides a score for each feature of your data frame
#The higher the score the more relevant the data is
X = nfl.drop('H_WinTeam', axis=1)
y = nfl['H_WinTeam']

model = ExtraTreesClassifier()
model.fit(X,y)

#print(model.feature_importances_)
feat_impotant = pd.Series(model.feature_importances_, index=X.columns)
feat_impotant.nlargest(20).plot(kind='barh')
feat_impot_df = feat_impotant.sort_values(ascending=False).to_frame().reset_index()
feat_impot_df.columns=['Features','Feat Import']

#### Pearson Correlation Coefficient: ####

# Features with high correlation to the final result
nfl_corr = nfl.corr()
top_features=nfl_corr.index
h_win_corr = nfl_corr['H_WinTeam'].abs().sort_values(ascending=False)
h_win_coff_df = h_win_corr.to_frame().reset_index()
h_win_coff_df.columns=['Features','Corr']
plt.figure(figsize=(20,20))
sns.heatmap(nfl[top_features].corr(), annot=True)

#Looking for features with high correlation scores
#dropping high correlated features

# The goal for correlation is to find features that have high correlation
#to the final answer. You also want to drop the features that have high
#correlation to each other. If left in it could cause an overfitted model

# Function to find features that have high correlation to each other
# Want to drop all but one of the features that have high correlation to each other
# It is NOT finding features that are high correlation to answer column
def feature_corr(dataset, threshold):
    #corr_col = set() # Create a set of correlate column names
    corr_col = {}
    corr_df = dataset.corr()
    for i in range(len(corr_df.columns)):
        for j in range(i):
            #need to use absolute correlation value
            if abs(corr_df.iloc[i,j]) > threshold:
                colname = corr_df.columns[i]
                corr_col[colname] = abs(corr_df.iloc[i,j])
                #corr_col.add(colname)
    return corr_col

corr_threshold = 0.8
feat_corr = nfl.drop('H_WinTeam', axis=1)

feat_corr_df = pd.Series(feature_corr(feat_corr, corr_threshold), name='Corr')
feat_corr_df.index.name = 'Feature'
feat_corr_df.reset_index()


#### Information Gain ####
#Looking to see what highly correlated features are important to the final answer

# closer to zero the more independent the high the value the more dependent
X = nfl.drop('H_WinTeam', axis=1)
y = nfl['H_WinTeam']

mutual_info_values = mutual_info_classif(X,y)
mutual_info = pd.Series(mutual_info_values, index=X.columns)
mutual_info.sort_values(ascending=False)
mutual_info_df = mutual_info.sort_values(ascending=False).to_frame().reset_index()
mutual_info_df.columns=['Features','Mutual Info']


# Comparing feature selection methods to see what columns come up the most often
# univar_df, feat_impot_df, h_win_coff_df, mutual_info_df

feat_select = pd.concat([univar_df, feat_impot_df, h_win_coff_df, mutual_info_df], axis=1)
feat1 = feat_select.iloc[:10,0].tolist()
feat2 = feat_select.iloc[:10,2].tolist()
feat3 = feat_select.iloc[:10,4].tolist()
feat4 = feat_select.iloc[:10,6].tolist()
top_features = feat1+feat2+feat3+feat4

top_feat_unique = []
for feat in top_features:
    if feat not in top_feat_unique:
        top_feat_unique.append(feat)

"""
#### Apply Standard Scaler on all stats ####
X = nfl.drop('H_WinTeam', axis=1)
y = nfl['H_WinTeam']

scaler = StandardScaler()
scaled = scaler.transform(X)
X = pd.DataFrame(scaled, columns=X.columns)
"""



# --------------------- Machine Learning Methods ---------------------
# mse - mean squared error -> want the number closer to zero

X = nfl[top_feat_unique]
y = nfl['H_WinTeam']

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.33, random_state=42)


#### Linear Regression
# multiple cv with for loop
lin_reg = LinearRegression()
mse = cross_val_score(lin_reg,X,y,scoring='neg_mean_squared_error',cv=7)
mean_mse = np.mean(mse)
print(mean_mse)

#predition
lin_reg.predict('add prediction model')

#### Linear Regression with Train Test Split
mse_train = cross_val_score(lin_reg,X_train,y_train,scoring='neg_mean_squared_error',cv=7)
mean_mse_train = np.mean(mse_train)
print(mean_mse_train)

# prediction
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
r2_score1 = r2_score(y_pred,y_test)
print(r2_score1)


# The mse got better with the train test split

#### Ridge Regression and hyper parameter tuning
# Ridge method tries to reduce the overfitting

ridge = Ridge()

params = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_regressor=GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=7)
ridge_regressor.fit(X,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

# with train test split
ridge_regressor.fit(X_train,y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#### Lasso Regression and hyper parameter tuning

lasso = Lasso()

params = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

lasso_regressor=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=7)
lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

# with train test split
lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


#### Logistic Regression

LogisticRegression

# Train Test Split
log_reg = LogisticRegression()





