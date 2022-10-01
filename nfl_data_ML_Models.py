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


Quarterback = QB
Offense = Off
Defense = Def


"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif


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


#Clean data
# weighted mean of all ranks
# add dummy values
# feature selection for stadium
# feature selection for offense stats
#   make all stats the same scale?
# feature selection for defense stats
# feature selection for quarterback stats
# build models


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
univar_df['Score'].sort_values()
univar_df.nlargest(20, 'Score')

##### Feature Importance: #####
#This method provides a score for each feature of your data frame
#The higher the score the more relevant the data is
X = nfl.drop('H_WinTeam', axis=1)
y = nfl['H_WinTeam']

model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)
feat_impot_df = pd.Series(model.feature_importances_, index=X.columns)
feat_impot_df.nlargest(20).plot(kind='barh')


#### Pearson Correlation Coefficient: ####
nfl_corr = nfl.corr()
top_features=nfl_corr.index
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
mutual_info_df = pd.Series(mutual_info_values, index=X.columns)
mutual_info_df.sort_values(ascending=False)



