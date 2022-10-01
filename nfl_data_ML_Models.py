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


Quarterback = QB
Offense = Off
Defense = Def


"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm



# Download data file
data_file_path = '/Users/lynnpowell/Documents/DS_Projects/NFL_Analysis/Data_Sets/'
nfl = pd.read_csv(data_file_path+'nfl_cleaned_data.csv', parse_dates=['DateTime'])

print(nfl.columns)

# Columns to drop:
drop_col = ['Season', 'Week', 'DateTime', 'HomeTeam', 'HomePts',
       'HomeYds', 'HomeTO', 'AwayTeam', 'AwayPts', 'AwayYds', 'AwayTO',
       'QBR_Off_Home', 'QBName_Home','QBR_Off_Away','QBName_Away',
       'QB_Pre_RK_Away']

nfl = nfl.drop(drop_col, axis=1)


#Clean data
# weighted mean of all ranks
# feature selection for stadium
# feature selection for offense stats
#   make all stats the same scale?
# feature selection for defense stats
# feature selection for quarterback stats
# build models



