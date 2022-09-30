#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 19Aug2022

@author: Lynn Menchaca
"""

"""
The purpose of this document is to clean and combine all collected data files.

Processing steps:
    - schedule data file: combine all, clean headers and organize home and away teams
    - Teams and stadium
    - Team season rankings for QB, Offense and Defense
    - Create a look up table and document to use a unified format for the team name
       for all data sets. This will make everything easier to read,
       along with combining the data files easier. 
    - Combine all data files to one data file.
    - Clean final data files to get ready for analysis


Resources:
how to merge multiple files:
https://www.geeksforgeeks.org/how-to-merge-multiple-csv-files-into-a-single-pandas-dataframe/   

Quarterback = QB
Offense = Off
Defense = Def


"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import re

data_file_path = '/Users/lynnpowell/Documents/DS_Projects/NFL_Analysis/Data_Sets/'

# merging all the schedule files
joined_files = os.path.join(data_file_path, "nfl_*_schedule.csv")
# A list of all joined files is returned
joined_list = glob.glob(joined_files)
# Finally, the files are joined
df_sche = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

#Data cleaning for all the game schedules for all weeks and seasons
#This data files also has wins and losses
#removing null rows
df_sche.dropna(axis=0, inplace = True)

#combining date and time to one column
#Then sorting data frame by date and time
df_sche['DateTime'] = pd.to_datetime(df_sche['Date'] + ' ' + df_sche['Time'], format='%m/%d/%y %I:%M%p')
#df_sche['Season'] = pd.DatetimeIndex(df_sche['DateTime']).year
df_sche = df_sche.sort_values(by='DateTime',ascending=False)



#cleaning team names to make sure all names match current name of team
sche_teams = df_sche['Winner/tie'].value_counts()
#Washington Redskins = Washington Football Team, Oakland Raiders=Las Vegas Raiders,
#St. Louis Rams = Los Angeles Rams, San Diego Chargers = Los Angeles Chargers
df_sche['Winner/tie'] = df_sche['Winner/tie'].str.replace('Washington Redskins','Washington Football Team')
df_sche['Winner/tie'] = df_sche['Winner/tie'].str.replace('Oakland Raiders','Las Vegas Raiders')
df_sche['Winner/tie'] = df_sche['Winner/tie'].str.replace('St. Louis Rams','Los Angeles Rams')
df_sche['Winner/tie'] = df_sche['Winner/tie'].str.replace('San Diego Chargers','Los Angeles Chargers')
sche_teams = df_sche['Winner/tie'].value_counts()
df_sche['Loser/tie'] = df_sche['Loser/tie'].str.replace('Washington Redskins','Washington Football Team')
df_sche['Loser/tie'] = df_sche['Loser/tie'].str.replace('Oakland Raiders','Las Vegas Raiders')
df_sche['Loser/tie'] = df_sche['Loser/tie'].str.replace('St. Louis Rams','Los Angeles Rams')
df_sche['Loser/tie'] = df_sche['Loser/tie'].str.replace('San Diego Chargers','Los Angeles Chargers')
sche_teams = df_sche['Loser/tie'].value_counts()



#Bringing in remaining files
#Quarterback = QB
df_qb = pd.read_csv(data_file_path+'nfl_qb_ranking.csv')
df_qb.dropna(axis=0, inplace = True)
df_offense = pd.read_csv(data_file_path+'nfl_offense_ranking.csv')
df_defense = pd.read_csv(data_file_path+'nfl_defense_ranking.csv')
df_stadium = pd.read_csv(data_file_path+'nfl_stadiums.csv')

#Renameing columns
df_defense = df_defense.rename({'#': 'Rank'}, axis=1)
df_offense = df_offense.rename({'#': 'Rank'}, axis=1)

#Converting Name column to string type to troubleshoot error
#df_qb.dtypes
#df_qb['NAME'] = df_qb['NAME'].astype(str)
#df_qb['NAME'] = df_qb['NAME'].astype(pd.StringDtype())
#df_qb.dtypes
#Team name column for the QB data frame
pattern = r'\w([A-Z]+$)'
df_qb['QBTeam'] = df_qb['NAME'].str.extract(pattern, expand=False)
df_qb['QBName'] = df_qb.apply(lambda x: x['NAME'].replace(str(x['QBTeam']), ''), axis=1)

#Update teams to match the names in current season
#STL = LAR, OAK = LV, IIWSH = WSH, SD = LAC, AK = LV
df_qb['QBTeam'] = df_qb['QBTeam'].str.replace('STL','LAR')
df_qb['QBTeam'] = df_qb['QBTeam'].str.replace('OAK','LV')
df_qb['QBTeam'] = df_qb['QBTeam'].str.replace('AK','LV')
df_qb['QBTeam'] = df_qb['QBTeam'].str.replace('IIWSH','WSH')
df_qb['QBTeam'] = df_qb['QBTeam'].str.replace('SD','LAC')

#df_offense['Team'] = df_offense['Team'].astype(str)
df_offense['Team'] = df_offense['Team'].astype(pd.StringDtype())
df_defense['Team'] = df_defense['Team'].astype(pd.StringDtype())
#df_defense['Team'] = df_defense['Team'].astype(str)
#Cleaning team names to make all match relavent names
#offense: Oakland = Las Vegas, San Diego = L.A. Chargers, St. Louis = L.A. Rams, Los Angeles = L.A. Rams
#While troubleshooting later code found a space in the team name.
df_offense['Team'] = df_offense['Team'].str.strip()
df_offense['Team'] = df_offense['Team'].str.replace('Oakland','Las Vegas')
df_offense['Team'] = df_offense['Team'].str.replace('Oakland','Las Vegas')
df_offense['Team'] = df_offense['Team'].str.replace('San Diego','L.A. Chargers')
df_offense['Team'] = df_offense['Team'].str.replace('St. Louis','L.A. Rams')
df_offense['Team'] = df_offense['Team'].str.replace('Los Angeles','L.A. Rams')
#Defense: Oakland = Las Vegas, San Diego = L.A. Chargers, St. Louis = L.A. Rams, Los Angeles = L.A. Rams
#While troubleshooting later code found a space in the team name.
df_defense['Team'] = df_defense['Team'].str.strip()
df_defense['Team'] = df_defense['Team'].str.replace('Oakland','Las Vegas')
df_defense['Team'] = df_defense['Team'].str.replace('San Diego','L.A. Chargers')
df_defense['Team'] = df_defense['Team'].str.replace('St. Louis','L.A. Rams')
df_defense['Team'] = df_defense['Team'].str.replace('Los Angeles','L.A. Rams')

#creating a table of the different way team names are marked for each data frame
#qb_teams = df_qb.QBTeam.unique()
#offense_teams = df_offense.Team.unique()
#defense_teams = df_defense.Team.unique()
#stadium_teams = df_stadium['Team(s)'].unique()
#sche_teams = df_sche['Winner/tie'].unique()

#Improvement to code would be to make this step automated
team_data = [{'Mascot': 'Steeler', 'Initial': 'PIT', 'City': 'Pittsburgh', 'Name':'Pittsburgh Steelers'},
             {'Mascot': 'Raiders', 'Initial': 'LV', 'City': 'Las Vegas', 'Name':'Las Vegas Raiders'},
             {'Mascot': 'Chiefs', 'Initial': 'KC', 'City': 'Kansas City', 'Name':'Kansas City Chiefs'},
             {'Mascot': 'Cowboys', 'Initial': 'DAL', 'City': 'Dallas', 'Name':'Dallas Cowboys'},
             {'Mascot': 'Panthers', 'Initial': 'CAR', 'City': 'Carolina', 'Name':'Carolina Panthers'},
             {'Mascot': 'Broncos', 'Initial': 'DEN', 'City': 'Denver', 'Name':'Denver Broncos'},
             {'Mascot': 'WashFB', 'Initial': 'WSH', 'City': 'Washington', 'Name':'Washington Commanders'},
             {'Mascot': 'WashFB', 'Initial': 'WSH', 'City': 'Washington', 'Name':'Washington Football Team'},
             {'Mascot': 'Browns', 'Initial': 'CLE', 'City': 'Cleveland', 'Name':'Cleveland Browns'},
             {'Mascot': 'Lions', 'Initial': 'DET', 'City': 'Detroit', 'Name':'Detroit Lions'},
             {'Mascot': 'Patriots', 'Initial': 'NE', 'City': 'New England', 'Name':'New England Patriots'},
             {'Mascot': 'Dolphins', 'Initial': 'MIA', 'City': 'Miami', 'Name':'Miami Dolphins'},
             {'Mascot': 'Bills', 'Initial': 'BUF', 'City': 'Buffalo', 'Name':'Buffalo Bills'},
             {'Mascot': 'Packers', 'Initial': 'GB', 'City': 'Green Bay', 'Name':'Green Bay Packers'},
             {'Mascot': '49ers', 'Initial': 'SF', 'City': 'San Francisco', 'Name':'San Francisco 49ers'},
             {'Mascot': 'Eagles', 'Initial': 'PHI', 'City': 'Philadelphia', 'Name':'Philadelphia Eagles'},
             {'Mascot': 'Colts', 'Initial': 'IND', 'City': 'Indianapolis', 'Name':'Indianapolis Colts'},
             {'Mascot': 'Seahawks', 'Initial': 'SEA', 'City': 'Seattle', 'Name':'Seattle Seahawks'},
             {'Mascot': 'Ravens', 'Initial': 'BAL', 'City': 'Baltimore', 'Name':'Baltimore Ravens'},
             {'Mascot': 'Falcons', 'Initial': 'ATL', 'City': 'Atlanta', 'Name':'Atlanta Falcons'},
             {'Mascot': 'Giants', 'Initial': 'NYG', 'City': 'N.Y. Giants', 'Name':'New York Giants'},
             {'Mascot': 'Titans', 'Initial': 'TEN', 'City': 'Tennessee', 'Name':'Tennessee Titans'},
             {'Mascot': 'Texans', 'Initial': 'HOU', 'City': 'Houston', 'Name':'Houston Texans'},
             {'Mascot': 'Bengals', 'Initial': 'CIN', 'City': 'Cincinnati', 'Name':'Cincinnati Bengals'},
             {'Mascot': 'Buccaneers', 'Initial': 'TB', 'City': 'Tampa Bay', 'Name':'Tampa Bay Buccaneers'},
             {'Mascot': 'Rams', 'Initial': 'LAR', 'City': 'L.A. Rams', 'Name':'Los Angeles Rams'},
             {'Mascot': 'Bears', 'Initial': 'CHI', 'City': 'Chicago', 'Name':'Chicago Bears'},
             {'Mascot': 'Saints', 'Initial': 'NO', 'City': 'New Orleans', 'Name':'New Orleans Saints'},
             {'Mascot': 'Jaguars', 'Initial': 'JAX', 'City': 'Jacksonville', 'Name':'Jacksonville Jaguars'},
             {'Mascot': 'Cardinals', 'Initial': 'ARI', 'City': 'Arizona', 'Name':'Arizona Cardinals'},
             {'Mascot': 'Vikings', 'Initial': 'MIN', 'City': 'Minnesota', 'Name':'Minnesota Vikings'},
             {'Mascot': 'Jets', 'Initial': 'NYJ', 'City': 'N.Y. Jets', 'Name':'New York Jets'},
             {'Mascot': 'Chargers', 'Initial': 'LAC', 'City': 'L.A. Chargers', 'Name':'Los Angeles Chargers'}]

df_teams = pd.DataFrame(team_data)

#Unifying team names across all data frames
#Playing with different ways to see if team name value is in the column
def unify_team(row):
    #print(row)
    if row in df_teams['Mascot'].values:
        return row
    elif row in df_teams['Initial'].unique():
        return df_teams['Mascot'].loc[df_teams['Initial']==row].values[0]
    elif row in set(df_teams['Name']):
        return df_teams['Mascot'].loc[df_teams['Name']==row].values[0]
    elif row in df_teams['City'].values:
        #print('City Found')
        return df_teams['Mascot'].loc[df_teams['City']==row].values[0]
    else:
        return np.nan

#This part gave me the hardest part trying to convert all team names to a unified name
#For the offense and defense team names, none were converting. All rows were coming back as nan
#   Solution -> There was a space at the beginning of the word
df_stadium['TeamName'] = df_stadium['Team(s)'].apply(unify_team)
df_qb['TeamName'] = df_qb.QBTeam.apply(unify_team)
df_defense['TeamName'] = df_defense['Team'].apply(unify_team)
df_offense['TeamName'] = df_offense['Team'].apply(unify_team)
df_sche['WinTeam'] = df_sche['Winner/tie'].apply(unify_team)
df_sche['LoseTeam'] = df_sche['Loser/tie'].apply(unify_team)

#Checking for null values
#print('Stadium')
#print(df_stadium.isna().sum())
#print('Quarterbacks')
#print(df_qb.isna().sum())
#print('Defense')
#print(df_defense.isna().sum())
#print('Offense')
#print(df_offense.isna().sum())
#print('Season Schedules')
#print(df_sche.isna().sum())

#Adding previous season rank to QB offense, deffense
#Quarterback Dataframe
df_qb['QB_Pre_RK'] = np.nan
#for each row in the dataframe 
for i in df_qb.index:
    season = df_qb.loc[i, 'Season'] - 1
    qb = df_qb.loc[i, 'QBName']
    if season < 2006:
        df_qb.loc[i, 'QB_Pre_RK'] = np.nan
        #continue
    else:
        qb_list = df_qb.loc[df_qb['Season']==season]
        if qb in qb_list['QBName'].values:
            df_qb.loc[i, 'QB_Pre_RK'] = qb_list['RK'][qb_list['QBName']==qb].values[0]
        else:
            df_qb.loc[i, 'QB_Pre_RK'] = np.nan



#Finding duplicate quarter backs for a team in one season
#Keeping the name and stats of the higher ranked quarterback
#adding the number of plays from both quarterbacks together
df = df_qb[['Season', 'TeamName']][df_qb[['Season', 'TeamName']].duplicated(keep=False)]
qb_extra = df.groupby(list(df)).apply(lambda x: tuple(x.index)).tolist()

for qb1, qb2 in qb_extra:
    df_qb.loc[qb1, 'Plays'] = df_qb.loc[qb1, 'Plays'] + df_qb.loc[qb2, 'Plays']
    df_qb = df_qb.drop(qb2, axis=0)
    

#Defense dataframe
df_defense['Def_Pre_RK'] = np.nan
#for each row in the dataframe 
for i in df_defense.index:
    season = df_defense.loc[i, 'Season'] - 1
    team = df_defense.loc[i, 'TeamName']
    if season < 2006:
        df_defense.loc[i, 'Def_Pre_RK'] = np.nan
        #continue
    else:
        def_list = df_defense.loc[df_defense['Season']==season]
        if team in def_list['TeamName'].values:
            df_defense.loc[i, 'Def_Pre_RK'] = def_list['Rank'][def_list['TeamName']==team].values[0]
        else:
            df_defense.loc[i, 'Def_Pre_RK'] = np.nan
    
#Offfense dataframe
df_offense['Off_Pre_RK'] = np.nan
#for each row in the dataframe 
for i in df_offense.index:
    season = df_offense.loc[i, 'Season'] - 1
    team = df_offense.loc[i, 'TeamName']
    if season < 2006:
        df_offense.loc[i, 'Off_Pre_RK'] = np.nan
        #continue
    else:
        off_list = df_offense.loc[df_offense['Season']==season]
        if team in off_list['TeamName'].values:
            df_offense.loc[i, 'Off_Pre_RK'] = off_list['Rank'][off_list['TeamName']==team].values[0]
        else:
            df_offense.loc[i, 'Off_Pre_RK'] = np.nan  


#romving the Superbowl game from the data frame to make it easier to sort the away and home teams
#With more time figure out a way to include superbowl stats
df_sche.drop(df_sche[df_sche['Week'] =='SuperBowl'].index, inplace = True)

#Labeling all Tied games
df_sche['Win_Loc'] = df_sche.apply(lambda x: 'T' if x['Pts']==x['Pts.1'] else x['Win_Loc'], axis=1)

#Creating Home Team and Away Team columns
df_sche['HomeTeam'] = df_sche.apply(lambda x: x['WinTeam'] if x['Win_Loc']=='H' 
                                    else x['LoseTeam'] if x['Win_Loc']=='A'
                                    else np.nan, axis=1)
df_sche['HomePts'] = df_sche.apply(lambda x: x['Pts'] if x['Win_Loc']=='H' 
                                    else x['Pts.1'] if x['Win_Loc']=='A'
                                    else x['Pts'], axis=1)
df_sche['HomeYds'] = df_sche.apply(lambda x: x['YdsW'] if x['Win_Loc']=='H' 
                                    else x['YdsL'] if x['Win_Loc']=='A'
                                    else np.nan, axis=1)
df_sche['HomeTO'] = df_sche.apply(lambda x: x['TOW'] if x['Win_Loc']=='H' 
                                    else x['TOL'] if x['Win_Loc']=='A'
                                    else np.nan, axis=1)
df_sche['AwayTeam'] = df_sche.apply(lambda x: x['WinTeam'] if x['Win_Loc']=='A' 
                                    else x['LoseTeam'] if x['Win_Loc']=='H'
                                    else np.nan, axis=1)
df_sche['AwayPts'] = df_sche.apply(lambda x: x['Pts'] if x['Win_Loc']=='A' 
                                    else x['Pts.1'] if x['Win_Loc']=='H'
                                    else x['Pts.1'], axis=1)
df_sche['AwayYds'] = df_sche.apply(lambda x: x['YdsW'] if x['Win_Loc']=='A' 
                                    else x['YdsL'] if x['Win_Loc']=='H'
                                    else np.nan, axis=1)
df_sche['AwayTO'] = df_sche.apply(lambda x: x['TOW'] if x['Win_Loc']=='A' 
                                    else x['TOL'] if x['Win_Loc']=='H'
                                    else np.nan, axis=1)


#Manually assigning each Tie game home and away stats
#Given more time I would improve this and make it automated, leaving less room for error
index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Steeler') & (df_sche['LoseTeam'] == 'Lions') & 
        (df_sche['Win_Loc'] == 'T')].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Steeler'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOW']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Lions'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOL']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Eagles') & (df_sche['LoseTeam'] == 'Bengals') & 
        (df_sche['Win_Loc'] == 'T') & (df_sche['Season'] == 2020)].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Bengals'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOL']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Eagles'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOW']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Lions') & (df_sche['LoseTeam'] == 'Cardinals') & 
        (df_sche['Win_Loc'] == 'T')].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Lions'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOW']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Cardinals'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOL']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Vikings') & (df_sche['LoseTeam'] == 'Packers') & 
        (df_sche['Win_Loc'] == 'T') & (df_sche['Season'] == 2018)].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Packers'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOL']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Vikings'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOW']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Steeler') & (df_sche['LoseTeam'] == 'Browns') & 
        (df_sche['Win_Loc'] == 'T')].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Browns'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOL']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Steeler'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOW']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Seahawks') & (df_sche['LoseTeam'] == 'Cardinals') & 
        (df_sche['Win_Loc'] == 'T')].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Cardinals'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOL']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Seahawks'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOW']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Bengals') & (df_sche['LoseTeam'] == 'Panthers') & 
        (df_sche['Win_Loc'] == 'T')].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Bengals'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOW']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Panthers'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOL']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Vikings') & (df_sche['LoseTeam'] == 'Packers') & 
        (df_sche['Win_Loc'] == 'T') & (df_sche['Season'] == 2013)].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Packers'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOL']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Vikings'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOW']

index_lst = df_sche.loc[(df_sche['WinTeam'] == '49ers') & (df_sche['LoseTeam'] == 'Rams') & 
        (df_sche['Win_Loc'] == 'T')].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Rams'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOL']
df_sche.loc[index_lst[0], 'AwayTeam'] = '49ers'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOW']

index_lst = df_sche.loc[(df_sche['WinTeam'] == 'Eagles') & (df_sche['LoseTeam'] == 'Bengals') & 
        (df_sche['Win_Loc'] == 'T') & (df_sche['Season'] == 2008)].index.tolist()
df_sche.loc[index_lst[0], 'HomeTeam'] = 'Bengals'
df_sche.loc[index_lst[0], 'HomeYds'] = df_sche.loc[index_lst[0], 'YdsL']
df_sche.loc[index_lst[0], 'HomeTO'] = df_sche.loc[index_lst[0], 'TOL']
df_sche.loc[index_lst[0], 'AwayTeam'] = 'Eagles'
df_sche.loc[index_lst[0], 'AwayYds'] = df_sche.loc[index_lst[0], 'YdsW']
df_sche.loc[index_lst[0], 'AwayTO'] = df_sche.loc[index_lst[0], 'TOW']

#2016 Washington Football Team vs Bengals Tie -> location London
df_sche.drop(df_sche.loc[(df_sche['WinTeam'] == 'WashFB') & (df_sche['LoseTeam'] == 'Bengals') & 
                         (df_sche['Win_Loc'] == 'T')].index, inplace=True)
#With more time I would figure out what games were played not at a teams true home field (example London).
#With these games I would remove the information from the home and away team and replace with null.


#Problems with combining dataframes
#The season year was not all in the same data type
#There is not a QB rank for all teams all years, this leaves some nans in the rank dataframe
#Another common problem is spaces in the string columns, these dataframes did not have this problem
#troubleshooting merge -> indicator=True
#print(df_stadium['TeamName'].isin(df_sche['HomeTeam']))
df_qb['Season'] = df_qb['Season'].astype('int64')
df_sche['Season'] = df_sche['Season'].astype('int64')

#Combining all data frames to one data frame.
#Combine all ranks: QB, Offense, Defense
#df_sche: HomeTeam: all ranks, stadium
#         AwayTeam: all ranks, stadium (division and conference)

#Combine ranks: Quarterback, Offense, Defense
#Combining Offense and Defense
df_rank = pd.merge(df_offense,df_defense, how='left',left_on=['Season', 'TeamName'],
                   right_on=['Season', 'TeamName'],suffixes=('_Off', '_Def'))
#Adding Quarterback rank
df_rank = pd.merge(df_rank,df_qb, how='left',left_on=['Season', 'TeamName'],
                   right_on=['Season', 'TeamName'],suffixes=('', '_QB'))

#Home stadium
df_nfl = pd.merge(df_sche,df_stadium, how='left',left_on=['HomeTeam'],
                   right_on=['TeamName'],suffixes=('_Sche', '_Stadium'))
#Renameing stadium columns
df_nfl = df_nfl.drop('TeamName', axis=1)
df_nfl = df_nfl.rename({'Division': 'Div_Home', 'Conference':'Conf_Home'}, axis=1)
#Home Ranks
#print(df_nfl['Season'].isin(df_rank['Season']))
df_nfl = pd.merge(df_nfl,df_rank, how='left',left_on=['Season', 'HomeTeam'],
                   right_on=['Season', 'TeamName'],suffixes=('_NFL', '_Home'))

df_nfl = df_nfl.drop('TeamName', axis=1)

rank_rename = ['Rank_Off', 'Team_Off',
'Yards/G_Off', 'Rush/G_Off', 'Rush/P_Off', 'Pass/G_Off', 'QBR_Off',
'Sacks_Off', '3rd%_Off', 'Poss/G_Off', 'Pts/G_Off', 'Off_Pre_RK',
'Rank_Def', 'Team_Def', 'Yards/G_Def', 'Rush/G_Def', 'Rush/P_Def',
'Pass/G_Def', 'QBR_Def', 'Sacks_Def', '3rd%_Def', 'Poss/G_Def',
'Pts/G_Def', 'Def_Pre_RK', 'RK', 'QBName', 'QBR', 'PAA', 'Plays', 'EPA',
'PASS', 'RUN', 'SACK', 'PEN', 'RAW', 'QBTeam','QB_Pre_RK']

for name in rank_rename:
    df_nfl = df_nfl.rename({name: name+'_Home'},axis=1)

#Away Stadium
df_nfl = pd.merge(df_nfl,df_stadium[['TeamName','Division','Conference']], how='left',left_on=['AwayTeam'],
                   right_on=['TeamName'],suffixes=('_Sche', '_Stadium'))
#Renameing stadium columns
df_nfl = df_nfl.drop('TeamName', axis=1)
df_nfl = df_nfl.rename({'Division': 'Div_Away', 'Conference':'Conf_Away'}, axis=1)
#Away Ranks
df_nfl = pd.merge(df_nfl,df_rank, how='left',left_on=['Season', 'AwayTeam'],
                   right_on=['Season', 'TeamName'],suffixes=('_NFL', '_Away'))

df_nfl = df_nfl.drop('TeamName', axis=1)

for name in rank_rename:
    df_nfl = df_nfl.rename({name: name+'_Away'},axis=1)


#Adding column for conference game information
#This game is a division game
#df_nfl['div_game'] = 0
#condition = df_nfl['Div_Home']==df_nfl['Div_Away']
#df_nfl[condition]['div_game'] = 1
df_nfl['div_game'] = df_nfl.apply(lambda x: 1 if x['Div_Home']==x['Div_Away'] else 0, axis=1)
#This game is not a conference game
df_nfl['out_conf'] = df_nfl.apply(lambda x: 0 if x['Conf_Home']==x['Conf_Away'] else 1, axis=1)

#Adding the amount of time each team had between games

#for each row in the dataframe
#I can go down each row becuase the games have been sorted by date
df_nfl['Rest_Away'] = np.nan
df_nfl['Rest_Home'] = np.nan
#df_nfl['last_game_div_H'] = np.nan
#df_nfl['last_game_div_A'] = np.nan
#df_nfl['next_game_div_H'] = np.nan
#df_nfl['next_game_div_A'] = np.nan
#df_nfl['next_game_bye_H'] = np.nan
#df_nfl['next_game_bye_A'] = np.nan
team_list = df_teams['Mascot'].unique().tolist()
games = []
team_loc = []
for team in team_list:
    #print(team)
    games.clear()
    x = 0
    team_loc.clear()
    #(df_nfl.shape[0] - 1)
    while x <= (df_nfl.shape[0] - 1):
        #print(x)
        if team == df_nfl.loc[x, 'HomeTeam']:
            #print('home')
            games.append(x)
            team_loc.append('H')
        if team == df_nfl.loc[x, 'AwayTeam']:
            #print('away')
            games.append(x)
            team_loc.append('A')    
        x += 1
    i_now = games[0]
    games.pop(0)
    team_loc.pop(-1)
    #print(games, team_loc)
    #Adding the time the teams were able to rest
    for i, loc in zip(games, team_loc):
        #print(i_now, loc)
        if loc == 'H':
            #print(df_nfl.loc[i_now, 'DateTime'] - df_nfl.loc[i, 'DateTime'])
            df_nfl.loc[i_now, 'Rest_Home'] = df_nfl.loc[i_now, 'DateTime'] - df_nfl.loc[i, 'DateTime']
            #df_nfl.loc[i_now, 'last_game_div_H'] = df_nfl.loc[i, 'Div_Home']
        if loc == 'A':
            #print(df_nfl.loc[i_now, 'DateTime'] - df_nfl.loc[i, 'DateTime'])
            df_nfl.loc[i_now, 'Rest_Away'] = df_nfl.loc[i_now, 'DateTime'] - df_nfl.loc[i, 'DateTime']
            #df_nfl.loc[i_now, 'last_game_div_A'] = df_nfl.loc[i, 'Div_Away']
        i_now = i

#rest time delta (winner - loser)

#df_nfl['win_rest_delta'] = df_nfl.apply(lambda x: 0 if x['Win_Loc']=='A' else 1 , axis=1)

df_nfl['win_rest_delta'] = df_nfl.apply(lambda x: x['Rest_Away'] - x['Rest_Home'] if x['Win_Loc']=='A' 
                                        else x['Rest_Home'] - x['Rest_Away'], axis=1)

#df_nfl['win_rest_delta_days'] = pd.DatetimeIndex(df_nfl['win_rest_delta']).day
 
#drop extra team name columns
#print(df_nfl.columns)
drop_col = ['Day', 'Date', 'Time', 'Winner/tie', 'Win_Loc','Loser/tie', 'LoseTeam', 'Pts', 'Pts.1', 
            'YdsW', 'TOW', 'YdsL', 'TOL', 'Name', 'Team(s)', 'Team_Off_Home', 'Team_Def_Home',
            'NAME_NFL', 'QBTeam_Home', 'Team_Off_Away', 'Team_Def_Away','NAME_Away', 'QBTeam_Away']
df_nfl = df_nfl.drop(drop_col, axis=1)

#print(df_nfl.columns)

#Changing the winning team to a 1 (hometeam) or 0 (away team), if tie null
df_nfl['H_WinTeam'] = df_nfl.apply(lambda x: 1 if x['WinTeam'] == x['HomeTeam'] else 0, axis=1)
df_nfl['WinTeam'] = df_nfl.apply(lambda x: np.nan if x['HomePts']==x['AwayPts'] else x['H_WinTeam'], axis=1)
df_nfl = df_nfl.drop('H_WinTeam',axis=1)
df_nfl = df_nfl.rename({'WinTeam':'H_WinTeam'},axis=1)

#Save cleaned and full NFL data file
#df_nfl.to_csv(data_file_path+'nfl_cleaned_data.csv', index = False)
