#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 08:01:45 2022

@author: Lynn Menchaca
"""

"""
Date: 17Aug2022

I did not do any web scraping for this first portion of data collection and
analysis. Below are the resources used to copy or download data for
stadium information, game schedules per week and season, QB, defense and 
offense rankings per season.

Resources:
    https://www.stadiumsofprofootball.com/comparisons/
    https://fansided.com/2020/07/31/10-toughest-nfl-stadiums-road-teams/
    https://en.wikipedia.org/wiki/List_of_current_National_Football_League_stadiums
    https://www.pro-football-reference.com/years/2021/games.htm
    https://www.covers.com/sport/football/nfl/statistics/team-offense/2021-2022
    https://www.covers.com/sport/football/nfl/statistics/team-defense/2021-2022
    https://www.espn.com/nfl/qbr

"""

"""
Goals for part 2.


"""


#Gathers NFL stats as a dataframe, scraped from NFL.com
    
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
import time
import pandas as pd


def get_jobs(keyword, num_jobs, verbose, path, slp_time):    
    
    #Initializing the webdriver
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(executable_path=path, options=options)
    driver.set_window_size(1120, 1000)
    
    url = 'https://www.nfl.com/standings/league/2021/REG'
    
    driver.get(url)
    
    teams = []

