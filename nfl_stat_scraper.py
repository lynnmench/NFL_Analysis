#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 08:01:45 2022

@author: Lynn Menchaca
"""

#Gathers NFL stats as a dataframe, scraped from NFL.com

    
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
import time
import pandas as pd


