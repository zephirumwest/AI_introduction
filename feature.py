# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:09:38 2021

@author: STUDIO.D#1
"""

import numpy as np
import pandas as pd

def get_data():
    data = pd.read_csv("./data/iris.csv")
    print(data.info())
    print(data.describe()["petalWidthCm"][2])
    
get_data()
    
# def get_data2():
#     data = pd.read_csv("./data/titanic.csv")
#     print(data.info())
#     print(data.isnull().sum())
#     age_mean = data.describe()["Age"][1]
#     data["Age"].fillna(age_mean,inplace = True)
#     print(data.isnull().sum())
        
    

# # get_data2()
    
# def get_data3():
#     data = pd.read_csv("./data/adult.csv")
#     print(data.info())
#     print(data.isnull().sum())
    
# # get_data3()

# def get_data4():
#     data = pd.read_csv("./data/adult.csv")
#     print(data.info())
#     print(data.isnull().sum())
    
#     a = data["workclass"].value_counts().idxmax()
#     print(a)
#     data["workclass"].fillna(a,inplace = True)
    
#     b = data["occupation"].value_counts().idxmax()
#     print(b)
#     data["occupation"].fillna(b,inplace = True)
    
#     c = data["native-country"].value_counts().idxmax()
#     print(c)
#     data["native-country"].fillna(c,inplace = True)
#     print(data.isnull().sum())
    
#     data.to_csv("my1_Data")
    
# get_data4()

























