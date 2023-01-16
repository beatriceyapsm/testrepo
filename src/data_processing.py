import pandas as pd
import numpy as np
from scipy import stats
import sqlite3

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def dropCarID(df):
    if len(df["Car ID"].unique()) == len(df["Car ID"]):
        df = df.drop(columns=["Car ID"])
    return df

def dropnull(df):
    df = df.dropna()
    return df

def dropoutliers(df):
    z = np.abs(stats.zscore(df._get_numeric_data()))
    df = df[(z < 3).all(axis=1)]
    return df

def convert_temp(df):
    #Split temperature into 2 columns - one for number, and one for unit
    df['Temp-C']=df['Temperature'].apply(lambda x: x.split(" ")[0])
    df['Temp-unit']=df['Temperature'].apply(lambda x: x.split(" ")[1])
    df = df.drop(columns=["Temperature"])
    #Convert Temp-C data from string to numeric
    df["Temp-C"]=pd.to_numeric(df["Temp-C"])
    #Convert all deg F numbers (>200) to deg C. (deg F - 32)*5/9 = deg C
    df.loc[df['Temp-C'] > 200, 'Temp-C'] = (df['Temp-C']-32)*5/9
    #round temp to 1 dp, as it should not have higher precision than raw data
    df['Temp-C']=df['Temp-C'].round(1) 
    #Temperature-unit no longer required
    df = df.drop(columns=["Temp-unit"])
    return df

def convert_RPM(df):
    df.loc[df["RPM"] <0,'RPM'] = -df["RPM"]
    return df

def convert_model(df):
    #Split Model into 2 columns - one for model, and one for year
    df['Model-no']=df['Model'].apply(lambda x: x.split(",")[0])
    df['Model-year']=df['Model'].apply(lambda x: x.split(",")[1])
    df = df.drop(columns=["Model"])
    df['Model-no']= df['Model-no'].str.replace('Model',' ')
    df['Model-no']=pd.to_numeric(df['Model-no'])
    df['Model-year']=pd.to_numeric(df['Model-year'])
    return df

def convert_factory(df):
    array = ["Shang Hai, China", "Berlin, Germany", "New York, U.S"]
    df = df.loc[df['Factory'].isin(array)]
    return df

def textToNumbers(df):
    #Map keys to numbers
    key1 =('Blue' ,'Black', 'Grey' ,'White' ,'Red')
    df['Color'] = df['Color'].map(lambda x: key1.index(x))
    key2 =('Shang Hai, China' ,'Berlin, Germany', 'New York, U.S')
    df['Factory'] = df['Factory'].map(lambda x: key2.index(x))
    key3 =('Low' ,'Medium' ,'High')
    df['Usage'] = df['Usage'].map(lambda x: key3.index(x))
    key4 =('None','Normal' ,'Premium')
    df['Membership'] = df['Membership'].map(lambda x: key4.index(x))

    #Combine all failures into 1 column since objective to predict failure occurrence but does not focus on specific failures.
    df["Failures"]=df["Failure A"]+df["Failure B"]+df["Failure C"]+df["Failure D"]+df["Failure E"]
    df = df.drop(columns=["Failure A", "Failure B","Failure C", "Failure D","Failure E"])
    return df

def dropUnrelatedColumns(df):
    #Drop unrelated data: Color, Model no., Usage
    df = df.drop(columns=["Color", "RPM", "Factory","Model-no","Temperature"])
    return df