# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:44:43 2019

@author: Shahabbahrami
"""
import numpy as np
from numpy.random import randint
from sites.DQN.PriceFunc import PriceFunc
import pandas as pd
from datetime import date
from datetime import datetime as dt
import datetime
def OutdoorTemp(t):
    if t%96<=48:
       Tout= 0.25*randint(56, 64)
    else:
       Tout= 0.25*randint(44, 52)
    return Tout

def People(t,FromHour,ToHour):
    if t%96<FromHour:
        N= 0
    elif t%96>=FromHour and t%96<=ToHour:
        N=  randint(1, 4)
    else:
        N=  0
    return N

def Price(t):
    D=96
    T=D*100
    PriceValue=np.zeros(T)
    PriceValue=np.tile(PriceFunc(D),int(T/D))
    return PriceValue[t]


def Desire(t, Desire,FromHour,ToHour):
    if t%96<FromHour:
        Tdes= 0
    elif t%96>=FromHour and t%96<=ToHour:
        Tdes=  Desire
    else:
        Tdes=  0
    return Tdes

def LoadData():
    df = pd.read_csv('sites/DQN/csvfiles/Hobo_15minutedata_2020.csv')
    # df.loc[:,'Date'] = pd.to_datetime(df.Date.astype(str)+' '+df.Time.astype(str))
    df['DateTime']=pd.to_datetime(df['Date'] + ' ' + df['Time'],errors='coerce')
    df1 = df[['DateTime','Temperature (S-THB 10510805:10502491-1), *C']]
    df1_tidy = df1.rename(columns = {'Temperature (S-THB 10510805:10502491-1), *C': 'Temp'}, inplace = False)
    return df1_tidy

def OutdoorTemp2(t):
    avg_temp, std_temp=Tempstatistics(t)
    Tout=np.round(np.random.normal(avg_temp, std_temp, 1),1)
    print('Tout=', Tout)
    return Tout[0]

def Tempstatistics(timeslot):
    backdays=7
    FixTemp=20
    today_day = datetime.date(datetime.date.today().year - 1, datetime.date.today().month, datetime.date.today().day)
    today=dt.combine(today_day, dt.min.time())+datetime.timedelta(minutes=1)
    past= today-datetime.timedelta(days=backdays)
    # print('today', today)
    df=LoadData()
    # print(df.loc[(df['DateTime'] <= today) & (df['DateTime'] >= past)])
    df1=df.loc[(df['DateTime'] <= today) & (df['DateTime'] >= past)]
    temp=df1['Temp'].to_numpy().astype(np.float)
    x=[]
    for i in range(backdays):
        x.append(temp[timeslot+i*96]+FixTemp)
    # print(temp)
    avg_temp=np.mean(x)
    std_temp=np.std(x)
    # print(np.round(np.random.normal(avg_temp, std_temp, 1),1))
    return avg_temp, std_temp


def DailyTempstatistics():
    backdays=7
    FixTemp=20
    today_day = datetime.date(datetime.date.today().year - 1, datetime.date.today().month, datetime.date.today().day)
    today=dt.combine(today_day, dt.min.time())+datetime.timedelta(minutes=1)
    past= today-datetime.timedelta(days=backdays)
    # print('today', today)
    df=LoadData()
    # print(df.loc[(df['DateTime'] <= today) & (df['DateTime'] >= past)])
    df1=df.loc[(df['DateTime'] <= today) & (df['DateTime'] >= past)]
    temp=df1['Temp'].to_numpy().astype(np.float)
    DailyAvg=[]
    DailyStd=[]
    for j in range(96):
        x=[]
        for i in range(backdays):
            x.append(temp[j+i*96]+FixTemp)
        # print(temp)
        avg_temp=np.mean(x)
        std_temp=np.std(x)
        DailyAvg.append(avg_temp)
        DailyStd.append(std_temp)
    # print(np.round(np.random.normal(avg_temp, std_temp, 1),1))
    return DailyAvg, DailyStd

def OutdoorTemp3(avg_temp, std_temp):
    Tout=np.round(np.random.normal(avg_temp, std_temp, 1),1)
    return Tout[0]
