# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:25:30 2023

@author: user
"""
'''For pure fantasy'''
import pandas as pd
import numpy as np
def make_env(start, n):
    n=200
    date=pd.date_range(start='2020-01-01', periods=n)
    dow=np.where(date.dayofweek.values<5, 1.0, 0.9)
    environment=pd.DataFrame({'date':date.strftime('%Y-%m-%d'),
                              'survival':np.random.normal(1,0.02,n)*np.linspace(1,1.1,n)*1.2*(dow), 
                              'socialization':np.random.normal(1,0.01,n)*np.linspace(1,1.1,n), 
                              'self_realization':np.random.normal(1,0.01,n)*np.linspace(1,1.1,n)*.7*(2.-dow),
                             'crisis':np.zeros(n).astype(int)+1.-dow})
    crd=pd.date_range(start='2020-03-27', periods=50).strftime('%Y-%m-%d').to_list()
    i=environment[environment.date==crd[0]].index[0]
    environment.iloc[i:i+len(crd), 4]=.7/(np.exp(np.linspace(-0.5,3,len(crd))**2))
    for d in ['2019-05-06','2019-05-07','2019-05-08','2019-05-09','2019-01-01'
        '2020-05-06','2020-05-07','2020-05-08','2020-05-09','2020-01-01', '2020-01-02','2020-01-03']:
        i=environment[environment.date==d].index
        environment.iloc[i, 4]=.3  
    environment.date=pd.to_datetime(environment.date)
    environment.set_index('date', drop=True, inplace=True)
    # d=environment[(environment.date>='2022-12-20')&(environment.date<='2022-12-31')].index
    # environment.iloc[d, 1:]=[np.array([1,2,1.5,0.5,0.1,1.3,1.5,1.7,0.6,1.3,1.8,2,1.4])*i for i in np.random.normal(1,.2,len(d))]
    return environment#.iloc[d,:]