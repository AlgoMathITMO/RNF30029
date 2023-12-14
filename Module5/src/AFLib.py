import numpy as np
import pandas as pd
import matplotlib.style
import matplotlib.pyplot as plt
matplotlib.style.use('classic')
matplotlib.use('TkAgg')

from PySimpleGUI.PySimpleGUI import Text,Checkbox, Submit, Input, Button, Radio, popup
from PySimpleGUI.PySimpleGUI import Exit, Window



def PlotSingle(db, gtgs):
    s=db.amt.sum()
    dates=pd.DataFrame({'date': pd.date_range(start=db.date.min(), end=db.date.max(), freq='1d').strftime('%Y-%m-%d')})
    plt.figure(figsize=(12,9))
    #gtgs=['food', 'outfit', 'health',  'travel','fun']
    markers=['.','*','X','^','+','o']*3
    for i,cat in enumerate(gtgs):
        catdb=pd.merge(dates, db[db.category==cat][['date','amt']], on='date', how='outer').fillna(value=0.)
        plt.plot(catdb.amt, marker=markers[i], markersize=16, ls=':', label=cat)
    plt.title('Expenses for %.2f thousand per %i days'%(s/1000, len(dates)), size=20)
    xt=np.arange(0, len(catdb), 14)
    plt.xticks(xt, catdb.date.values[xt], size=14, rotation=60)
    plt.yticks(size=14)
    plt.xlabel('DATE', size=16)
    plt.ylabel('PAYMENT', size=16)
    plt.grid()
    plt.legend(fontsize=14)
    plt.show()
    return()

def TabSingle(db, gtgs):
    catdb=pd.DataFrame({'date': pd.date_range(start=db.date.min(), end=db.date.max(), freq='1d').strftime('%Y-%m-%d')})
    for i,cat in enumerate(gtgs):
        catdb=pd.merge(catdb, db[db.category==cat][['date','amt']], on='date', how='outer').fillna(value=0.)
        catdb.rename(columns={catdb.columns[-1]:cat}, inplace = True)
    return catdb
    
def choose_single(db):
    choice=db.category.unique()
    layout = [
        [Text('\tMake your choice')],
        [Text('Customer\'s number'), Input('0',key= '-CID-', size=(7,1))],
        [Text('\tWhich of categories?')],
        [[Checkbox(i,True)] for i in choice],#5
        [Checkbox('Save table',False, key='-STab-')],
        [Submit(), Button('Clear all', key='_Clear_'), Exit()] 
    ]
    window = Window('Choice for a customer', layout, icon='bspb-ru.ico')
    window.read(timeout=.1)
    while True:
        event, val = window.read()
        if event in (None,'Exit'):
            cid=None
            choice=np.array([])
            break
        if event=='_Clear_':
            for i in range(db.category.nunique()):
                window[i](False)
        else: 
            if val['-CID-'].isdecimal(): 
                cid='Cust'+val['-CID-'].zfill(4)
                if not (cid in db.id.unique()):
                    popup('No %s among customers! Number < %i'%(cid,db.id.nunique()))
                    continue
                elif sum(list(val.values())[1:])<1:
                    popup('At least one category should be chosen')
                    continue
                else:
                    choice=choice[list(val.values())[1:-1]]
                    break
            else:
                popup('Amount of customers should be a number(1-5mln)')
            continue
    window.close()
    return (cid, choice, val['-STab-'])

def choose_crowd(db):
    choice=db.category.unique()
    layout = [
        [Text('\tMake your choice')],
        [Text('What kind of plot?')],
             [Radio('Separate series', "RADIO1", default=True)],#0
             [Radio('Cumulative series', "RADIO1", default=False)],
             [Radio('Shares of groups', "RADIO1", default=False)],
        [Text('\tWhich of categories?')],
        [[Checkbox(i,True)] for i in choice],#5
        [Submit(), Button('Clear all', key='_Clear_'), Exit()] 
    ]
    window = Window('Choice for the crowd', layout, icon='bspb-ru.ico')
    window.read(timeout=.1)
    while True:
        event, val = window.read()
        if event in (None,'Exit'):
            choice=np.array([])
            kind=None
            break
        if event=='_Clear_':
            for i in range(db.category.nunique()):
                window[i+3](False)
        else: 
            if sum(list(val.values())[3:])<1:
                popup('At least one category should be chosen')
                continue
            else:
                choice=choice[list(val.values())[3:]]
                kind=list(val.values())[0:3].index(True)
                break
            continue
    window.close()
    return (kind, choice)

def plot_sep(data):
    data['idate']=pd.to_datetime(data.date)
    d=data.groupby(['idate','category']).amt.sum().unstack().fillna(value=0.)#.rolling(14).mean()
    d.plot(figsize=(12,5),grid=True, xlabel='Date', ylabel='Payment')
    plt.legend([c.capitalize() for c in d.columns] )
    plt.show()
    return None

def plot_cumul(data):
    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
            '#7f7f7f', '#bcbd22', '#17becf']*2
    data['idate']=pd.to_datetime(data.date)
    ams=data.groupby(['idate','category']).amt.sum().unstack().fillna(value=0.)
    k=ams.columns
    scale=max([ams.loc[j,k].sum() for j in ams.index])
    plt.figure(figsize=(12,9))
    plt.title('Expenses by groups', size=20)
    ser=np.zeros(len(ams))
    print(ams.columns)
    for c,i in enumerate(ams.columns):
        s1=ser.copy()
        ser+=ams[i].values/scale
        plt.plot(ser,  c=colors[c])
        plt.fill_between(np.arange(len(ser)), s1, ser, alpha=.5, label=i.capitalize(), color=colors[c])
    xt=np.arange(0, len(ams)+1, 14)
    dates=ams.index[xt].strftime('%Y-%m-%d')
    plt.xticks(xt, dates, size=14, rotation=60)
    plt.yticks(size=14)
    plt.xlabel('DATE', size=16)
    plt.ylabel('Normalized PAY_AMOUNT', size=16)
    plt.legend(fontsize=14)
    plt.grid(axis='both')
    plt.show()
    return None

def plot_share(data):
    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
            '#7f7f7f', '#bcbd22', '#17becf']*2
    data['idate']=pd.to_datetime(data.date)
    ams=data.groupby(['idate','category']).amt.sum().unstack().fillna(value=0.)
    plt.figure(figsize=(12,9))
    plt.title('Expenses share by MCC', size=20)
    s={}
    scale=ams.sum(axis=1)
    for i in ams.columns:
        s[i]=ams[i]/scale
    ser=np.zeros(len(ams))
    for c,i in enumerate(ams.columns):
        s1=ser.copy()
        ser+=s[i] #Norm01(ams[i].values)[0]
        plt.plot(ser.values, c=colors[c])
        plt.fill_between(np.arange(len(ser)), s1, ser, alpha=.5, color=colors[c], label=i.capitalize())
    xt=np.arange(0, len(ams)+1, 14)
    dates=ams.index[xt].strftime('%Y-%m-%d')
    plt.xticks(xt, dates, size=14, rotation=60)
    plt.yticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('DATE', size=16)
    plt.ylabel('Expenses share', size=16)
    plt.legend(fontsize=14)
    plt.grid(axis='x')
    plt.show()
    return None

#
    
