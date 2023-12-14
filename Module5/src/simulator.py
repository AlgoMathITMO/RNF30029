# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.optimize import linprog
#from tqdm import tqdm

'''скользящее среднее'''
def  MovingAverage(x, numb=10):
    n=len(x)//numb
    ma=list(x[:n])
    for j in range(len(x)-n):
        ma.append(np.mean(x[j:j+n]))
    return np.array(ma)

class customer():
    def __init__(self, custid, environment):
        def init_proprrties():
            proper0={'wealth':10000,
                     'salary':50000,
                    'survival':{'P_reg':1000, 
                    'k': 0.23,
                    'R0':200,
                    'k1':1,
                    'a':100,
                    'b':0.27,
                    'ave_w_sum':1500,
                    'ave_w_dev':70,
                    'J0':20,
                    'Ip': 3,
                    'Is':5,
                    'P': 1000,
                    'importance':5,
                    'predictability':5},
                     'socialization':{'P_reg':750, #
                    'k': 0.23,
                    'R0':200,
                    'k1':1,
                    'a':100,
                    'b':0.17,
                    'ave_w_sum':1000,
                    'ave_w_dev':70,
                    'J0':20,
                    'Ip': 2,
                    'Is':5,
                    'P' : 1000,
                    'importance':5,
                    'predictability':5},
                    'self_realization':{'P_reg':500, #
                    'k': 0.23,
                    'R0':200,
                    'k1':1,
                    'a':100,
                    'b':0.10,
                    'ave_w_sum':650,
                    'ave_w_dev':70,
                    'J0':20,
                    'Ip': 1,
                    'Is':5,
                    'P' : 1000,
                    'importance':5,
                    'predictability':5}}
            return proper0
        def adjust_props(prp): #variations for different customers in population
            predvariance=[0,0,0,1,1,1,1,2,2,2,3,3,4,5,6,7,8,9]
            # strategies=['ascetic','ascetic', 'investor', 'strategist', 'alarmist', 'rationalist', 'rationalist'] #How to react in crysis
            properties=copy.deepcopy(prp)
            #scale=(1.+abs(np.random.standard_cauchy()))/2 #Cauchy distribution in theory
            scale=(1.+abs(np.random.lognormal()))/2 #log-normal, like in real data distribution
            properties['salary']*=scale
            for i in ['survival', 'socialization', 'self_realization']:
                properties[i]['P']*=scale
                for j in ['P_reg', 'k','R0','k1','a','b','ave_w_sum','ave_w_dev','J0','Ip','Is']:
                    properties[i][j]*=np.random.normal(1,0.2)
                properties[i]['importance']+=np.random.randint(-properties[i]['importance']+1, high=properties[i]['importance'])
                properties[i]['predictability']=np.random.choice(predvariance)
            #properties['strategy']=np.random.choice(strategies)    
            properties['wealth']*=scale
            return properties

        self.id=custid
        self.environment=environment
        self.day=pd.to_datetime(environment.index.min())
        self.categories=['survival', 'socialization', 'self_realization']
        self.prop=adjust_props(init_proprrties())
        self.prop['wealth']-=sum([self.prop[i]['P'] for i in self.categories])
        self.wagesday=np.random.randint(2, 13)
        #self.strat=get_strat(cust_prop['strategy'])
        self.J2 = lambda k,r,R,J0, k1, Ip, Is, R0: \
            (R)**(-k1*(Ip*k-Is*k-1)/(k*r))*J0*(-k*r+R)**(-k1/(k*r))/(R0**(-k1*(Ip*k-Is*k-1)/(k*r))*(R)**(-k1/(k*r)))

        self.J = lambda P, deltaT, k1: np.exp(k1*P*deltaT/2)

        self.prob = lambda a, b, J: 1/(1 + a*np.exp(-J*b))
        self.eval_w = lambda P, ave_w_sum, ave_w_dev : np.random.normal((ave_w_sum + (ave_w_sum - P)) , 
                                                np.sqrt(ave_w_dev))# *J,#  пропорционально тек. состоянию и прогнозу на будущее -- сколько нужно накопить 
        # -- тоже эмоциональная оценка количества благ *e*pi
    
    def time_step(self):
        self.day+=pd.Timedelta('1D')
        #needs=[] # What do we need
        needs={}
        #cri=environment[environment.date==self.day.strftime('%Y-%m-%d')].crisis.values[0]
        if (self.day.day+self.wagesday)%15 == 0: # get money 
            self.prop['wealth']+=self.prop['salary']       
        for i in self.categories:
            self.prop[i]['P'] -= self.prop[i]['P_reg'] #*environment.loc[self.day, i]
            # есть тек. сост. Р, прогноз, на сколько его  хватит -- время (длина интервала) до дедлайна, 
            # доступность привычных средств реализации потребности -- прогноз времени на поиск решения?
            # -- это влияет на эмоции
            J_t = self.J2(self.prop[i]['k'],self.prop[i]['P_reg'],self.prop[i]['P'],
                          self.prop[i]['J0'], self.prop[i]['k1'], self.prop[i]['Ip'], 
                          self.prop[i]['Is'],self.prop[i]['R0'])#J2(t,P_reg,k,R0,J0, k1, Ip, Is) #J(P, deltaT)
            p = self.prob(self.prop[i]['a'],self.prop[i]['b'],J_t)
            if np.random.random() < p: # если эмоций достаточно, совершаем покупку (w, t, cat) - cat фикс, t==t, w ~ J?...
                w = self.eval_w(self.prop[i]['P'], self.prop[i]['ave_w_sum'], self.prop[i]['ave_w_dev']) # вычисляем пропорционально эмоциям
                if w>0:
                    self.prop[i]['P'] += w #*environment.loc[self.day, i]
                    needs[i]=w*(1-self.environment.loc[self.day, 'crisis']/3)*self.environment.loc[self.day, i]
        return needs
    

def market(cust, needs):
    global environment
    l=len(needs)
    #fluctuation=environment[environment.index==cust.day.strftime('%Y-%m-%d')] #environmental influence
    #cri=environment[environment.index==cust.day.strftime('%Y-%m-%d')].crisis.values[0]
    purchases={}
    if l>1:
        c=-np.array([cust.prop[s]['importance'] for s in needs.keys()])
        bounds=np.array(list(needs.values()))
        A_ub = np.ones(l).reshape(1,l)
        b_ub = np.array([cust.prop['wealth']])
        pur=linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=([i for i in zip(np.zeros(l), bounds)]))['x']
        if pur is None:
            print('Ой')
            purchases={}
        else:
            purchases={list(needs.keys())[i]:pur[i] for i in range(l)}
    elif l==1:
        purchases={list(needs.keys())[0]:min(cust.prop['wealth'], list(needs.values())[0])} 
    if len(purchases):
        cust.prop['wealth']-=sum(list(purchases.values()))
    return purchases

"""Population exploring"""
def draw1(ams):
    scale=ams.sum(axis=1).max()
    plt.figure(figsize=(18,13))
    plt.title('Expenses', size=20)
    ser=np.zeros(len(ams))
    for i in ams.columns:
        s1=ser.copy()
        ser+=ams[i].values/scale
        plt.plot(ser)
        plt.fill_between(np.arange(len(ser)), s1, ser, alpha=.5, label=i.replace('_','-').capitalize())
    print(min(ser))
    xt=np.arange(0, len(ams)+1, 14)
    plt.xticks(xt, ams.index.strftime('%Y-%m-%d').values[xt], size=14, rotation=30)
    plt.yticks(size=14)
    plt.xlabel('DATE', size=16)
    plt.ylabel('Normalized PAY_AMOUNT', size=16)
    plt.legend(fontsize=14)
    plt.grid(axis='both')
    plt.show()
    return None

def draw2(ams):
    plt.figure(figsize=(18,13))
    plt.title('Relative expenses', size=20)
    s={}
    for i in ams.columns: # 6536,
        s[i]=[]
        for j in ams.index:
            scale=ams.sum(axis=1)
            s[i].append(ams.loc[j][i]/scale[j])
    ser=np.zeros(len(ams))
    for i in ams.columns:
        s1=ser.copy()
        ser+=s[i] #Norm01(ams[i].values)[0]
        plt.plot(ser)
        plt.fill_between(np.arange(len(ser)), s1, ser, alpha=.5, label=i.replace('_','-').capitalize())
    xt=np.arange(0, len(ams)+1,14)
    plt.xticks(xt, ams.index.strftime('%Y-%m-%d').values[xt], size=14, rotation=30)
    plt.yticks(size=14)
    plt.xlabel('DATE', size=16)
    plt.ylabel('RELATIVE PAY_AMOUNT', size=16)
    plt.legend(fontsize=14)
    plt.grid(axis='x')
    plt.show()
    return None

