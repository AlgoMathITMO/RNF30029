# Работа с данными
import pandas as pd
import numpy as np
# Предобработка
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import hankel
# Визуализация
import matplotlib.pyplot as plt
# Метрики качества
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def client_groups_nth_week(df_2models,df_real_id, bool_clients, nth_week=0, window_size=21):
    '''
    Подсчет сумм транзакций по группам клиентов, выделенных на заданную неделю.
    
    Параметры:
        df_2models -- датасет с предсказаниями двух моделей
        df_real_id -- датасет с настоящими идентификаторами клиентов
        bool_clients -- массив с распределением "хороших" и "плохих" клиентов по неделям
        nth_week -- номер недели, для которой искать группы клиентов
        window_size -- размер окна сглаживания
    Вывод:
        Идентификаторы "хороших", суммы транзакций для групп клиентов
    '''
    good_clients_idx = np.where(bool_clients[:,nth_week]==1)
    # Т.к. брали массивы из датасета, то индексы оттуда пропали; возвращаем соответствие!
    good_clients_id_upto8k = pd.Series(df_2models.index).iloc[good_clients_idx].values
    # Переходим к реальным идентификаторам
    good_clients_id = pd.Series(df_real_id.index).iloc[good_clients_id_upto8k].values
    # Суммы хорошо-предсказуемых клиентов
    gclients_amount = df_real_id.loc[good_clients_id]['money_trans']

    # Суммы плохо-предсказуемых клиентов
    # их реальные id
    bad_clients_idx = np.where(bool_clients[:,nth_week]==0)
    bad_clients_id_upto8k = pd.Series(df_2models.index).iloc[bad_clients_idx].values
    bad_clients_id = pd.Series(df_real_id.index).iloc[bad_clients_id_upto8k].values
    # суммы
    bclients_amount = df_real_id.loc[bad_clients_id]['money_trans']

    # Суммы всех клиентов
    aclients_amount = df_real_id.loc[pd.Series(df_real_id.index).iloc[df_2models.index].values]['money_trans']
        
    gclients = gclients_amount.squeeze().to_list()
    bclients = bclients_amount.squeeze().to_list()
    aclients = aclients_amount.squeeze().to_list()
    
    # Скользящая медиана, чтобы избавиться от провалов в конце кварталов и аномально богатых
    good_rw = pd.Series(np.array(gclients).sum(0)[2,:]).rolling(window=window_size).median()
    bad_rw = pd.Series(np.array(bclients).sum(0)[2,:]).rolling(window=window_size).median()
    all_rw = pd.Series(np.array(aclients).sum(0)[2,:]).rolling(window=window_size).median()
    
    return good_clients_id, good_rw, bad_rw, all_rw

def prepare_data(clients_amount, split, dim=28, mem=5, scaler=0):
    '''
    Выделение признаков и целевой переменной.
    
    Параметры:
        clients_amount -- pd.Series с суммой транзакций
        split -- кол-во дней для тестовой выборки
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM
        scaler -- sklearn скейлер для нормализации, если есть
    Вывод:
        Скейлер, признаки и целевая переменная, нормализованные данные целиком.
    '''
    clients_amount = pd.DataFrame(clients_amount)
    
    # ОТДЕЛЬНО нормализуем трейн и тест
    # Используем скейлер, если задан
    if not scaler:
        scaler = MinMaxScaler(clip=False, feature_range=(-1, 1))
        dat_train = scaler.fit_transform(clients_amount[:-split])
    else:
        dat_train = scaler.transform(clients_amount[:-split])      
    dat_test = scaler.transform(clients_amount[-split:])
    dat = np.concatenate((dat_train,dat_test))
    
    # Составляем сдвиги временного ряда
    ser=dat_train.reshape(-1)[:]
    x, y = MakeSet(ser, dim, mem)
    
    return scaler, x, y, dat

def MakeSet(ser, dim, mem):
    '''
    Создание сдвигов для временного ряда.
    
    Параметры:
        ser -- временной ряд
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM
    Вывод:
        Признаки и целевая переменная.
    '''
    H=hankel(ser)
    X0=H[:-dim, :dim]
    X=[]
    for i in range(X0.shape[0]-mem-1):
        X.append(X0[i:i+mem, :])  
    X=np.array(X)
    y=H[mem+1:-dim, dim:dim+1]
    
    return X, y

def plot_clients_pred(clients_amount, full_fwd_pred_tr,full_fwd_true,y_true,y_pred_tr,
        dim, mem, split, fwd, days, tick_freq, shift_days):
    '''
    График как модель подстроилась под обучающую выборку и какие предсказания делает.
    
    Параметры:
        clients_amount -- суммы транзакций
        full_fwd_pred_tr -- список предсказаний на тестовой выборке
        full_fwd_true -- список реальных значений на тестовой выборке
        y_true -- список реальных значений на обучающей
        y_pred_tr -- список предсказаний на обучающей
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM 
        split -- кол-во дней для тестовой выборки
        fwd -- сколько дней предсказывается 
        days -- даты
        tick_freq -- частота меток на оси X
        shift_days -- первый валидный индекс после скользящей медианы
    '''
    yt= clients_amount.iloc[mem+dim:-split].values.reshape(-1)
    ticks = days[shift_days+mem+dim:] 
    xt=np.arange(0, len(ticks), tick_freq)
    
    fig = plt.figure()
    # На обучающей
    plt.plot(yt, c='darkblue', alpha=.5, label='Initial series')
    plt.plot(y_pred_tr, lw=2, label=f'Model on history. R2={r2_score(y_true[1:], y_pred_tr):.2f}')
    # На тестовой
    t=np.arange(len(yt), len(yt)+fwd)
    plt.plot(t, full_fwd_true[-1], c='darkblue', alpha=.1, label=None)
    plt.plot(t, full_fwd_pred_tr[-1], lw=2,c='green')
    plt.plot(np.arange(len(yt), len(yt)+split-2), clients_amount.iloc[-split:-2].values.reshape(-1),c='darkblue', alpha=.5)
    # Граница обучающей и тестовой
    plt.axvline(len(yt), ls=':', c='k')
    plt.text((len(yt)), plt.gca().get_ylim()[1]*0.95, 
             f'{days.iloc[-split]}', rotation=0)
    plt.xticks(xt, ticks.iloc[xt], rotation=45, ha='right')

    plt.title('Предсказания для клиентов')
    plt.xlabel('Дата')
    plt.ylabel('Сумма')

    plt.legend()
    plt.grid()
    
    return fig

def plot_clients_pred_base(chosen_client_group,a1_dm,df_2models,df_real_id,bool_clients,
                           full_fwd_pred_tr,full_fwd_true,split,
                            dim, mem, fwd, days, tick_freq, n_weeks,shift_days):
    '''
    График: какие предсказания делает модель.
    
    Параметры:
        chosen_client_group -- good/bad/all
        a1_dm -- недели в тестовой выборке
        df_2models -- датасет с предсказаниями двух моделей
        df_real_id -- датасет с настоящими идентификаторами клиентов
        bool_clients -- массив с распределением "хороших" и "плохих" клиентов по неделям
        full_fwd_pred_tr -- список предсказаний на тестовой выборке
        full_fwd_true -- список реальных значений на тестовой выборке
        y_true -- список реальных значений на обучающей
        y_pred_tr -- список предсказаний на обучающей
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM 
        split -- кол-во дней для тестовой выборки
        fwd -- сколько дней предсказывается 
        days -- даты
        tick_freq -- частота меток на оси X
        n_weeks -- количество тестовых недель
        shift_days -- первый валидный индекс после скользящей медианы
    '''
    fig = plt.figure()
        
    for nth_week in range(n_weeks-1):
        good_clients_id, next_good_rw, next_bad_rw, \
            next_all_rw = client_groups_nth_week(df_2models,df_real_id,bool_clients,nth_week)
        # well.... оставим пока так
        if chosen_client_group == 'good':
            client_group = next_good_rw
        elif chosen_client_group == 'bad':
            client_group = next_bad_rw
        elif chosen_client_group == 'all':
            client_group = next_all_rw
        
        last_day = days[days==f'{a1_dm[nth_week]}'].index[0]
        next_split= days.shape[0]-last_day-1
        yt= client_group[shift_days+mem+dim:-next_split].values.reshape(-1)
        
        # Реальные данные
        plt.plot(client_group[shift_days+mem+dim:-split].values.reshape(-1), c='darkblue', alpha=.02)
        
        ticks=days[shift_days+mem+dim:]
        xt=np.arange(0, len(ticks), tick_freq)
        
        # На тестовой
        t=np.arange(len(yt), len(yt)+fwd)
        plt.plot(t, full_fwd_true[nth_week], c='darkblue', alpha=.5, label=None)
        plt.plot(t, full_fwd_pred_tr[nth_week], lw=2, c='green')

        # Граница обучающей и тестовой
        #plt.axvline(len(yt), ls=':', c='k')
        plt.xticks(xt, ticks.iloc[xt], rotation=45, ha='right')
        
    plt.title('Предсказания для клиентов')
    plt.xlabel('Дата')
    plt.ylabel('Сумма')
    plt.grid()
    
    return fig