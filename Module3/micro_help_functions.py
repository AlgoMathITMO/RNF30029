# Работа с данными
import pandas as pd
import numpy as np


def get_df(df, client_id, k=1, n_cats=3, n_train=28, n_pred=7):
    '''
    Получение сдвигов для временного ряда клиента.
    
    Параметры:
        df -- датасет с транзакционными данными
        client_id --идентификатор клиента
        k -- брать каждый k-ый сдвиг
        n_cats -- количество категорий транзакций
        n_train -- кол-во дней, на основании которых делается прогноз
        n_pred -- сколько дней прогнозируется за раз
    Вывод:
        Датафрейм со сдвигами.
    '''
    a = pd.DataFrame(df.loc[client_id], 
                 columns=[*[f'event_{i}' for i in range(n_cats)]
                          ,'d_sin','d_cos','m_sin','m_cos'])
    
    # Для экономии меняем тип
    cols = ['d_sin','d_cos','m_sin','m_cos']
    a[cols] = a[cols].apply(lambda x: x.astype('float16'))
    cols = [f'event_{i}' for i in range(n_cats)]
    a[cols] = a[cols].apply(lambda x: x.astype('int8'))

    orig_cols = a.columns
    
    # Вычисляем сдвиги всех признаков для входных данных модели
    to_add = []
    for i in range(1,n_train):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).iloc[::k].rename(f'{col}-{i}'))
    # Сдвиги целевого признака
    for i in range(n_train, n_train+n_pred):
          for col in orig_cols[:n_cats]: #предсказываем факт совершения транзакции
                to_add.append(a[col].shift(-i).iloc[::k].rename(f'{col}+{i-n_train+1}'))
    # Объединяем
    a = pd.concat([a.iloc[::k]]+to_add, axis=1) 
    a.dropna(inplace=True)
    
    return a


def get_splits_by_client(client_id, train_df, test_df, n_cats=3, n_train=28, n_pred=7, n_features=7):
    '''
    Получение признаков и целевой переменной для клиента.
    
    Параметры:
        client_id -- идентификатор клиента
        train_df -- датасет с транзакционными данными за обучающий период
        test_df -- датасет с транзакционными данными за тестовый период
        n_cats -- количество категорий транзакций
        n_train -- кол-во дней, на основании которых делается прогноз
        n_pred -- сколько дней прогнозируется за раз
        n_features -- кол-во признаков для обучения
    Вывод:
        Признаки и целевая переменная для обучающей и тестовой выборок.
    '''
    train_client = get_df(train_df, client_id, k=1, n_train=n_train, n_pred=n_pred)
    test_client = get_df(test_df, client_id, k=1, n_train=n_train, n_pred=n_pred)
    
    all_x = train_client.iloc[:,:-n_pred*n_cats].values.reshape(-1,n_train,n_features)
    all_y = train_client.iloc[:,-n_pred*n_cats:].values.reshape(-1,n_pred,n_cats)

    all_test_x = test_client.iloc[:,:-n_pred*n_cats].values.reshape(-1,n_train,n_features)
    all_test_y = test_client.iloc[:,-n_pred*n_cats:].values.reshape(-1,n_pred,n_cats)
    
    return all_x,all_y,all_test_x,all_test_y


def get_df_full_y_new(df, min_date='2020-07-01',max_date='2021-06-29', n_cats=3,
                     n_train=28,n_pred=7):
    '''
    Получение сдвигов для всех временных рядов с добавлением даты к целевой переменной.
    Параметры:
        df -- датасет с транзакционными данными
        min_date -- дата начала учета транзакций (YYYY-MM-DD)
        max_date -- дата конца учета транзакций (YYYY-MM-DD)
        n_cats -- количество категорий транзакций
        n_train -- кол-во дней, на основании которых делается прогноз
        n_pred -- сколько дней прогнозируется за раз
    Вывод:
        Датафрейм со сдвигами.
    '''
    a = pd.DataFrame(df, columns=[*[f'event_{i}' for i in range(n_cats)]
                          ,'d_sin','d_cos','m_sin','m_cos'])

    cols = ['d_sin','d_cos','m_sin','m_cos']
    a[cols] = a[cols].apply(lambda x: x.astype('float16'))
    cols = [f'event_{i}' for i in range(n_cats)]
    a[cols] = a[cols].apply(lambda x: x.astype('int32'))
    a['date'] = pd.date_range(min_date,max_date)

    orig_cols = a.columns
    # Вычисляем сдвиги всех признаков для входных данных модели
    to_add = []
    for i in range(1,n_train):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).rename(f'{col}-{i}'))
    # Вычисляем сдвиги всех признаков как целевых
    for i in range(n_train, n_train+n_pred):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).rename(f'{col}+{i-n_train+1}'))
    # Объединяем            
    a = pd.concat([a]+to_add, axis=1) 
    a.dropna(inplace=True)
    
    return a


def F1metr(x_real, x_pred): #классы: 1 - positive, O - negative
    '''
    Подсчет F-меры вручную, чтобы F1-score([0,0,0],[0,0,0]) был 1.
    '''
    x_pred, x_real= x_pred.astype(int), x_real.astype(int) 
    
    tp=len(np.where(x_pred[np.where(x_real==1)]==1)[0])
    tn=len(np.where(x_pred[np.where(x_real==0)]==0)[0])
    fp=len(np.where(x_pred[np.where(x_real==0)]==1)[0])
    fn=len(np.where(x_pred[np.where(x_real==1)]==0)[0])
    
    if (tp+fp)*(tp+fn)*tp:
        precision, recall = tp/(tp+fp), tp/(tp+fn)
        f1=2*precision*recall/(precision+recall) 
    else:
        f1=0.
        
    if (tp+tn+fp+fn):
        accuracy=(tp+tn)/(tp+tn+fp+fn)*100
    else:
        accuracy=0.
        
    if accuracy>99.: f1=1
    
    return f1


def apply_metric(metric, array):
    '''
    Посчитать заданную метрику в массиве реальных и предсказанных значений
    для каждой недели и каждой категории.
    '''
    return np.array([[[metric(ys[0],(ys[1]>0).astype('int')) for ys in weeks] for weeks in cats] for cats in array])
    
    
def get_base_inc_arrays(df_2models, metric, col_base='base', col_inc='inc',
                       n_cats=3, n_test_weeks=25, n_pred=7):
    '''
    Получение массивов с посчитанной метрикой качества из датасета.
    
    Параметры:
        df_2models -- датасет с предсказаниями двух моделей
        metric -- метрика качества
        col_base -- название колонки в датасете с выводами базовой модели
        col_inc -- название колонки в датасете с выводами инкрементальной
        n_cats -- кол-во категорий транзакций
        n_test_weeks -- кол-во недель в тестовой выборке
        n_pred - сколько дней прогнозируется за раз
    Вывод:
        Numpy массивы с посчитанной метрикой и "средней" метрикой по категориям
        для базовой и икнрементальной моделей.
    '''
    # Отдельно выделим массивы для каждой модели
    base_array = np.array([np.vstack(x) for x in df_2models[col_base].to_numpy().reshape(-1)],dtype=object
                         ).reshape(-1,n_cats,n_test_weeks,2,n_pred) # 2 -- real values and predictions

    inc_array = np.array([np.vstack(x) for x in df_2models[col_inc].to_numpy().reshape(-1)],dtype=object
                        ).reshape(-1,n_cats,n_test_weeks,2)
    inc_array = np.array([i.flatten() for i in inc_array.reshape(-1)],dtype=object
                        ).reshape(-1,n_cats,n_test_weeks,2,n_pred)
    
    # Считаем заданную метрику по предсказаниям и реальным данным моделей
    inc_with_metric = apply_metric(metric, inc_array)
    base_with_metric = apply_metric(metric, base_array)
    
    # Считаем "среднюю" метрику для каждой недели, чтобы из трех категорий было одно число
    # "средняя": корень из суммы квадратов метрик по категориям делим на корень трех.
    inc_with_metric_3in1 = np.array([np.sqrt(np.sum(user**2,0))/np.sqrt(3) for user in inc_with_metric])
    base_with_metric_3in1 = np.array([np.sqrt(np.sum(user**2,0))/np.sqrt(3) for user in base_with_metric])
    
    return inc_with_metric, inc_with_metric_3in1, base_with_metric, base_with_metric_3in1