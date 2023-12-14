import pandas as pd
import numpy as np


def cats_to_14(df, col='mcc', new_col='group_14'):
    '''
    Группировка MCC кодов по 14 категориям пользовательских интересов. 
    
    Параметры:
        df -- Датасет
        col -- Название колонки с MCC кодами
        new_col -- Название колонки с категориями
    Вывод:
        Датасет <df> с категориями MCC-кодов в колонке <new_col>.
    '''
    
    # Определяем, какие MCC-коды входят в категории
    cat_food = [5411,5814,5499,5812,5462,5441,5422,5451,5309]
    cat_outfit = [5691,5651,5661,5621,5699,5949]
    cat_dwelling = [5211,4900,5722,5712,5261,5719,5251,5714,5039]
    cat_health = [5912,8099,8011,8021,8043,8062,8071]
    cat_beauty = [5977,7230,7298,5631]
    cat_money = [6011,6536,6012,6538,6010,9311,9222,6051,6300,6540]
    cat_travel = [4111,5541,4121,4131,7512,4784,4112,5533,7011,7523,
                  7542,5511,4511,5542,4789,7538,3011,5521] # added 5521
    cat_kids = [5641,5945,5943,8299,8220]
    cat_nonfood = [5331,5999,5311,5200,5399,5931,5948]
    cat_remote = [5968,5964]
    cat_telecom = [4814,4816,5732,4812,9402,4215,4899]
    cat_fun = [5921,5813,5993,5995,5192,5816,5992,5735,5942,7832,5941,
               7995,5947,7922,5944,7997,7999,5193,7941,742,7991,7994,
               7221,7841,5815,4722]
    cat_charity = [8398]
    cat_misc = [8999,7299,7311,7399,9399]
    
    # Составляем словарь вида "MCC-код":"категория"
    cats_14_dict = dict(zip(cat_food+cat_outfit+cat_dwelling+cat_health+cat_beauty+cat_money+cat_travel+
                            cat_kids+cat_nonfood+cat_remote+cat_telecom+cat_fun+cat_charity+cat_misc, 
                                    ['food' for i in cat_food]+['outfit' for i in cat_outfit]+
                                    ['dwelling' for i in cat_dwelling]+['health' for i in cat_health]+
                                    ['beauty' for i in cat_beauty]+['money' for i in cat_money]+
                                    ['travel' for i in cat_travel]+['kids' for i in cat_kids]+
                                    ['nonfood' for i in cat_nonfood]+['remote' for i in cat_remote]+
                                    ['telecom' for i in cat_telecom]+['fun' for i in cat_fun]+
                                    ['charity' for i in cat_charity]+['misc' for i in cat_misc]))
    
    # Если MCC-код никуда не входит, то запишем в "разное"
    df[new_col] = df[col].apply(lambda x: cats_14_dict.get(x, 'misc'))
    
    return df


def cats_to_3(df, col='group_14', new_col='group_3'):
    '''
    Группировка 14 пользовательских интересов по 3 базовым ценностям. 
    
    Параметры:
        df -- Датасет
        col -- Название колонки пользовательскими интересами
        new_col -- Название колонки с базовыми ценностями
    Вывод:
        Датасет <df> с базовыми ценностями в колонке <new_col>.
        
    '''
    # не будем учитывать категорию деньги
    df = df[~df[col].isin(['money'])]
    # Составляем словарь вида "MCC-код":"категория"
    cats_3_dict = {'food':'survival', 'outfit':'survival', 'dwelling':'survival', 'health':'survival',
         'remote':'socialization','travel':'socialization','nonfood':'socialization','telecom':'socialization','misc':'socialization',
         'beauty':'self_realization', 'kids':'self_realization','fun':'self_realization', 'charity':'self_realization'}

    # Создаем новую колонку с тремя категориями
    df[new_col] = df[col].apply(lambda x: cats_3_dict[x])
    
    return df


def create_trans_data(df, col_clientid='REGNUM', col_date='date', 
                        col_group='group_3', col_amount='PRC_AMT',
                        add_count=False):
    '''
    Составление датафрейма с информацией о потраченной сумме и факте совершения транзакции 
    по каждому дню по трем категориям для каждого клиента.
    
    Параметры:
        df -- Датасет
        col_clientid -- Название колонки с id клиентов
        col_date -- Название колонки с датой транзакции
        col_group -- Название колонки с тремя категориями MCC-кодов
        col_amount -- Название колонки с суммой транзакции
        add_count -- Добавить инф-ию о количестве транзакций по дням
    Вывод:
        Датафрейм размера (кол-во_клиентов, 2) с колонками money_trans и bin_trans.
    '''
    
    #    Общая сумма транзакций по категориям за день для каждого клиента
    
    # Составим сводную таблицу с подсчетом суммы транзакций
    pivot_money_tr = pd.pivot_table(df, index=col_clientid, columns=[col_date,col_group],
                                    values=col_amount, aggfunc='sum', fill_value=0)
    # Используем следующую конструкцию, чтобы были учтены все дни (даже если там не было транзакций)
    pivot_money_tr = pivot_money_tr.stack()
    pivot_money_tr = pivot_money_tr.reindex(pd.MultiIndex.from_product(pivot_money_tr.index.levels, 
                                                                       names=pivot_money_tr.index.names), fill_value=0)
    # Для удобства получим массив. Поменяем размерность, чтобы был нормальный нампай массив
    pivot_money_tr_arr = pivot_money_tr.values.reshape(
            df[col_clientid].nunique(),3,-1) # кол-во клиентов, кол-во категорий, кол-во дней
    # Заполняем наны, если вдруг
    pivot_money_tr_arr[np.isnan(pivot_money_tr_arr)] = 0 
    # Соединим пользователей с их массивами
    money_compressed = pd.DataFrame(dict(money_trans = list(pivot_money_tr_arr.astype(int))),
                                    index=np.sort(df[col_clientid].unique()))
    
    #    Факт совершения транзакций по категориям за день для каждого клиента
    
    # Факт совершения у нас зависит от потраченной суммы
    pivot_bin_tr_arr = pivot_money_tr_arr.copy()
    pivot_bin_tr_arr[np.where(pivot_bin_tr_arr<10)] = 0 # если потрачено менее 10 рублей, то транзакция не учитывается
    pivot_bin_tr_arr[np.where(pivot_bin_tr_arr>=10)] = 1
    # Соединим пользователей с их массивами
    bin_compressed = pd.DataFrame(dict(bin_trans = list(pivot_bin_tr_arr.astype(int))),
                                    index=np.sort(df[col_clientid].unique()))
    
    # Соединяем вместе
    bin_compressed['money_trans'] = money_compressed['money_trans']
    
    # Добавляем сумму
    if add_count:
        pivot_cnt_tr =pd.pivot_table(df, index=col_clientid, columns=[col_date,col_group],
                                    values=col_amount, aggfunc='count', fill_value=0)
        pivot_cnt_tr = pivot_cnt_tr.stack()
        pivot_cnt_tr = pivot_cnt_tr.reindex(pd.MultiIndex.from_product(pivot_cnt_tr.index.levels, 
                                                                       names=pivot_cnt_tr.index.names), fill_value=0)
        pivot_cnt_tr_arr = pivot_cnt_tr.values.reshape(df[col_clientid].nunique(),3,-1)
        pivot_cnt_tr_arr[np.isnan(pivot_cnt_tr_arr)] = 0 # заполняем наны
        count_compressed = pd.DataFrame(dict(count_trans= list(pivot_cnt_tr_arr)),
                                        index=np.sort(df[col_clientid].unique()))
        bin_compressed['count_trans'] = count_compressed['count_trans']
    
    return bin_compressed
    
    
def add_date_features(main_feat_trans, start_date='2020-01-01', end_date='2021-06-29'):
    '''
    Добавление дня недели и месяца, закодированных с помощью синуса и косинуса, 
    к основному признаку. 
    
    Параметры:
        main_feat_trans -- pd.Series с основным признаком по транзакциям
        start_date -- Первый день в формате YYYY-MM-DD
        end_date -- Последний день в формате YYYY-MM-DD
    Вывод:
        pd.Series размера (кол-во_клиентов) с признаками на каждого пользователя по дням. 
    '''
    
    dates = pd.date_range(start_date, end_date)
    n_clients = main_feat_trans.shape[0]
    n_cats = main_feat_trans.iloc[0].shape[0]
    main_feat_trans = np.concatenate(main_feat_trans.explode().values).reshape(n_clients, n_cats, -1)
    # Дни недели
    
    # Получим дни недели в цифровом виде для заданного промежутка 
    f = lambda x: x.weekday
    squares = f(dates)
    # Закодируем дни недели через синус и косинус
    f = lambda x: [np.sin(x*(2.*np.pi/7)), np.cos(x*(2.*np.pi/7))]
    sin_l, cos_l = f(squares.to_numpy())
    # Копируем до нужного размера
    sin = np.tile(sin_l,(n_cats,1))
    sin = np.tile(sin,(n_clients,1,1))
    cos = np.tile(cos_l,(n_cats,1))
    cos = np.tile(cos,(n_clients,1,1))
    
    # Получим месяцы в цифровом виде для заданного промежутка 
    f_m = lambda x: x.month
    squares_m = f_m(dates) - 1 # чтобы начиналось с нуля
    # Закодируем через синус и косинус
    f_m = lambda x: [np.sin(x*(2.*np.pi/12)), np.cos(x*(2.*np.pi/12))]
    sin_l_m, cos_l_m = f_m((squares_m).to_numpy())
    # Копируем до нужного размера
    sin_m = np.tile(sin_l_m,(n_cats,1))
    sin_m = np.tile(sin_m,(n_clients,1,1))
    cos_m = np.tile(cos_l_m,(n_cats,1))
    cos_m = np.tile(cos_m,(n_clients,1,1))

    # Запишем в один датафрейм
    # (n_clients,1) -> (n_clients, n_cats, n_days)
    feat_df = pd.DataFrame.from_records([main_feat_trans, sin, cos, sin_m, cos_m]).T
    feat_df.columns = ['feat_trans', 'dow_sin', 'dow_cos', 'm_sin', 'm_cos']
    feat_df = feat_df.apply(lambda x: np.column_stack((*[x['feat_trans'][cat,:] for cat in range(n_cats)],
                                                          x['dow_sin'][0,:], x['dow_cos'][0,:],
                                                          x['m_sin'][0,:], x['m_cos'][0,:])),
                           axis=1)

    
    return feat_df