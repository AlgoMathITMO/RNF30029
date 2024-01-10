# Работа с данными
import numpy as np
import macro_help_functions
# Процесс выполнения
from tqdm.notebook import tqdm,trange
from tqdm.keras import TqdmCallback
# Tensorflow
import tensorflow
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense, concatenate, Dropout, LSTM
from keras import backend as K
# Предобработка
from sklearn.preprocessing import MinMaxScaler
# Метрики качества
from sklearn.metrics import mean_absolute_percentage_error


def fit_lstm(X, y, n_epoch, n_batch, n_neurons,learning_rate=0.00005):
    '''
    Обучение модели.
    
    Параметры: 
        X -- входные данные
        y -- целевая переменная
        n_epoch -- кол-во эпох обучения
        n_batch -- размер батча
        n_neurons -- кол-во нейронов в LSTM
        learning_rate -- шаг обучения
    Вывод:
        Обученная модель.
    '''
    # Архитектура
    in1 = Input(batch_shape = (n_batch,X.shape[1], X.shape[2]))
    out = LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), 
                stateful=True, return_sequences=False, activation='relu')(in1)
    out = Dropout(0.1)(out)
    out = Dense(32, activation='relu')(out)
    out = Dropout(0.1)(out)
    x = Dense(y.shape[1], activation='linear')(out)
    model = Model(inputs=[in1], outputs=x)
    
    # Обучаем
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error',metrics=['mape'], optimizer=optimizer,  run_eagerly=True)
    model.fit([X], y, validation_split=0.2,
                     epochs=n_epoch, batch_size=n_batch, verbose=0, callbacks=[TqdmCallback(verbose=0)], shuffle=False)

    return model

def make_forecast(model, dat, dim, mem, split, fwd):
    '''
    Прогнозирование.
    
    Параметры:
        model -- обученная модель
        dat -- нормализованные данные
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM 
        split -- кол-во дней для тестовой выборки
        fwd -- сколько дней предсказывается 
    Вывод:
        Список предсказаний
    '''
    zfwd=np.array([])
    trg=dat[-(dim+mem+split+1):-split+1]
    
    # Прогнозируем последовательно <fwd> дней, не за раз
    for i in range(fwd):
        X, y = macro_help_functions.MakeSet(trg, dim, mem)
        inp=[X[:1]]
        
        z=model.predict(inp, verbose=0)[0]
        # добавляем предсказание в массив, чтобы 
        # след.день прогнозировать в том числе и на его основе
        zfwd=np.concatenate((zfwd, z))
        trg=np.concatenate((trg[1:], z))
    return zfwd

def make_model(scaler, X, y, dat, clients_amount,
               dim, mem, split, fwd,
               n_epochs_each_model, n_batch_each_model,n_neurons_each_model,lr_each_model):
    '''
    Создание модели и прогнозирование.
    
    Параметры:
        scaler -- sklearn скейлер для нормализации
        X -- входные данные
        y -- целевая переменная
        dat -- нормализованные данные
        clients_amount -- суммы транзакций
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM 
        split -- кол-во дней для тестовой выборки
        fwd -- сколько дней предсказывается 
        n_epochs_each_model -- кол-во эпох обучения
        n_batch_each_model -- размер батча
        n_neurons_each_model -- кол-во нейронов в LSTM
        lr_each_model -- шаг обучения
    Вывод:
        Модель, ошибки предсказаний, предсказания и реальные данные на тестовой, 
        предсказания и реальные данные на обучающей.
    '''
    # Если до этого прогнозировали, то выкидываем из памяти лишнее
    try:
        del model
    except:
        pass
    model = fit_lstm(X[:-1], y[:], n_epochs_each_model, n_batch_each_model, 
                                            n_neurons_each_model,learning_rate=lr_each_model)
    # Задаем новую модель с размером батча 1, тк у нас stateful LSTM
    # (если при обучении был батч>1, то нельзя уже подавать другой размер батча)
    in1 = Input(batch_shape = (1, mem, dim))
    out = LSTM(n_neurons_each_model, batch_input_shape=(1, mem, dim), 
                        stateful=True, return_sequences=False, activation='relu')(in1)
    out = Dropout(0.1)(out)
    out = Dense(32, activation='relu')(out)
    out = Dropout(0.1)(out)
    x = Dense(1, activation='linear')(out)
    model_1 = Model(inputs=[in1], outputs=x)
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr_each_model)
    model_1.compile(loss='mean_squared_error',metrics=['mape'], optimizer=optimizer,  run_eagerly=True)
    # Передаем веса обученной модели
    model_1.set_weights(model.get_weights())
    
    # Смотрим, как модель подстроилась под обучающую выборку
    y_pred = []
    for i in trange(len(X), desc='Predict on train data'):
        y_pred.append(model_1.predict([X[i:i+1]], verbose=0))
    # Трансформируем из нормализованных  
    y_pred_tr = scaler.inverse_transform(np.array(y_pred).reshape(-1,1)).reshape(-1)
    y_true = clients_amount.iloc[mem+dim:-split].values.reshape(-1)
    
    # Прогнозируем на <fwd> шагов
    fwd_mapes = []
    full_fwd_pred_tr = []
    full_fwd_true = []

    fwd_pred = make_forecast(model_1, dat.reshape(-1), dim, mem, split,fwd)
    # Трансформируем значения из нормализованных
    fwd_pred_tr = scaler.inverse_transform(fwd_pred.reshape(-1,1)).reshape(-1)
    # Реальные данные за тот же промежуток
    fwd_true = clients_amount.iloc[-split:-split+fwd].values.reshape(-1)

    full_fwd_pred_tr.append(fwd_pred_tr)
    full_fwd_true.append(fwd_true)
    # Считаем MAPE
    fwd_mapes.append(mean_absolute_percentage_error(fwd_true, fwd_pred_tr))
    
    return model_1, fwd_mapes, full_fwd_pred_tr, full_fwd_true, y_pred_tr, y_true

def make_pred_base_model(model, scaler, dat, clients_amount,
                        dim, mem, split, fwd,
                        fwd_mapes, full_fwd_pred_tr, full_fwd_true):
    '''
    Прогнозирование (добавление к 0-й неделе).
    
    Параметры:
        model -- обученная модель
        scaler -- sklearn скейлер для нормализации
        dat -- нормализованные данные
        clients_amount -- суммы транзакций
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM 
        split -- кол-во дней для тестовой выборки
        fwd -- сколько дней предсказывается 
        fwd_mapes -- ошибки для 0-й недели
        full_fwd_pred_tr --предсказания для 0-й недели
        full_fwd_true -- реальные данные для 0-й недели
    Вывод:
        Ошибки предсказаний, предсказания и реальные данные на тестовой.
    '''
    fwd_pred = make_forecast(model, dat.reshape(-1), dim, mem, split,fwd)
    # Трансформируем значения из нормализованных
    fwd_pred_tr = scaler.inverse_transform(fwd_pred.reshape(-1,1)).reshape(-1)
    # Реальные данные за тот же промежуток
    fwd_true = clients_amount.iloc[-split:-split+fwd].values.reshape(-1)

    full_fwd_pred_tr.append(fwd_pred_tr)
    full_fwd_true.append(fwd_true)
    # Считаем MAPE
    fwd_mapes.append(mean_absolute_percentage_error(fwd_true, fwd_pred_tr))
    
    return fwd_mapes, full_fwd_pred_tr, full_fwd_true

def make_forecast_inc(model, dat, dim, mem, split, fwd, ep):
    '''
    Прогнозирование  с дообучением.
    
    Параметры:
        model -- обученная модель
        dat -- исходные данные
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM 
        split -- кол-во дней для тестовой выборки
        fwd -- сколько дней предсказывается 
        ep -- кол-во эпох дообучения
    Вывод:
        Предсказания.
    '''
    zfwd=np.array([])
    trg=dat[-(dim+mem+split+1):-split+1]
    
    x_for_train = []
    y_for_train = []
    
    # Прогнозируем на <fwd> шагов
    for i in range(fwd):
        X, y = macro_help_functions.MakeSet(trg, dim, mem)
        inp=[X[:1]]
        x_for_train.append(inp[0])
        y_for_train.append(y)

        z=model.predict(inp, verbose=0)[0]
        zfwd=np.concatenate((zfwd, z))
        trg=np.concatenate((trg[1:], z))
        
    # Дообучаем на <fwd> примерах
    model.fit(np.stack(x_for_train,axis=1)[0], np.stack(y_for_train,axis=1)[0], validation_split=0,
               epochs=ep, batch_size=1, verbose=0, shuffle=False)
    
    return zfwd

def use_inc_model(model, scaler, dat, clients_amount,
               dim, mem, split, fwd,
                  fwd_mapes, full_fwd_pred_tr, full_fwd_true,ep):
    '''
    Прогнозирование (добавление к 0-й неделе) с дообучением.
    
    Параметры:
        model -- обученная модель
        scaler -- sklearn скейлер для нормализации
        dat -- нормализованные данные
        clients_amount -- суммы транзакций
        dim -- кол-во дней, на основании которых делается прогноз
        mem -- глубина памяти для LSTM 
        split -- кол-во дней для тестовой выборки
        fwd -- сколько дней предсказывается 
        fwd_mapes -- ошибки для 0-й недели
        full_fwd_pred_tr --предсказания для 0-й недели
        full_fwd_true -- реальные данные для 0-й недели
        ep -- кол-во эпох дообучения
    Вывод:
        Ошибки предсказаний, предсказания и реальные данные на тестовой.
    '''
    # Прогнозируем и дообучаем
    fwd_pred = make_forecast_inc(model, dat.reshape(-1), dim, mem, split,fwd,ep)
    # Трансформируем значения из нормализованных
    fwd_pred_tr = scaler.inverse_transform(fwd_pred.reshape(-1,1)).reshape(-1)
    # Реальные данные за тот же промежуток
    fwd_true = clients_amount.iloc[-split:-split+fwd].values.reshape(-1)

    full_fwd_pred_tr.append(fwd_pred_tr)
    full_fwd_true.append(fwd_true)
    # Считаем MAPE
    fwd_mapes.append(mean_absolute_percentage_error(fwd_true, fwd_pred_tr))
    
    return fwd_mapes, full_fwd_pred_tr, full_fwd_true