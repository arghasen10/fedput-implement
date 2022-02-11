import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from tensorflow.keras import optimizers
from sklearn import metrics
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import os


def timeseries_data(data, history_window, future_window):
    X = []
    Y = []
    i = 0
    while (i < len(data) - history_window - future_window + 1):
        temp_df = data[i:i + history_window]
        ### 11 here is the column index of target variable(throughput)===================>.
        target_variable = np.mean(data[i + history_window:i + history_window + future_window, 11])
        X.append(temp_df)
        Y.append(target_variable)

        i += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def preprocess_data(data, test_split):
    ### Features used in the dataset ==>
    data = data[
        ['latitude', 'longitude', 'speed', 'net_type', 'data_state', 'data_act', 'gsm_neighbors', 'umts_neighbors',
         'lte_neighbors', 'rssi_strongest', 'dist', 'Throughput tcp']]

    # x = np.array(data['Throughput tcp'])
    # y3 = gaussian_filter1d(x, sigma, order=0)
    # data['Throughput tcp'] = y3

    train = data.iloc[0:int((1 - test_split) * data.shape[0])]
    test = data.iloc[int((1 - test_split) * data.shape[0]):]

    sc = StandardScaler()
    train = sc.fit_transform(train)
    test = sc.transform(test)

    ####(5,1 are the History window , future window size)=>
    x_train, y_train = timeseries_data(train, 5, 1)
    x_test, y_test = timeseries_data(test, 5, 1)
    return sc, x_train, y_train, x_test, y_test


def create_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(5, 12)))
    model.add(Dropout(0.2))
    model.add(LSTM(128))

    model.add(Dropout(0.2))
    model.add(Dense(1))
    sgd = optimizers.SGD(lr=0.3)

    # print(model.summary())
    return model


### NOTHING TO CHANGE HERE , MOVE ON


def first_federated_iter(model, layer_index, users):
    layer_wts = []
    for layer in model.layers:
        layer_wts.append(layer.get_weights())

    user_models = []
    for user in users:
        user_model = create_model()
        layers = user_model.layers
        for i in range(layer_index):
            layers[i].set_weights(layer_wts[i])

        user_model.fit(user[1], user[2], epochs=25, batch_size=16, validation_data=(user[3], user[4]), shuffle=False)
        user_models.append(user_model)

    final_wts = []

    for i in range(layer_index):
        final_layer_wt = []
        if len(model.layers[i].get_weights()) != 0:
            layer_wts = []
            c = 0
            su = 0
            for user_model in user_models:
                ll = user_model.layers[i].get_weights()
                ll = [len(users[c][1]) * x for x in ll]
                su += len(users[c][1])
                layer_wts.append(ll)
                c += 1

            for j in range(len(layer_wts[0])):
                sums = [ll[j] for ll in layer_wts]
                final_layer_wt.append(sum(sums) / (1.0 * su))

        final_wts.append(final_layer_wt)

    return final_wts, user_models


def collect_4G_data():
    root = './dataset/labelencoded4Gdatasets/'
    ls = os.listdir(root)
    users = []
    for path in ls:
        fullpath = root + path
        data = pd.read_csv(fullpath)
        sc, x_train, y_train, x_test, y_test = preprocess_data(data, 0.3)
        users.append([sc, x_train, y_train, x_test, y_test])

    return users


### sigma , H, F : self explanatory

def score(user_model, final_wts, user, sigma, H, F):
    user_model.set_weights(user_model.get_weights())
    layers = user_model.layers

    ### THE PARAMETER IN range() CONTROLS the layer upto which federated aggregation is carried out , default : 2
    ### MODEL ARCHITECTURE FOR LAYER INDEX. (0=> first lstm layer , 1=> dropout , 2=> second lstm layer , 3=> dropout,
    ### 4=> fully connected)
    for i in range(2):
        layers[i].set_weights(final_wts[i])

    results = user_model.evaluate(user[3], user[4], batch_size=16)
    print('test loss, test acc: ', results)
    y_pred = user_model.predict(user[3])
    y_pred = y_pred.reshape(y_pred.shape[0])
    sc = user[0]

    ### 11 here is the column index of target variable(throughput) (depends on dataset)
    y3 = sc.mean_[11] + np.sqrt(sc.var_[11]) * y_pred
    y3 = gaussian_filter1d(y3, sigma, order=0)

    ### LOG BEGIN , DO NOT CHANGE ANYTHING
    log = [y3[1] - 2 * y3[0]]
    for i in range(1, len(y3) - 1):
        log.append(-2 * y3[i] + y3[i - 1] + y3[i + 1])

    log.append(y3[len(y3) - 2] - 2 * y3[len(y3) - 1])

    log = np.array(log)
    ### LOG END

    ### REMOVE THIS LINE TO , REMOVE LOG=>
    y3 = y3 - 1.0 * log

    y_test = user[5]

    ### 0.7 is the train-test split , ensure that proper value (used at preprocess_data) is put here.
    sii = int(0.7 * y_test.shape[0]) + H
    yorg = []
    for i in range(len(y3)):
        yorg.append(np.mean(y_test[sii + i: sii + i + F]))

    print(metrics.r2_score(yorg, y3))
    print(np.corrcoef(yorg, y3))
    return

#
# users = collect_4G_data()
#
# model = create_model()
# final_wts, user_models = first_federated_iter(model, 5, users)
#
# i = 0
# for user in users:
#     score(user_models[i], final_wts, users[i], 1, 5, 10)
#     i+=1

