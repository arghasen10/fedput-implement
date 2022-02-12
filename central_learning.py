import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import Counter
import param_list
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from tensorflow.keras import optimizers
from sklearn import metrics
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from pprint import pprint
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(10)

print('this should to write to the log file')
DATASETS = ['IRISH', 'LUMOS', 'MNWILD', 'CENTRAL', 'MERGED']


def collect_data_irish(type=1):
    files = [x for x in
             os.listdir('dataset/5G-production-dataset/Amazon_Prime/Driving/animated-AdventureTime')]
    files2 = [x for x in
              os.listdir('dataset/5G-production-dataset/Amazon_Prime/Driving/Season3-TheExpanse')]
    for file in files2:
        files.append(file)
    column_name = ['Speed', 'CellID', 'RSRP', 'RSRQ', 'SNR', 'RSSI', 'DL_bitrate', 'NRxRSRP', 'NRxRSRQ']
    cols = [3, 5, 7, 8, 9, 11, 12, 24, 25]
    data_irish = []
    for file in files:
        rootPathToIrish = 'dataset/5G-production-dataset/Amazon_Prime/Driving/animated-AdventureTime/'
        rootPathToIrish2 = 'dataset/5G-production-dataset/Amazon_Prime/Driving/Season3-TheExpanse/'
        try:
            path = rootPathToIrish + file
            df = pd.read_csv(path, usecols=cols)
            df.dropna(inplace=True)
            data_irish.append(df)
        except:
            path = rootPathToIrish2 + file
            df = pd.read_csv(path, usecols=cols)
            df.dropna(inplace=True)
            data_irish.append(df)
    df_irish = pd.concat(data_irish, axis=0, ignore_index=True)
    df_irish.dropna(inplace=True)
    df_irish['Handover'] = df_irish['CellID'].diff()
    df_irish['Handover'][df_irish['Handover'] != 0] = 1
    if type == 0:
        df_irish = df_irish[['Speed', 'RSRP', 'RSRQ', 'NRxRSRP', 'NRxRSRQ', 'DL_bitrate']]
        df_irish.columns = ['Speed', 'lte_rsrp', 'lte_rsrq', 'nr_rsrp', 'nr_rsrq', 'Throughput']
        df_irish1 = df_irish[pd.to_numeric(df_irish.nr_rsrq, errors='coerce').isnull()]
        df_irish.drop(df_irish1.index, inplace=True)
        df_irish1 = df_irish[pd.to_numeric(df_irish.lte_rsrq, errors='coerce').isnull()]
        df_irish.drop(df_irish1.index, inplace=True)
    else:
        df_irish = df_irish[['Speed', 'Handover', 'RSRP', 'DL_bitrate']]
        df_irish.columns = ['Speed', 'Handover', 'lte_rsrp', 'Throughput']
    # print(df_irish.head())
    return df_irish


def collect_data_lumos(type=1):
    column_name_2 = ['movingSpeed', 'lte_rssi', 'lte_rsrp', 'lte_rsrq', 'lte_rssnr', 'nr_ssRsrp', 'nr_ssRsrq',
                     'Throughput', 'tower_id']
    data_lumos = pd.read_csv('dataset/Lumos5G-v1.0/Lumos5G-v1.0.csv', usecols=[5, 8, 9, 10, 11, 12, 13, 15, 18])
    data_lumos.dropna(inplace=True)
    data_lumos['Handover'] = data_lumos['tower_id'].diff()
    data_lumos['Handover'][data_lumos['Handover'] != 0] = 1
    if type == 0:
        data_lumos = data_lumos[['movingSpeed', 'lte_rsrp', 'lte_rsrq', 'nr_ssRsrp', 'nr_ssRsrq', 'Throughput']]
        data_lumos.columns = ['Speed', 'lte_rsrp', 'lte_rsrq', 'nr_rsrp', 'nr_rsrq', 'Throughput']
    else:
        data_lumos = data_lumos[['movingSpeed', 'Handover', 'lte_rsrp', 'Throughput']]
        data_lumos.columns = ['Speed', 'Handover', 'lte_rsrp', 'Throughput']
    # print(data_lumos.head())
    return data_lumos


def collect_mn_wild():
    files = [x for x in os.listdir('dataset/merged-logs/')]
    mnfiles = []
    for file in files:
        if file[0:5] == 'S20UP':
            mnfiles.append(file)
    data_all = []
    for fileName in mnfiles:
        rootpath = 'dataset/merged-logs/'
        path = rootpath + fileName
        dataWild = pd.read_csv(path)
        data_all.append(dataWild)
        df_5Gwild = pd.concat(data_all, axis=0, ignore_index=True)
        df_5Gwild.dropna(inplace=True)

        mnWild = df_5Gwild[['movingSpeed', 'rsrp', 'nr_ssRsrp_avg', 'Throughput']]
        mnWild.columns = ['Speed', 'lte_rsrp', 'nr_rsrp', 'Throughput']
        return mnWild


def collect_df_central():
    data_central = pd.read_csv('dataset/centralised_4G.csv')
    data_central = data_central[['speed', 'Handover', 'rssi_strongest', 'Throughput tcp']]
    data_central.columns = ['Speed', 'Handover', 'lte_rsrp', 'Throughput']
    return data_central


def collect_df_merged():
    data_central = pd.read_csv('dataset/merged.csv')
    return data_central


def preprocess_data(data, sigma, test_split):
    print('Printing data here just before the error', data.head())
    x = np.array(data['Throughput'])
    y3 = gaussian_filter1d(x, sigma, order=0)
    data['Throughput'] = y3
    noise = x - y3
    train = data.iloc[0:int((1 - test_split) * data.shape[0])]
    test = data.iloc[int((1 - test_split) * data.shape[0]):]
    sc = StandardScaler()
    train = sc.fit_transform(train)
    test = sc.transform(test)

    return data, train, test, sc, x, noise


def timeseries_data(data, history_window, future_window):
    X = []
    Y = []
    i = 0
    while i < len(data) - history_window - future_window + 1:
        temp_df = data[i:i + history_window]
        target_variable = np.mean(data[i + history_window:i + history_window + future_window, 3])
        X.append(temp_df)
        Y.append(target_variable)

        i += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def create_model(shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(shape[1], shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(128))

    model.add(Dropout(0.2))
    model.add(Dense(1))
    sgd = optimizers.SGD(lr=0.3)
    model.compile(loss='msle', optimizer='adam')
    # print(model.summary())
    return model


def return_rf_model(iter):
    return RandomForestRegressor(bootstrap=param_list.param[2]['bootstrap'], max_depth=param_list.param[2]['max_depth'],
                                 max_features=param_list.param[2]['max_features'],
                                 min_samples_leaf=param_list.param[2]['min_samples_leaf'],
                                 min_samples_split=param_list.param[2]['min_samples_split'],
                                 n_estimators=param_list.param[2]['n_estimators'])


def lstm_split(dataset):
    data, train, test, sc, x, noise = preprocess_data(dataset, 1, 0.3)
    x_train, y_train = timeseries_data(train, 5, 1)
    x_test, y_test = timeseries_data(test, 5, 1)
    return x_train, x_test, y_train, y_test


def rf_split(dataset):
    labels = dataset.pop("Throughput")
    return train_test_split(dataset, labels, test_size=0.3)


def pred_rf_model(rf_model, x_test, y_test, traindataI, testdataI, modelType):
    y_pred = rf_model.predict(x_test)
    y_pred = y_pred.reshape(y_pred.shape[0])
    sourceModel = DATASETS[traindataI]
    testModel = DATASETS[testdataI]
    with open("dataset/Output.txt", "a") as text_file:
        text_file.write(f'{sourceModel} {modelType}: R2 Score with {testModel} test is {metrics.r2_score(y_test, y_pred)}\n')
    with open("dataset/Output.txt", "a") as text_file:
        text_file.write(f'{sourceModel} {modelType}: Corr coeff with {testModel} test {np.corrcoef(y_test, y_pred)}\n')
    with open("dataset/Output.txt", "a") as text_file:
        text_file.write(f'{sourceModel} {modelType}: MSE with {testModel} test {mean_squared_error(y_test, y_pred)}\n')

    print(f'{sourceModel} {modelType}: R2 Score with {testModel} test is {metrics.r2_score(y_test, y_pred)}')
    print(f'{sourceModel} {modelType}: Corr coeff with {testModel} test {np.corrcoef(y_test, y_pred)}')
    print(f'{sourceModel} {modelType}: MSE with {testModel} test {mean_squared_error(y_test, y_pred)}')


def evaluate(model, test_features, test_labels, base):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    mystr = 'Base' if base is True else 'Best'

    str1 = f'{mystr} Model Performance'
    str2 = 'Average Error: {:0.4f} degrees.'.format(np.mean(errors))
    str3 = 'Accuracy = {:0.2f}%.'.format(accuracy)
    with open("dataset/Output.txt", "a") as text_file:
        text_file.write(f'{mystr} Model Performance\n')
    with open("dataset/Output.txt", "a") as text_file:
        text_file.write('Average Error: {:0.4f} degrees.\n'.format(np.mean(errors)))
    with open("dataset/Output.txt", "a") as text_file:
        text_file.write('Accuracy = {:0.2f}%.\n'.format(accuracy))
    print(str1)
    print(str2)
    print(str3)
    return accuracy


def test_RF(dataset, datasetName, test_dataset1, test_dataset2, test_dataset3=None):
    x_train, x_test, y_train, y_test = rf_split(dataset)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=20, stop=2000, num=20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # print('Just a random grid')
    # pprint(random_grid)
    # rf_model = RandomForestRegressor(max_depth=50, n_estimators=70)
    # rf_model = RandomForestRegressor()
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # rf = RandomForestRegressor()
    rf_model = return_rf_model(datasetName)
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    # rf_random = GridSearchCV(estimator=rf, param_grid=random_grid, cv=3, verbose=2, n_jobs=-1)
    # Fit the random search model
    # rf_random.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    # str = f'So far best parameters after random search: {rf_random.best_params_}'
    # print(str)
    # with open("Output.txt", "a") as text_file:
    #     text_file.write(f'So far best parameters after random search: {rf_random.best_params_}\n')

    base_model = RandomForestRegressor(n_estimators=70, max_depth=15)
    base_model.fit(x_train, y_train)
    base_accuracy = evaluate(base_model, x_train, y_train, True)
    # best_random = rf_random.best_estimator_
    random_accuracy = evaluate(rf_model, x_train, y_train, False)
    str2 = 'Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy)
    with open("dataset/Output.txt", "a") as text_file:
        text_file.write('Improvement of {:0.2f}%.\n'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
    pred_rf_model(rf_model, x_train, y_train, datasetName, datasetName, modelType='RF')
    _, test1_x_test, _, test1_y_test = rf_split(test_dataset1)
    pred_rf_model(rf_model, test1_x_test, test1_y_test, datasetName, 1 if datasetName == 0 else 0, modelType='RF')
    _, test2_x_test, _, test2_y_test = rf_split(test_dataset2)
    pred_rf_model(rf_model, test2_x_test, test2_y_test, datasetName, 2 if datasetName in [0, 1] else 1, modelType='RF')
    if test_dataset3 is None:
        return
    else:
        _, test3_x_test, _, test3_y_test = rf_split(test_dataset3)
        pred_rf_model(rf_model, test3_x_test, test3_y_test, datasetName, 2, modelType='RF')


def test_LSTM(dataset, datasetName, test_dataset1, test_dataset2, test_dataset3=None):
    data, train, test, sc, x, noise = preprocess_data(dataset, 1, 0.3)
    x_train, y_train = timeseries_data(train, 5, 1)
    x_test, y_test = timeseries_data(test, 5, 1)
    model = create_model(x_train.shape)
    history = model.fit(x_train, y_train, epochs=25, batch_size=256,
                        validation_data=(x_test, y_test))
    pred_rf_model(model, x_test, y_test, datasetName, datasetName, modelType='LSTM')
    _, test1_x_test, _, test1_y_test = lstm_split(test_dataset1)
    pred_rf_model(model, test1_x_test, test1_y_test, datasetName, 1 if datasetName == 0 else 0, modelType='LSTM')
    _, test2_x_test, _, test2_y_test = lstm_split(test_dataset2)
    pred_rf_model(model, test2_x_test, test2_y_test, datasetName, 2 if datasetName in [0, 1] else 1, modelType='LSTM')
    if test_dataset3 is None:
        return
    else:
        _, test3_x_test, _, test3_y_test = lstm_split(test_dataset3)
        pred_rf_model(model, test3_x_test, test3_y_test, datasetName, 2, modelType='LSTM')


def train_model(datasetName):
    df_irish = collect_data_irish()
    df_lumos = collect_data_lumos()
    df_mnWild = collect_mn_wild()
    df_central = collect_df_central()
    df_merged = collect_df_merged()
    train_dataset = None
    test_dataset1 = None
    test_dataset2 = None
    test_dataset3 = None
    if datasetName == 0:
        train_dataset = df_irish
        test_dataset1 = df_lumos
        test_dataset2 = df_mnWild
    elif datasetName == 1:
        train_dataset = df_lumos
        test_dataset1 = df_irish
        test_dataset2 = df_mnWild
    elif datasetName == 2:
        train_dataset = df_mnWild
        test_dataset1 = df_irish
        test_dataset2 = df_lumos
    elif datasetName == 3:
        train_dataset = df_central
        test_dataset1 = df_irish
        test_dataset2 = df_lumos
        test_dataset3 = df_mnWild
    elif datasetName == 4:
        train_dataset = df_merged
        test_dataset1 = df_irish
        test_dataset2 = df_lumos
        test_dataset3 = df_mnWild
    # test_RF(train_dataset, datasetName, test_dataset1, test_dataset2, test_dataset3)
    test_LSTM(train_dataset, datasetName, test_dataset1, test_dataset2, test_dataset3)
    # df_irish = collect_data_irish(0)
    # print(df_irish)


if __name__ == '__main__':
    # print('Total datasets available', DATASETS)
    for i in range(len(DATASETS)):
        str = f"Starting the analysis with dataset {DATASETS[i]}"
        with open("dataset/Output.txt", "a") as text_file:
            text_file.write(f"Starting the analysis with dataset {DATASETS[i]}\n")
        print(str)
        train_model(i)