import numpy as np
from data_process import *
from sklearn.linear_model import LinearRegression
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def average_imputation(s, col_ave, missing_value=-1):
    """Data imputation with average value of each column.

    :param s: data serires with missing value
    :param col_ave: average value of each column
    :param missing_value: representation of missing values
    :return: imputation value
    """
    completed_series = s.copy()
    index_missing = np.array([i for i, val in enumerate(s == missing_value) if val])  # index of missing values
    col_index = index_missing % 96
    imputation = col_ave[col_index]
    completed_series[index_missing] = imputation
    return imputation, completed_series


def last_observation_imputation(s, missing_value=-1):
    """Data imputation with the last data observation.

    :param s: data series with missing value
    :param missing_value: representation of missing value
    :return: imputation value, completed data series
    """
    imputation = []
    completed_series = s.copy()
    for i, item in enumerate(s):
        if item == missing_value:
            # if the first value is missing
            if i == 0:
                imputation.append(s[s != missing_value][0])  # search for first valid value
                completed_series[i] = imputation[0]
            else:
                imputation.append(completed_series[i-1])
                completed_series[i] = completed_series[i-1]
    imputation = np.array(imputation)
    return imputation, completed_series


def rand_imputation(s, missing_value=-1):
    """Randomly pick a value as an estimation of missing value

    :param s: data series with missing value
    :param missing_value: representation of missing value
    :return: imputation value, completed data series
    """
    completed_series = s.copy()

    # Avoid picking another missing value
    index = np.linspace(0, len(s)-1, len(s))
    index_missing = np.array([i for i, val in enumerate(s == missing_value) if val])  # index of missing values
    # filter index to exclude the indexes of missing values
    index_filtered = np.array([i for i in index if i not in index_missing], dtype=int)

    num_missing = len(index_missing)
    rand_index = np.random.randint(low=0, high=len(index_filtered), size=num_missing)
    rand_index = index_filtered[rand_index]

    imputation = s[rand_index]
    completed_series[index_missing] = imputation

    return imputation, completed_series


def regression_imputation(s, missing_value=-1, window=5):
    """Data imputation with linear regression.

    :param s: data series with missing data
    :param missing_value: representation of missing value
    :param window: window size of linear imputation
    :return: imputation value, completed data series
    """
    index_missing = np.array([i for i, val in enumerate(s == missing_value) if val])
    completed_series = s.copy()
    imputation = []
    for i, item in enumerate(index_missing):
        # For training data at the left side of the missing point
        # At the beginning of the series
        if i == 0:
            left = max(item - window, 0) if item != 0 else 0
        else:
            left = max(item - window, 0)
        y_train_left = completed_series[left:item]

        # For training data at the right side of the missing point
        # At the end of the series
        if i == (len(index_missing)-1):
            right = min(item + window, len(s)) if item != (len(s)-1) else len(s)-1
        else:
            right = min(item + window, index_missing[i+1])
        y_train_right = s[item+1:right] if item != (len(s)-1) else np.array([])
        y_train = np.append(y_train_left, y_train_right)

        # In case of continuing missing data
        if len(y_train) == 0:
            completed_series[i] = 0
            imputation.append(0)
            continue

        x_train_left = np.linspace(0, len(y_train_left)-1, len(y_train_left))
        x_train_right = np.linspace(len(y_train_left) + 1, len(y_train_left) + len(y_train_right), len(y_train_right))
        x_train = np.append(x_train_left, x_train_right)

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        # Fit the regressor
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        # Make the prediction
        y_pred = regressor.predict([[len(y_train_left)]])
        y_pred = y_pred.squeeze()
        completed_series[i] = y_pred
        imputation.append(y_pred)

    return np.array(imputation), completed_series


def lstm_imputation(dataset, s, missing_value=-1, train=True, look_back=1, time_step=1):
    """Data imputation using LSTM time series prediction.

    :param dataset: training data
    :param s: data series with missing data
    :param missing_value: representation of missing value
    :param train: train model or load model
    :param look_back: training parameter
    :param time_step: training parameter
    :return: model, prediction, imputation
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_reshape = dataset.reshape(-1, 1)
    data_normalized = scaler.fit_transform(data_reshape)
    trainX, trainY = create_dataset(data_normalized, look_back)

    # The LSTM network expects the input data to be provided with a specific array structure in the form of:
    # [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], time_step, trainX.shape[1]))

    if train:
        # Build & train the model
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=64)

        # save the model
        model.save('model/lstm_model')
        print("LSTM model training finished")
        print("Model saved to model/lstm_model")
    else:
        model = load_model('model/lstm_model')
    # Imputation
    index_missing = np.array([i for i, val in enumerate(s == missing_value) if val])
    pred = model.predict(trainX)
    pred = scaler.inverse_transform(pred)
    pred = pred.squeeze()
    imputation = pred[index_missing]

    return model, pred, imputation

