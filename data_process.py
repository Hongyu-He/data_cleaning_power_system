import pandas as pd
import numpy as np
from math import sqrt
import pickle
from sklearn.metrics import mean_squared_error


def import_data(path='data/wind.txt', save_to_csv=True, save_path='data/wind_power_gen.csv'):
    """
    Import data from given path. Save the imported data to a csv file by default.
    :param str save_path: Path to save the .csv file
    :param bool save_to_csv: Whether to save data to a .csv file
    :param str path: Path from which the data is imported
    :return: DataFrame
    """
    data = pd.read_table(path, header=None, index_col=1)  # indexed by the date
    data = data.drop([0], axis=1)
    if save_to_csv:
        data.to_csv(save_path)
    return data


def data_mask(mask_rate, dataseries):
    """Mask some data as missing value. Missing values are represented as -1

    :param mask_rate: mask data by the rate of mask_rate
    :param dataseries: data series to be masked
    :return: masked data, masked index
    """
    masked_series = dataseries.copy()  # Make a deep copy of original array
    length = len(masked_series)
    mask_num = round(length * mask_rate)
    rand_index = np.random.randint(low=0, high=length, size=mask_num)
    masked_series[rand_index] = -1
    return masked_series, rand_index


def map2matrix(matrix_size, index):
    """Map index in a time series to the corresponding index in the matrix.

    :param matrix_size:
    :param index:
    :return: index in the matrix
    """
    row_index = index // matrix_size[1]
    col_index = index % matrix_size[1]
    matrix_index = (row_index, col_index)
    return matrix_index


def mean_cal(matrix):
    """calculate the mean values of the matrix by the column.

    :param matrix: input matrix
    :return: mean values of each column
    """
    ave = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        temp = matrix[:, i]
        temp_pos = temp[temp > -1]
        ave[i] = temp_pos.mean()
    return ave


def rmse(gt, prediction):
    """calculate the RMSE (Root Mean Squared Error) of prediction

    :param gt: ground truth
    :param prediction: prediction
    :return: RMSE error
    """
    mse = mean_squared_error(gt, prediction)
    rmse_error = sqrt(mse)
    return rmse_error


def create_dataset(dataset, look_back=1):
    """Convert an array of values into a dataset matrix.

    :param dataset:
    :param look_back:
    :return:
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def save_item(save_data, save_dirname):
    """save the results

    :param save_dirname: where to save
    :param save_data: data to save
    """
    pickle_out = open(save_dirname, 'wb')
    pickle.dump(save_data, pickle_out)
    pickle_out.close()


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        contents = pickle.load(f)
    return contents
