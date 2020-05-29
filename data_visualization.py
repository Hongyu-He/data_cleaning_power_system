import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from data_process import *

if __name__ == '__main__':
    # Import Data
    data = import_data(path="data/wind.txt")
    describe = data.describe()
    # Convert to numpy array
    data_numpy = data.to_numpy()
    # Generate data series, flatten the matrix
    data_series = data_numpy.flatten()

    rmse_results = pd.read_csv('results/rmse_results.csv')
    data_masked = load_pickle('results/data_masked.pickle')
    index_mask = load_pickle('results/index_mask.pickle')
    masked_value = load_pickle('results/masked_value.pickle')
    average = load_pickle('results/average.pickle')
    pred_average = load_pickle('results/pred_average.pickle')
    pred_loi = load_pickle('results/pred_loi.pickle')
    pred_rand = load_pickle('results/pred_rand.pickle')
    pred_regression = load_pickle('results/pred_regression.pickle')
    pred_lstm = load_pickle('results/pred_lstm.pickle')
    prediction = load_pickle('results/lstm_prediction.pickle')

    data_masked_matrix = np.resize(data_masked, (data_numpy.shape[0], data_numpy.shape[1]))

    day1 = data.loc['2018-03-10'].to_numpy()
    day2 = data.loc['2018-12-29'].to_numpy()
    day3 = data.loc['2018-05-26'].to_numpy()
    x = np.linspace(0, 95, 96)
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    axis[0].set_title('2018-03-10')
    axis[0].plot(x, day1)
    axis[0].set_ylabel('power')
    axis[0].set_xlabel('time')
    axis[1].set_title('2018-12-29')
    axis[1].plot(x, day2)
    axis[1].set_ylabel('power')
    axis[1].set_ylabel('time')
    axis[2].set_title('2018-05-26')
    axis[2].plot(x, day3)
    axis[2].set_ylabel('power')
    axis[2].set_ylabel('time')
    plt.show()

    fig, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].set_title('Average Power')
    axis[0].plot(x, average)
    axis[0].set_ylabel('Power')
    axis[0].set_xlabel('Time')
    describe_T = describe.T
    describe_T = describe_T.to_numpy()
    axis[1].set_title('Statistic Description')
    axis[1].plot(describe_T[:, 1:])
    axis[1].legend(['Mean', 'STD', 'Min', '25%', '50%', '75%', 'Max'])
    axis[1].set_ylabel('Power')
    axis[1].set_xlabel('Time')
    plt.show()

    # Plot RMSE of regression imputation based on different window size
    rmse_regression = load_pickle('results/rmse_regressions.pickle')
    window_size = [1, 2, 3, 4, 5, 6]
    plt.figure()
    plt.bar(window_size, rmse_regression)
    plt.title('RMSE of Regression Imputations')
    plt.xlabel('window size')
    plt.ylabel('RMSE')
    plt.show()

    # Models Comparison
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    standard = np.arange(0, max(masked_value)+10, 1)
    axis[0].set_xlim(0, max(masked_value)+10)
    axis[0].set_xlabel('Ground Truth')
    axis[0].set_ylim(0, max(masked_value)+10)
    axis[0].set_ylabel('Mean Imputation')
    axis[0].scatter(masked_value, pred_average)
    axis[0].plot(standard, standard, 'r--')

    axis[1].set_xlim(0, max(masked_value)+10)
    axis[1].set_xlabel('Ground Truth')
    axis[1].set_ylim(0, max(masked_value)+10)
    axis[1].set_ylabel('LOI Imputation')
    axis[1].scatter(masked_value, pred_loi)
    axis[1].plot(standard, standard, 'r--')

    axis[2].set_xlim(0, max(masked_value)+10)
    axis[2].set_xlabel('Ground Truth')
    axis[2].set_ylim(0, max(masked_value)+10)
    axis[2].set_ylabel('Random Imputation')
    axis[2].scatter(masked_value, pred_rand)
    axis[2].plot(standard, standard, 'r--')

    plt.show()

    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    axis[0].set_xlim(0, max(masked_value)+10)
    axis[0].set_xlabel('Ground Truth')
    axis[0].set_ylim(0, max(masked_value)+10)
    axis[0].set_ylabel('Regression Imputation')
    axis[0].scatter(masked_value, pred_regression)
    axis[0].plot(standard, standard, 'r--')

    axis[1].set_xlim(0, max(masked_value)+10)
    axis[1].set_xlabel('Ground Truth')
    axis[1].set_ylim(0, max(masked_value)+10)
    axis[1].set_ylabel('LSTM Imputation')
    axis[1].scatter(masked_value, pred_lstm)
    axis[1].plot(standard, standard, 'r--')

    method = ['Mean', 'LOI', 'Random', 'Regression', 'LSTM']
    rmse_numpy = rmse_results.to_numpy()
    rmse_numpy = rmse_numpy[0, 1:]
    axis[2].set_xlabel('Methods')
    axis[2].set_ylabel('RMSE')
    axis[2].bar(method, rmse_numpy)

