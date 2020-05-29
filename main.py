from data_process import *
from model import *
import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Configuration
    parser = argparse.ArgumentParser(description="Power system Data Cleaning")
    parser.add_argument('--data_path', type=str, default="data/wind.txt",
                        help="Path of the raw data")
    parser.add_argument('--preprocess', type=bool, default=True,
                        help="Whether preprocess the raw data")
    parser.add_argument('--absent_rate', type=float, default=0.01, help="Proportion of missing data")
    parser.add_argument('--regression_win_size', type=int, default=2, help="Window size of regression imputation")

    args = parser.parse_args()

    # Import the data
    if args.preprocess:
        data = import_data(path=args.data_path)
    else:
        data = pd.read_csv("data/wind_power_gen.csv")
    describe = data.describe()

    # Convert to numpy array
    data_numpy = data.to_numpy()

    # Generate data series, flatten the matrix
    data_series = data_numpy.flatten()

    # Mask part of data as absent value
    data_masked, _ = data_mask(args.absent_rate, data_series)
    data_masked_matrix = np.resize(data_masked, (data_numpy.shape[0], data_numpy.shape[1]))
    index_mask = np.array([i for i, val in enumerate(data_masked == -1) if val])
    masked_value = data_series[index_mask]
    print("Data masked ", str(len(masked_value)))

    # Average Imputation
    # Calculate the mean values for each column
    average = mean_cal(data_masked_matrix)
    pred_average, _ = average_imputation(data_masked, average)
    rmse_average = rmse(masked_value, pred_average)
    print("Average Imputation Finished")
    print("RMSE of average imputation: ", str(rmse_average))

    # Last observation Imputation
    pred_loi, _ = last_observation_imputation(data_masked)
    rmse_loi = rmse(masked_value, pred_loi)
    print("Last Observation Imputation Finished")
    print("RMSE of last observation imputation: ", str(rmse_loi))

    # Random Imputation
    pred_rand, _ = rand_imputation(data_masked)
    rmse_rand = rmse(masked_value, pred_rand)
    print("Random Imputation Finished")
    print("RMSE of random imputation: ", str(rmse_rand))

    # Regression Imputation
    pred_regression, _ = regression_imputation(data_masked, window=args.regression_win_size)
    rmse_regression = rmse(masked_value, pred_regression)
    print("Regression Imputation Finished")
    print("RMSE of regression imputation: ", str(rmse_regression))

    # LSTM Imputation
    model, prediction, pred_lstm = lstm_imputation(data_series, data_masked)
    rmse_lstm = rmse(masked_value, pred_lstm)
    print("LSTM Imputation Finished")
    print("RMSE of LSTM imputation: ", str(rmse_lstm))

    rmse_list = [[rmse_average, rmse_loi, rmse_rand, rmse_regression, rmse_lstm]]
    rmse_results = pd.DataFrame(rmse_list, columns=['Mean', 'LOI', 'Random', 'Regression', 'LSTM'])
    rmse_results.to_csv('results/rmse_results.csv')

    # Save items
    save_item(data_masked, 'results/data_masked.pickle')
    save_item(index_mask, 'results/index_mask.pickle')
    save_item(masked_value, 'results/masked_value.pickle')
    save_item(average, 'results/average.pickle')
    save_item(pred_average, 'results/pred_average.pickle')
    save_item(pred_loi, 'results/pred_loi.pickle')
    save_item(pred_rand, 'results/pred_rand.pickle')
    save_item(pred_regression, 'results/pred_regression.pickle')
    save_item(pred_lstm, 'results/pred_lstm.pickle')
    save_item(prediction, 'results/lstm_prediction.pickle')
