from data_process import *
from model import *

data = import_data(save_to_csv=False)
data_numpy = data.to_numpy()

# Generate data series, flatten the matrix
data_series = data_numpy.flatten()

# Mask part of data as absent value
data_masked, index_mask = data_mask(0.01, data_series)
data_masked_matrix = np.resize(data_masked, (data_numpy.shape[0], data_numpy.shape[1]))

# Imputation
imputation, _ = regression_imputation(data_masked)

