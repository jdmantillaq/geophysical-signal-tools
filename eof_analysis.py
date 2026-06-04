
import numpy as np
import scipy.linalg as la


def eof_decomposition(data):
    """
    Computes Empirical Orthogonal Functions (EOFs) from the given data.

    Parameters:
    - data (3D array or xarray DataArray): Input data with dimensions 
                                            [time, latitude, longitude]

    Returns:
    - val_prop (1D array): Eigenvalues
    - vec_prop (2D array): Eigenvectors
    - eof (3D array): Empirical Orthogonal Functions reshaped back to
                                            the original dimensions
    - var_exp (1D array): Variance explained by each mode
    """

    # Verify if input data is numpy array or xarray object
    if not isinstance(data, np.ndarray):
        try:
            # Try to extract numpy array fromxarray DataArray
            data = data.values
        except AttributeError:
            raise ValueError("Input data should be either a numpy array or '\
                'an xarray DataArray.")

    # Extract the shape dimensions
    ntime, nlat, nlon = data.shape

    # Reshape the 3D data into 2D
    data_reshape = data.reshape(ntime, nlat*nlon)

    # Find positions of NaN values in the reshaped data
    idx_nan_array = np.where(np.isnan(data_reshape[0, :]))

    # Remove NaN values to create a new matrix without NaN
    data_reshape_No_NaN = np.delete(data_reshape, idx_nan_array[0], 1)

    # Check for NaN or Inf values in the data
    if np.isnan(data_reshape_No_NaN).any() or \
            np.isinf(data_reshape_No_NaN).any():
        pass
        # raise ValueError("Input data contains NaN or Inf values.")

    # Calculate the covariance matrix
    matriz_cov = np.dot(data_reshape_No_NaN, data_reshape_No_NaN.T)

    # Compute eigenvalues and eigenvectors of the covariance matrix
    val_prop, vec_prop = la.eig(matriz_cov)

    # Calculate the total variance
    sum_evals = np.sum(val_prop)

    # Calculate the percentage of variance explained by each mode
    var_exp = (val_prop / sum_evals) * 100

    # Project the eigenvectors onto the data to obtain the EOFs
    eof = np.dot(vec_prop.T, data_reshape_No_NaN)

    # Initialize a space filled with NaNs to store the EOF information
    eof_con_NaN = np.copy(data_reshape)*np.nan

    # Identify positions without NaN values
    dim_espacio = np.arange(data_reshape.shape[1])
    Not_Nan = np.setdiff1d(dim_espacio, idx_nan_array)

    # Store the non-NaN EOF information
    eof_con_NaN[:, Not_Nan] = eof

    # Update the EOF variable
    eof = eof_con_NaN

    # Reshape the EOFs back to their original 3D shape
    eof = eof.reshape(ntime, nlat, nlon)

    return val_prop, vec_prop, eof, var_exp


def project_onto_eofs(data, vec_prop):
    """
    Project 3D data onto a subspace defined by a set of eigenvectors.

    Parameters:
    - data (numpy.ndarray or xarray.DataArray): Input 3D data array with
                dimensions (time, latitude, longitude).
    - vec_prop (numpy.ndarray): Eigenvectors defining the subspace.
                Should have the same number of time steps as the input data.

    Returns:
    - numpy.ndarray: Projected data in the subspace defined
                by the eigenvectors, with the same dimensions as
                the input data.

    Raises:
    - ValueError: If the input data is not a numpy array or an
                xarray DataArray.
    - ValueError: If the time dimensions of the input data and
                eigenvectors do not match.
    """

    # Verify if input data is numpy array or xarray object
    if not isinstance(data, np.ndarray):
        try:
            # Try to extract numpy array fromxarray DataArray
            data = data.values
        except AttributeError:
            raise ValueError("Input data should be either a numpy array or '\
                'an xarray DataArray.")

    # Extract the shape dimensions
    ntime, nlat, nlon = data.shape

    if (ntime, ntime) != vec_prop.shape:
        raise ValueError(f"Time longitude does not coincide:\n"
                         f'\tdata_time:{ntime},  {vec_prop.shape}')

    # Reshape the 3D data into 2D
    data_reshape = data.reshape(ntime, nlat*nlon)

    # Find positions of NaN values in the reshaped data
    idx_nan_array = np.where(np.isnan(data_reshape[0, :]))

    # Remove NaN values to create a new matrix without NaN
    data_reshape_No_NaN = np.delete(data_reshape, idx_nan_array[0], 1)
    # Calculate the covariance matrix
    eof = np.dot(vec_prop.T, data_reshape_No_NaN)

    eof_con_NaN = np.copy(data_reshape)*np.nan
    # Identify positions without NaN values
    dim_espacio = np.arange(data_reshape.shape[1])
    Not_Nan = np.setdiff1d(dim_espacio, idx_nan_array)

    # Store the non-NaN EOF information
    eof_con_NaN[:, Not_Nan] = eof

    # Update the EOF variable
    eof = eof_con_NaN

    # Reshape the EOFs back to their original 3D shape
    eof = eof.reshape(ntime, nlat, nlon)

    return eof


def build_lagged_matrix(data_matrix, max_lag):
    '''
    Build the time-lag augmented matrix used for Extended EOF analysis.

    Given a 2D (time, space) matrix, this returns a matrix where each row
    contains the original spatial map concatenated with its lagged copies,
    so the resulting shape is (time - max_lag, space * (max_lag + 1)).
    '''
    n_time, n_space = data_matrix.shape

    lagged_matrix = np.zeros(
        (n_time - max_lag, n_space * (max_lag + 1))) * np.nan

    for i in range(max_lag + 1):
        if (max_lag - i) != 0:
            lagged_matrix[:, i*n_space: i*n_space +
                          n_space] = data_matrix[i:i - max_lag, :]
        else:
            lagged_matrix[:, i*n_space: i*n_space +
                          n_space] = data_matrix[i:, :]

    return lagged_matrix


def compute_eeof(data_3d, data_matrix, max_lag):
    """
    Compute Time-Extended Empirical Orthogonal Functions (EEOF).

    Usage:
    ------
    MAX_LAG = 12

    # Get the shape of the anomalies
    n_time, n_lat, n_lon = anomalies.shape

    data_matrix = anomalies.reshape(n_time, n_lat*n_lon)
    print(f"data_matrix   shape: (time={data_matrix.shape[0]}, space={data_matrix.shape[1]})")


    lagged_matrix = build_lagged_matrix(data_matrix, MAX_LAG)
    print(f"lagged_matrix shape: (time={lagged_matrix.shape[0]}, space={lagged_matrix.shape[1]})")


    eigenvalues, eigenvectors, eeofs, variance_explained = \
        compute_eeof(anomalies, data_matrix, MAX_LAG)

    Parameters
    ----------
    data_3d : ndarray
        Original 3D data array with shape (time, lat, lon). Used only to
        recover the spatial shape when reshaping the EOFs.
    data_matrix : ndarray
        2D (time, space) matrix obtained by reshaping `data_3d`.
    max_lag : int
        Maximum lag (in time steps) used to build the augmented matrix.
        The number of lag blocks is `max_lag + 1`.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues of the covariance matrix.
    eigenvectors : ndarray
        Eigenvectors (principal components in time) of the covariance matrix.
    eeof_reshaped : ndarray
        EEOF spatial patterns with shape (max_lag + 1, n_modes, lat, lon).
    variance_explained : ndarray
        Percentage of variance explained by each oscillation mode.
    """

    lagged_matrix = build_lagged_matrix(data_matrix, max_lag)

    nan_indices = np.where(np.isnan(lagged_matrix[0]))

    lagged_matrix_no_nan = np.delete(lagged_matrix, nan_indices, 1)

    cov_matrix = np.dot(lagged_matrix_no_nan, lagged_matrix_no_nan.T)

    eigenvalues, eigenvectors = la.eig(cov_matrix)

    total_variance = np.sum(eigenvalues)

    variance_explained = (eigenvalues / total_variance) * 100

    # Project eigenvectors onto the data to obtain the EOF spatial patterns
    eof_modes = np.dot(eigenvectors.T, lagged_matrix_no_nan)

    eof_with_nan = np.copy(lagged_matrix) * np.nan

    all_space_indices = np.arange(lagged_matrix.shape[1])
    valid_indices = np.setdiff1d(all_space_indices, nan_indices)

    eof_with_nan[:, valid_indices] = eof_modes

    eof_modes = eof_with_nan

    n_modes, n_space_total = eof_modes.shape

    eeof_reshaped = [np.copy(data_3d) * np.nan] * (max_lag + 1)

    # Split the EOF matrix into its lag blocks
    n_space_per_lag = int(n_space_total / (max_lag + 1))
    eof_split = np.zeros((max_lag + 1, n_modes, n_space_per_lag)) * np.nan

    for i in range(max_lag + 1):
        eof_split[i, :, :] = eof_modes[:, i *
                                       n_space_per_lag: (i + 1)*n_space_per_lag]

    for i in range(max_lag + 1):
        eeof_reshaped[i] = eof_split[i, :, :].reshape(
            data_3d.shape[0] - max_lag, data_3d.shape[1], data_3d.shape[2])

    eeof_reshaped = np.array(eeof_reshaped)

    return eigenvalues, eigenvectors, eeof_reshaped, variance_explained
