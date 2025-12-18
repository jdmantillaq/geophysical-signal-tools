def compute_eof(data):
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
    import numpy as np
    import scipy.linalg as la

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


def project_data(data, vec_prop):
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
    import numpy as np

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
