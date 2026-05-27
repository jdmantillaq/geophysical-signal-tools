#%%
def bandpass_filter(series, low_period, high_period):
    '''
    Apply a bandpass filter to a time series using the Fourier transform.
    Keeps frequencies corresponding to periods between low_period and high_period.

    Parameters:
        series (array-like): Input time series data.
        low_period (float): Lower bound of the period (shorter period, higher frequency).
        high_period (float): Upper bound of the period (longer period, lower frequency).

    Returns:
        np.ndarray: Filtered time series.
    '''
    import numpy as np
    sampling_interval = 1
    mean_value = np.mean(series)
    detrended_series = series - mean_value

    freqs = np.fft.fftfreq(len(detrended_series), sampling_interval)
    # Handle division by zero
    periods = np.where(freqs != 0, 1 / np.abs(freqs), np.inf)

    # Keep frequencies with periods in [low_period, high_period]
    filter_mask = (periods >= low_period) & (periods <= high_period)

    fourier_coeffs = np.fft.fft(detrended_series)
    fourier_coeffs[~filter_mask] = 0

    filtered_series = np.fft.ifft(fourier_coeffs).real
    filtered_series += mean_value

    return filtered_series


def lowpass_filter(series, cutoff_period):
    '''
    Apply a lowpass filter to a time series using the Fourier transform.
    It passes signals with a frequency lower than a selected cutoff frequency
    and attenuates signals with frequencies higher than the cutoff frequency.

    Parameters:
        series (array-like): Input time series data.
        cutoff_period (float): Cutoff period for the lowpass filter.

    Returns:
        np.ndarray: Filtered time series.
    '''
    import numpy as np
    sampling_interval = 1
    mean_value = np.mean(series)
    detrended_series = series - mean_value

    freqs = np.fft.fftfreq(len(detrended_series), sampling_interval)
    # Handle division by zero
    periods = 1 / np.where(freqs != 0, freqs, np.inf)

    # Mask frequencies with periods SHORTER than cutoff (higher frequencies)
    filter_mask = np.abs(periods) < cutoff_period

    fourier_coeffs = np.fft.fft(detrended_series)
    fourier_coeffs[filter_mask] = 0  # Remove high frequencies

    filtered_series = np.fft.ifft(fourier_coeffs).real
    filtered_series += mean_value

    return filtered_series


def highpass_filter(series, cutoff_period):
    '''
    Apply a highpass filter to a time series using the Fourier transform.

    Parameters:
        series (array-like): Input time series data.
        cutoff_period (float): Cutoff period for the highpass filter.

    Returns:
        np.ndarray: Filtered time series.
    '''
    import numpy as np
    sampling_interval = 1
    mean_value = np.mean(series)
    detrended_series = series - mean_value

    freqs = np.fft.fftfreq(len(detrended_series), sampling_interval)
    periods = 1 / np.where(freqs != 0, freqs, np.inf)

    # Mask frequencies below the cutoff period
    filter_mask = np.abs(periods) > cutoff_period

    fourier_coeffs = np.fft.fft(detrended_series)
    fourier_coeffs[filter_mask] = 0

    filtered_series = np.fft.ifft(fourier_coeffs).real
    filtered_series += mean_value

    return filtered_series

def bandpass_filter_3d(data, low_period, high_period):
    """
    Apply bandpass filter to 3D array using Fourier transform (vectorized version).

    Parameters:
    -----------
    data : ndarray
        3D array with shape (time, lat, lon)
    low_period : float
        Lower bound of the period for the bandpass filter (in time units)
    high_period : float
        Upper bound of the period for the bandpass filter (in time units)

    Returns:
    --------
    filtered_data : ndarray
        Bandpass filtered data, same shape as input
    """
    import numpy as np
    import time

    print("=" * 60)
    print("Applying bandpass filter")  # Fixed: was "lowpass"
    print("=" * 60)
    start_time = time.time()

    print(f"\nInput configuration:")
    print(
        f"  Data shape: {data.shape} (time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]})")
    print(f"  Low period: {low_period}, High period: {high_period} time units")  # Fixed capitalization

    # Store original shape
    orig_shape = data.shape
    n_time = orig_shape[0]

    # Reshape to (time, space) for vectorized operations
    data_2d = data.reshape(n_time, -1)
    n_space = data_2d.shape[1]

    print(f"\nProcessing:")
    print(f"  Reshaped to: {data_2d.shape} (time x space)")

    # Setup filter once (same for all spatial points)
    sampling_interval = 1
    freqs = np.fft.fftfreq(n_time, sampling_interval)
    periods = 1 / np.where(freqs != 0, np.abs(freqs), np.inf)

    # Keep frequencies with periods in the band [low_period, high_period]
    filter_mask = (periods >= low_period) & (periods <= high_period)

    print(
        f"  Frequencies to keep: {np.sum(filter_mask)} / {len(filter_mask)}")

    # Handle NaN values - identify valid points
    valid_points = ~np.all(np.isnan(data_2d), axis=0)
    n_valid = np.sum(valid_points)
    pct_valid = 100 * n_valid / n_space

    print(
        f"  Valid spatial points: {n_valid:,}/{n_space:,} ({pct_valid:.1f}%)")

    # Initialize output
    filtered_data_2d = np.full_like(data_2d, np.nan)

    if n_valid > 0:
        # Extract valid data
        data_valid = data_2d[:, valid_points]

        # Check if data is complete (no NaN in time series)
        if not np.any(np.isnan(data_valid)):
            print(f"\n  Status: Complete data - vectorized filtering")

            # Remove mean for each spatial point
            mean_values = np.mean(data_valid, axis=0)
            detrended_data = data_valid - mean_values

            # Apply FFT to all spatial points at once
            fourier_coeffs = np.fft.fft(detrended_data, axis=0)

            # Apply filter mask (broadcast across spatial dimension)
            fourier_coeffs[~filter_mask, :] = 0

            # Inverse FFT
            filtered_valid = np.fft.ifft(fourier_coeffs, axis=0).real

            # Add mean back
            filtered_valid += mean_values

            filtered_data_2d[:, valid_points] = filtered_valid
            print(f"  ✓ Filtered all {n_valid:,} points simultaneously")
        else:
            nan_count = np.sum(np.isnan(data_valid))
            print(
                f"\n  Status: Sparse data ({nan_count:,} missing values) - point-wise filtering")

            # Need to handle NaN values point by point
            processed = 0
            for i in range(data_valid.shape[1]):
                point_data = data_valid[:, i]
                valid_time = ~np.isnan(point_data)

                if np.sum(valid_time) > 10:  # need enough points for filtering
                    # Extract valid time points
                    valid_series = point_data[valid_time]

                    # Apply bandpass filter to this series
                    mean_value = np.mean(valid_series)
                    detrended_series = valid_series - mean_value

                    # Create filter for this length
                    n_valid_time = len(valid_series)
                    freqs_i = np.fft.fftfreq(n_valid_time, sampling_interval)
                    periods_i = 1 / \
                        np.where(freqs_i != 0, np.abs(freqs_i), np.inf)
                    filter_mask_i = (periods_i >= low_period) & (periods_i <= high_period)

                    # Apply filter
                    fourier_coeffs_i = np.fft.fft(detrended_series)
                    fourier_coeffs_i[~filter_mask_i] = 0  # Use complement of mask
                    filtered_series = np.fft.ifft(
                        fourier_coeffs_i).real + mean_value

                    # Store back
                    filtered_data_2d[valid_time,
                                     valid_points][:, i] = filtered_series
                    processed += 1

                if (i + 1) % 100 == 0:
                    print(
                        f"    Progress: {i+1:,}/{data_valid.shape[1]:,} processed")

            print(
                f"  ✓ Filtered {processed:,} / {data_valid.shape[1]:,} points")

    # Reshape back to original shape
    filtered_data = filtered_data_2d.reshape(orig_shape)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'=' * 60}\n")

    return filtered_data

def lowpass_filter_3d(data, cutoff_period):
    """
    Apply lowpass filter to 3D array using Fourier transform (vectorized version).

    Parameters:
    -----------
    data : ndarray
        3D array with shape (time, lat, lon)
    cutoff_period : float
        Cutoff period for the lowpass filter (in time units)

    Returns:
    --------
    filtered_data : ndarray
        Lowpass filtered data, same shape as input
    """
    import numpy as np
    import time

    print("=" * 60)
    print("Applying lowpass filter")
    print("=" * 60)
    start_time = time.time()

    print(f"\nInput configuration:")
    print(
        f"  Data shape: {data.shape} (time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]})")
    print(f"  Cutoff period: {cutoff_period} time units")

    # Store original shape
    orig_shape = data.shape
    n_time = orig_shape[0]

    # Reshape to (time, space) for vectorized operations
    data_2d = data.reshape(n_time, -1)
    n_space = data_2d.shape[1]

    print(f"\nProcessing:")
    print(f"  Reshaped to: {data_2d.shape} (time x space)")

    # Setup filter once (same for all spatial points)
    sampling_interval = 1
    freqs = np.fft.fftfreq(n_time, sampling_interval)
    periods = 1 / np.where(freqs != 0, np.abs(freqs), np.inf)

    # Mask frequencies with periods longer than cutoff (lower frequencies)
    filter_mask = np.abs(periods) < cutoff_period

    print(
        f"  Frequencies to remove: {np.sum(filter_mask)} / {len(filter_mask)}")

    # Handle NaN values - identify valid points
    valid_points = ~np.all(np.isnan(data_2d), axis=0)
    n_valid = np.sum(valid_points)
    pct_valid = 100 * n_valid / n_space

    print(
        f"  Valid spatial points: {n_valid:,}/{n_space:,} ({pct_valid:.1f}%)")

    # Initialize output
    filtered_data_2d = np.full_like(data_2d, np.nan)

    if n_valid > 0:
        # Extract valid data
        data_valid = data_2d[:, valid_points]

        # Check if data is complete (no NaN in time series)
        if not np.any(np.isnan(data_valid)):
            print(f"\n  Status: Complete data - vectorized filtering")

            # Remove mean for each spatial point
            mean_values = np.mean(data_valid, axis=0)
            detrended_data = data_valid - mean_values

            # Apply FFT to all spatial points at once
            fourier_coeffs = np.fft.fft(detrended_data, axis=0)

            # Apply filter mask (broadcast across spatial dimension)
            fourier_coeffs[filter_mask, :] = 0

            # Inverse FFT
            filtered_valid = np.fft.ifft(fourier_coeffs, axis=0).real

            # Add mean back
            filtered_valid += mean_values

            filtered_data_2d[:, valid_points] = filtered_valid
            print(f"  ✓ Filtered all {n_valid:,} points simultaneously")
        else:
            nan_count = np.sum(np.isnan(data_valid))
            print(
                f"\n  Status: Sparse data ({nan_count:,} missing values) - point-wise filtering")

            # Need to handle NaN values point by point
            processed = 0
            for i in range(data_valid.shape[1]):
                point_data = data_valid[:, i]
                valid_time = ~np.isnan(point_data)

                if np.sum(valid_time) > 10:  # need enough points for filtering
                    # Extract valid time points
                    valid_series = point_data[valid_time]

                    # Apply highpass filter to this series
                    mean_value = np.mean(valid_series)
                    detrended_series = valid_series - mean_value

                    # Create filter for this length
                    n_valid_time = len(valid_series)
                    freqs_i = np.fft.fftfreq(n_valid_time, sampling_interval)
                    periods_i = 1 / \
                        np.where(freqs_i != 0, np.abs(freqs_i), np.inf)
                    filter_mask_i = np.abs(periods_i) > cutoff_period

                    # Apply filter
                    fourier_coeffs_i = np.fft.fft(detrended_series)
                    fourier_coeffs_i[filter_mask_i] = 0
                    filtered_series = np.fft.ifft(
                        fourier_coeffs_i).real + mean_value

                    # Store back
                    filtered_data_2d[valid_time,
                                     valid_points][:, i] = filtered_series
                    processed += 1

                if (i + 1) % 100 == 0:
                    print(
                        f"    Progress: {i+1:,}/{data_valid.shape[1]:,} processed")

            print(
                f"  ✓ Filtered {processed:,} / {data_valid.shape[1]:,} points")

    # Reshape back to original shape
    filtered_data = filtered_data_2d.reshape(orig_shape)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'=' * 60}\n")

    return filtered_data


def highpass_filter_3d(data, cutoff_period):
    """
    Apply highpass filter to 3D array using Fourier transform (vectorized version).

    Parameters:
    -----------
    data : ndarray
        3D array with shape (time, lat, lon)
    cutoff_period : float
        Cutoff period for the highpass filter (in time units)

    Returns:
    --------
    filtered_data : ndarray
        Highpass filtered data, same shape as input
    """
    import numpy as np
    import time

    print("=" * 60)
    print("Applying highpass filter")
    print("=" * 60)
    start_time = time.time()

    print(f"\nInput configuration:")
    print(
        f"  Data shape: {data.shape} (time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]})")
    print(f"  Cutoff period: {cutoff_period} time units")

    # Store original shape
    orig_shape = data.shape
    n_time = orig_shape[0]

    # Reshape to (time, space) for vectorized operations
    data_2d = data.reshape(n_time, -1)
    n_space = data_2d.shape[1]

    print(f"\nProcessing:")
    print(f"  Reshaped to: {data_2d.shape} (time x space)")

    # Setup filter once (same for all spatial points)
    sampling_interval = 1
    freqs = np.fft.fftfreq(n_time, sampling_interval)
    periods = 1 / np.where(freqs != 0, np.abs(freqs), np.inf)

    # Mask frequencies with periods longer than cutoff (lower frequencies)
    filter_mask = np.abs(periods) > cutoff_period

    print(
        f"  Frequencies to remove: {np.sum(filter_mask)} / {len(filter_mask)}")

    # Handle NaN values - identify valid points
    valid_points = ~np.all(np.isnan(data_2d), axis=0)
    n_valid = np.sum(valid_points)
    pct_valid = 100 * n_valid / n_space

    print(
        f"  Valid spatial points: {n_valid:,}/{n_space:,} ({pct_valid:.1f}%)")

    # Initialize output
    filtered_data_2d = np.full_like(data_2d, np.nan)

    if n_valid > 0:
        # Extract valid data
        data_valid = data_2d[:, valid_points]

        # Check if data is complete (no NaN in time series)
        if not np.any(np.isnan(data_valid)):
            print(f"\n  Status: Complete data - vectorized filtering")

            # Remove mean for each spatial point
            mean_values = np.mean(data_valid, axis=0)
            detrended_data = data_valid - mean_values

            # Apply FFT to all spatial points at once
            fourier_coeffs = np.fft.fft(detrended_data, axis=0)

            # Apply filter mask (broadcast across spatial dimension)
            fourier_coeffs[filter_mask, :] = 0

            # Inverse FFT
            filtered_valid = np.fft.ifft(fourier_coeffs, axis=0).real

            # Add mean back
            filtered_valid += mean_values

            filtered_data_2d[:, valid_points] = filtered_valid
            print(f"  ✓ Filtered all {n_valid:,} points simultaneously")
        else:
            nan_count = np.sum(np.isnan(data_valid))
            print(
                f"\n  Status: Sparse data ({nan_count:,} missing values) - point-wise filtering")

            # Need to handle NaN values point by point
            processed = 0
            for i in range(data_valid.shape[1]):
                point_data = data_valid[:, i]
                valid_time = ~np.isnan(point_data)

                if np.sum(valid_time) > 10:  # need enough points for filtering
                    # Extract valid time points
                    valid_series = point_data[valid_time]

                    # Apply highpass filter to this series
                    mean_value = np.mean(valid_series)
                    detrended_series = valid_series - mean_value

                    # Create filter for this length
                    n_valid_time = len(valid_series)
                    freqs_i = np.fft.fftfreq(n_valid_time, sampling_interval)
                    periods_i = 1 / \
                        np.where(freqs_i != 0, np.abs(freqs_i), np.inf)
                    filter_mask_i = np.abs(periods_i) > cutoff_period

                    # Apply filter
                    fourier_coeffs_i = np.fft.fft(detrended_series)
                    fourier_coeffs_i[filter_mask_i] = 0
                    filtered_series = np.fft.ifft(
                        fourier_coeffs_i).real + mean_value

                    # Store back
                    filtered_data_2d[valid_time,
                                     valid_points][:, i] = filtered_series
                    processed += 1

                if (i + 1) % 100 == 0:
                    print(
                        f"    Progress: {i+1:,}/{data_valid.shape[1]:,} processed")

            print(
                f"  ✓ Filtered {processed:,} / {data_valid.shape[1]:,} points")

    # Reshape back to original shape
    filtered_data = filtered_data_2d.reshape(orig_shape)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'=' * 60}\n")

    return filtered_data


def compute_power_spectrum(time_series):
    """
    Computes the normalized power spectrum (percentage of variance) of a
    time series using the Fourier transform.

    Args:
        time_series (array-like): The input time series data.

    Returns:
        tuple: (periods, percent_variance) where
            periods (np.ndarray): Array of periods corresponding to the Fourier
                frequencies.
            percent_variance (np.ndarray): Percentage of variance explained by
                each frequency component.
    """
    import numpy as np
    sampling_interval = 1
    mean_value = np.mean(time_series)
    detrended_series = time_series - mean_value

    freqs = np.fft.fftfreq(len(detrended_series), sampling_interval)
    periods = 1 / freqs

    fourier_transform = np.fft.fft(detrended_series)
    amplitude = np.abs(fourier_transform)
    power = amplitude ** 2
    normalized_power = (power / np.sum(power)) * np.var(detrended_series)
    percent_variance = (normalized_power / np.var(detrended_series)) * 100.0

    return periods, percent_variance


def plot_power_spectrum(serie):
    """
    Plots the Fourier spectra of a time series.

    Args:
      serie (array-like): Input time series data.

    Returns:
      matplotlib.figure.Figure: The figure object containing the plot.
    """
    import matplotlib.pyplot as plt

    # Compute spectrum using the dedicated function
    periods, percent_variance = compute_power_spectrum(serie)

    # Plot the magnitude of the FFT
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(periods, percent_variance * 2, color='k')
    ax.set_xscale('log', base=10)
    ax.set_xlabel('Period')
    ax.set_ylabel('Explained variance [%]')
    ax.set_title('Fourier Spectra')
    ax.grid(True)

    return fig, ax


def compute_harmonic_anomalies(data, n_harmonics=4, year_period=365.25):
    """
    Remove seasonal cycle using harmonic regression (vectorized version).

    Parameters:
    -----------
    data : ndarray
        3D array with shape (time, lat, lon)
    n_harmonics : int
        Number of harmonics to remove (default: 4)
    year_period : float
        Period of the seasonal cycle (default: 365.25 days)

    Returns:
    --------
    anomalies : ndarray
        Data with seasonal cycle and harmonics removed, same shape as input
    """
    import numpy as np
    import time

    print("=" * 60)
    print("Removing seasonal cycle with harmonic regression")
    print("=" * 60)
    start_time = time.time()

    print(f"\nInput configuration:")
    print(
        f"  Data shape: {data.shape} (time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]})")
    print(f"  Harmonics: {n_harmonics}")
    print(f"  Year period: {year_period} days")

    # Store original shape
    orig_shape = data.shape
    n_time = orig_shape[0]

    # Reshape to (time, space) for vectorized operations
    data_2d = data.reshape(n_time, -1)
    n_space = data_2d.shape[1]

    print(f"\nProcessing:")
    print(f"  Reshaped to: {data_2d.shape} (time x space)")

    # Build design matrix once (same for all spatial points)
    t = np.arange(n_time)
    X = np.ones((n_time, 1 + 2*n_harmonics))

    for h in range(1, n_harmonics + 1):
        X[:, 2*h-1] = np.cos(2*np.pi*h*t / year_period)
        X[:, 2*h] = np.sin(2*np.pi*h*t / year_period)

    print(f"  Design matrix: {X.shape} (time x features)")

    # Handle NaN values - identify valid points
    valid_points = ~np.all(np.isnan(data_2d), axis=0)
    n_valid = np.sum(valid_points)
    pct_valid = 100 * n_valid / n_space

    print(
        f"  Valid spatial points: {n_valid:,}/{n_space:,} ({pct_valid:.1f}%)")

    # Initialize output
    data_anomalies_2d = np.full_like(data_2d, np.nan)

    if n_valid > 0:
        # Extract valid data
        data_valid = data_2d[:, valid_points]

        # For each valid point, handle any remaining NaN in time series
        # If all points are complete, we can do a single lstsq
        if not np.any(np.isnan(data_valid)):
            print(f"\n  Status: Complete data - vectorized solve")
            # Single least squares solve for all spatial points at once
            coeffs = np.linalg.lstsq(X, data_valid, rcond=None)[0]
            seasonal_component = X @ coeffs
            data_anomalies_2d[:, valid_points] = data_valid - \
                seasonal_component
            print(f"  ✓ Solved for all {n_valid:,} points simultaneously")
        else:
            nan_count = np.sum(np.isnan(data_valid))
            print(
                f"\n  Status: Sparse data ({nan_count:,} missing values) - point-wise solve")
            # Need to handle NaN values point by point
            processed = 0
            for i in range(data_valid.shape[1]):
                point_data = data_valid[:, i]
                valid_time = ~np.isnan(point_data)

                if np.sum(valid_time) >= X.shape[1]:  # enough data points
                    coeffs = np.linalg.lstsq(X[valid_time],
                                             point_data[valid_time],
                                             rcond=None)[0]
                    seasonal = X @ coeffs
                    data_anomalies_2d[valid_time, valid_points][:, i] = \
                        point_data[valid_time] - seasonal[valid_time]
                    processed += 1

                if (i + 1) % 100 == 0:
                    print(
                        f"    Progress: {i+1:,}/{data_valid.shape[1]:,} processed")

            print(
                f"  ✓ Solved for {processed:,} / {data_valid.shape[1]:,} points")

    # Reshape back to original shape
    data_anomalies = data_anomalies_2d.reshape(orig_shape)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'=' * 60}\n")

    return data_anomalies


# def remove_seasonal_cycle_harmonic(data, n_harmonics=4, year_period=365.25,
#                                    method='normal'):
#     """
#     Remove seasonal cycle using harmonic regression (vectorized version).
    
#     This method fits sine/cosine harmonics to the data and removes the
#     fitted seasonal component, leaving anomalies.
    
#     Parameters:
#     -----------
#     data : ndarray
#         3D array with shape (time, lat, lon)
#     n_harmonics : int
#         Number of harmonic pairs (sin/cos) to fit (default: 4)
#     year_period : float
#         Period of the seasonal cycle in time units (default: 365.25 days)
#     method : str
#         Solver method: 'lstsq' for least squares (default) or 'normal' for normal equation
    
#     Returns:
#     --------
#     anomalies : ndarray
#         Data with seasonal cycle removed, same shape as input
#     """
#     import numpy as np
#     import time

#     # Validate method parameter
#     if method not in ['lstsq', 'normal']:
#         raise ValueError(f"method must be 'lstsq' or 'normal', got '{method}'")

#     print("=" * 80)
#     print(f"Removing seasonal cycle with harmonic regression ({method} method)")
#     print("=" * 80)
#     start_time = time.time()
    
#     print(f"\nInput configuration:")
#     print(f"  Data shape: {data.shape} (time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]})")
#     print(f"  Harmonics: {n_harmonics}")
#     print(f"  Year period: {year_period} time units")
#     print(f"  Method: {method}")
    
#     # Store original shape
#     orig_shape = data.shape
#     n_time = orig_shape[0]
    
#     # Reshape to (time, space) for vectorized operations
#     data_2d = data.reshape(n_time, -1)
#     n_space = data_2d.shape[1]
    
#     print(f"\nProcessing:")
#     print(f"  Reshaped to: {data_2d.shape} (time x space)")
    
#     # Build design matrix (same for all spatial points)
#     t = np.arange(n_time)
#     X = np.ones((n_time, 2*n_harmonics + 1))
    
#     j = 1
#     for i in range(1, n_harmonics + 1):
#         X[:, j] = np.sin(i * 2 * np.pi * t / year_period)
#         X[:, j+1] = np.cos(i * 2 * np.pi * t / year_period)
#         j += 2
    
#     print(f"  Design matrix: {X.shape} (time x features: 1 constant + {2*n_harmonics} harmonics)")
    
#     # Handle NaN values - identify valid points
#     valid_points = ~np.all(np.isnan(data_2d), axis=0)
#     n_valid = np.sum(valid_points)
#     pct_valid = 100 * n_valid / n_space
    
#     print(f"  Valid spatial points: {n_valid:,}/{n_space:,} ({pct_valid:.1f}%)")
    
#     # Initialize output
#     anomalies_2d = np.full_like(data_2d, np.nan)
    
#     if n_valid > 0:
#         # Extract valid data
#         data_valid = data_2d[:, valid_points]
        
#         # Check if data is complete (no NaN in time series)
#         if not np.any(np.isnan(data_valid)):
#             print(f"\n  Status: Complete data - vectorized regression")
            
#             # Solve using selected method
#             if method == 'lstsq':
#                 # Least squares method
#                 coeffs = np.linalg.lstsq(X, data_valid, rcond=None)[0]
#                 seasonal_component = np.dot(X, coeffs)
#             else:  # method == 'normal'
#                 # Normal equation: C = (X^T X)^-1 X^T data_valid
#                 XtX_inv = np.linalg.inv(np.dot(X.T, X))
#                 coeffs = np.dot(XtX_inv, np.dot(X.T, data_valid))
#                 seasonal_component = np.dot(X, coeffs)
            
#             anomalies_2d[:, valid_points] = data_valid - seasonal_component
#             print(f"  ✓ Processed all {n_valid:,} points simultaneously")
#         else:
#             nan_count = np.sum(np.isnan(data_valid))
#             print(f"\n  Status: Sparse data ({nan_count:,} missing values) - point-wise regression")
            
#             # Need to handle NaN values point by point
#             processed = 0
#             for i in range(data_valid.shape[1]):
#                 point_data = data_valid[:, i]
#                 valid_time = ~np.isnan(point_data)
                
#                 # Need enough data points to fit the model
#                 if np.sum(valid_time) >= X.shape[1]:
#                     # Fit regression on valid time points only
#                     X_valid = X[valid_time, :]
#                     y_valid = point_data[valid_time]
                    
#                     # Solve using selected method
#                     if method == 'lstsq':
#                         # Least squares method
#                         coeffs = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
#                         seasonal = np.dot(X_valid, coeffs)
#                     else:  # method == 'normal'
#                         # Solve Normal equation: C = (X^T X)^-1 X^T y 
#                         #   to get harmonic coefficients.
#                         XtX_inv = np.linalg.inv(np.dot(X_valid.T, X_valid))
#                         coeffs = np.dot(XtX_inv, np.dot(X_valid.T, y_valid))
#                         seasonal = np.dot(X_valid, coeffs)
                    
#                     anomalies_2d[valid_time, valid_points][:, i] = y_valid - seasonal
#                     processed += 1
                
#                 if (i + 1) % 100 == 0:
#                     print(f"    Progress: {i+1:,}/{data_valid.shape[1]:,} processed")
            
#             print(f"  ✓ Processed {processed:,} / {data_valid.shape[1]:,} points")
    
#     # Reshape back to original shape
#     anomalies = anomalies_2d.reshape(orig_shape)
    
#     elapsed = time.time() - start_time
#     print(f"\n{'=' * 80}")
#     print(f"Completed in {elapsed:.2f} seconds")
#     print(f"{'=' * 80}\n")
    
#     return anomalies

def remove_seasonal_cycle_harmonic(data, n_harmonics=4, year_period=365.25,
                                   method='normal'):
    """
    Remove seasonal cycle using harmonic regression (vectorized version).
    
    This method fits sine/cosine harmonics to the data and removes the
    fitted seasonal component, leaving anomalies.
    
    Parameters:
    -----------
    data : ndarray
        3D array with shape (time, lat, lon)
    n_harmonics : int
        Number of harmonic pairs (sin/cos) to fit (default: 4)
    year_period : float
        Period of the seasonal cycle in time units (default: 365.25 days)
    method : str
        Solver method: 'lstsq' for least squares (default) or 'normal' for normal equation
    
    Returns:
    --------
    anomalies : ndarray
        Data with seasonal cycle removed, same shape as input
    """
    import numpy as np
    import time

    # Validate method parameter
    if method not in ['lstsq', 'normal']:
        raise ValueError(f"method must be 'lstsq' or 'normal', got '{method}'")

    print("=" * 80)
    print(f"Removing seasonal cycle with harmonic regression ({method} method)")
    print("=" * 80)
    start_time = time.time()
    
    print(f"\nInput configuration:")
    print(f"  Data shape: {data.shape} (time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]})")
    print(f"  Harmonics: {n_harmonics}")
    print(f"  Year period: {year_period} time units")
    print(f"  Method: {method}")
    
    # Store original shape
    orig_shape = data.shape
    n_time = orig_shape[0]
    
    # Reshape to (time, space) for vectorized operations
    data_2d = data.reshape(n_time, -1)
    n_space = data_2d.shape[1]
    
    print(f"\nProcessing:")
    print(f"  Reshaped to: {data_2d.shape} (time x space)")
    
    # Build design matrix (same for all spatial points)
    t = np.arange(n_time)
    X = np.ones((n_time, 2*n_harmonics + 1))
    
    j = 1
    for i in range(1, n_harmonics + 1):
        X[:, j] = np.sin(i * 2 * np.pi * t / year_period)
        X[:, j+1] = np.cos(i * 2 * np.pi * t / year_period)
        j += 2
    
    print(f"  Design matrix: {X.shape} (time x features: 1 constant + {2*n_harmonics} harmonics)")
    
    # Handle NaN values - identify valid points
    valid_points = ~np.all(np.isnan(data_2d), axis=0)
    n_valid = np.sum(valid_points)
    pct_valid = 100 * n_valid / n_space
    
    print(f"  Valid spatial points: {n_valid:,}/{n_space:,} ({pct_valid:.1f}%)")
    
    # Initialize output
    anomalies_2d = np.full_like(data_2d, np.nan)
    
    if n_valid > 0:
        # Extract valid data
        data_valid = data_2d[:, valid_points]
        
        # Check if data is complete (no NaN in time series)
        if not np.any(np.isnan(data_valid)):
            print(f"\n  Status: Complete data - vectorized regression")
            
            # Solve using selected method
            if method == 'lstsq':
                coeffs = np.linalg.lstsq(X, data_valid, rcond=None)[0]
                seasonal_component = np.dot(X, coeffs)
            else:  # method == 'normal'
                XtX_inv = np.linalg.inv(np.dot(X.T, X))
                coeffs = np.dot(XtX_inv, np.dot(X.T, data_valid))
                seasonal_component = np.dot(X, coeffs)
            
            anomalies_2d[:, valid_points] = data_valid - seasonal_component
            print(f"  ✓ Processed all {n_valid:,} points simultaneously")
        else:
            nan_count = np.sum(np.isnan(data_valid))
            print(f"\n  Status: Sparse data ({nan_count:,} missing values) - point-wise regression")
            
            # Pre-compute scalar spatial indices for direct assignment
            valid_indices = np.where(valid_points)[0]
            
            processed = 0
            for i in range(data_valid.shape[1]):
                point_data = data_valid[:, i]
                valid_time = ~np.isnan(point_data)
                
                # Need enough data points to fit the model
                if np.sum(valid_time) >= X.shape[1]:
                    X_valid = X[valid_time, :]
                    y_valid = point_data[valid_time]
                    
                    if method == 'lstsq':
                        coeffs = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                    else:  # method == 'normal'
                        XtX_inv = np.linalg.inv(np.dot(X_valid.T, X_valid))
                        coeffs = np.dot(XtX_inv, np.dot(X_valid.T, y_valid))
                    
                    seasonal = np.dot(X_valid, coeffs)
                    
                    # Use scalar spatial index to avoid boolean broadcasting error
                    anomalies_2d[valid_time, valid_indices[i]] = y_valid - seasonal
                    processed += 1
    
    # Reshape back to original shape
    anomalies = anomalies_2d.reshape(orig_shape)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'=' * 80}\n")
    
    return anomalies

def remove_linear_variability(x, y):
    """Remove the linear component of y explained by one or more indices.

    Parameters
    ----------
    x : array-like, shape (time,) or (time, n_predictors)
        Predictor/index time series used to regress out linear variability.
    y : array-like
        Data with time on axis 0. Supported shapes:
        - (time,) for a single time series
        - (time, lat, lon) for a 3D field

    Returns
    -------
    residual : ndarray
        y with the linearly related component removed.
    signal : ndarray
        Reconstructed linear signal related to x.
    beta : ndarray
        Regression coefficient(s):
        - shape (n_predictors,) for 1D y
        - shape (n_predictors, lat, lon) for 3D y
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim not in (1, 3):
        raise ValueError('y must be either 1D (time,) or 3D (time, lat, lon).')

    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError('x must be 1D (time,) or 2D (time, n_predictors).')

    n_time = y.shape[0]
    if x.shape[0] != n_time:
        raise ValueError('Time dimension mismatch between y and x.')

    # Require predictor values at all times; if needed, pre-fill before call.
    if np.isnan(x).any():
        raise ValueError('x contains NaN values; please fill or remove them before regression.')

    # Remove predictor means so beta captures anomalies, then solve in least squares sense.
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    xtx = x_centered.T @ x_centered
    if np.linalg.matrix_rank(xtx) < xtx.shape[0]:
        raise ValueError('Predictors are rank-deficient; cannot estimate unique coefficients.')

    if y.ndim == 1:
        valid = ~np.isnan(y)
        if np.sum(valid) <= x_centered.shape[1]:
            raise ValueError('Not enough valid samples in y to fit regression.')

        y_mean = np.nanmean(y)
        y_centered = y - y_mean

        beta = np.linalg.lstsq(x_centered[valid, :], y_centered[valid], rcond=None)[0]
        signal = x_centered @ beta
        residual = y - signal
    else:
        y_shape = y.shape
        y_2d = y.reshape(n_time, -1)
        n_space = y_2d.shape[1]

        signal_2d = np.full_like(y_2d, np.nan)
        beta_2d = np.full((x_centered.shape[1], n_space), np.nan)

        # Fast path when no NaNs are present in y.
        if not np.isnan(y_2d).any():
            y_mean = np.mean(y_2d, axis=0, keepdims=True)
            y_centered = y_2d - y_mean
            beta_2d = np.linalg.lstsq(x_centered, y_centered, rcond=None)[0]
            signal_2d = x_centered @ beta_2d
            residual_2d = y_2d - signal_2d
        else:
            for i in range(n_space):
                yi = y_2d[:, i]
                valid = ~np.isnan(yi)
                if np.sum(valid) <= x_centered.shape[1]:
                    continue

                yi_mean = np.nanmean(yi)
                yi_centered = yi - yi_mean
                beta_i = np.linalg.lstsq(x_centered[valid, :], yi_centered[valid], rcond=None)[0]
                signal_2d[:, i] = x_centered @ beta_i
                beta_2d[:, i] = beta_i

            residual_2d = y_2d - signal_2d

        signal = signal_2d.reshape(y_shape)
        residual = residual_2d.reshape(y_shape)
        beta = beta_2d.reshape((x_centered.shape[1],) + y_shape[1:])

    return residual, signal, beta

def remove_trailing_mean(data, window=120):
    """
    Remove interannual/decadal variability and trends by subtracting a trailing
    mean at each time step (vectorized via cumsum).

    For each time t, subtracts the mean of days [t-window, t-1] from data[t].
    The first `window` time steps are set to NaN as there is insufficient
    prior data to compute the mean.

    Parameters:
    -----------
    data : ndarray
        3D array with shape (time, lat, lon)
    window : int
        Number of prior days to average (default: 120)

    Returns:
    --------
    filtered : ndarray
        Data with trailing mean removed, same shape as input.
        First `window` time steps will be NaN.
    """
    import numpy as np

    n_time = data.shape[0]

    # Replace NaN with 0 for summation; track valid counts separately
    nan_mask = np.isnan(data)
    data_filled = np.where(nan_mask, 0.0, data)
    valid = (~nan_mask).astype(float)

    # Cumulative sums along time axis, prepend a zero slice so that
    # cumsum_padded[t] = sum(data[0:t]), making window subtraction clean
    zero = np.zeros((1,) + data.shape[1:])
    cumsum  = np.concatenate([zero, np.cumsum(data_filled, axis=0)], axis=0)
    cumcount = np.concatenate([zero, np.cumsum(valid,       axis=0)], axis=0)

    # sum(data[t-window : t]) = cumsum[t] - cumsum[t-window]
    # for t = window, window+1, ..., n_time-1
    trailing_sum   = cumsum[window:n_time]   - cumsum[:n_time - window]
    trailing_count = cumcount[window:n_time] - cumcount[:n_time - window]

    # Avoid division by zero where the entire window is NaN
    trailing_mean = np.where(trailing_count > 0, trailing_sum / trailing_count, np.nan)

    # Subtract; NaNs in data[window:] are preserved naturally (NaN - x = NaN)
    filtered = np.full_like(data, np.nan)
    filtered[window:] = data[window:] - trailing_mean

    return filtered


def crox_corr(Serie_a, Serie_b, lag=10):
    import numpy as np
    from scipy.stats import pearsonr

    serie_a = np.asarray(Serie_a, dtype=float).ravel()
    serie_b = np.asarray(Serie_b, dtype=float).ravel()

    if lag < 0:
        raise ValueError('lag must be >= 0')

    # Align lengths first so lag slicing is consistent.
    n = min(len(serie_a), len(serie_b))
    serie_a = serie_a[:n]
    serie_b = serie_b[:n]

    lags = np.arange(-lag, lag + 1, dtype=int)
    corr = np.full(lags.shape, np.nan, dtype=float)

    for idx, lag in enumerate(lags):
        if lag < 0:
            x = serie_a[-lag:]
            y = serie_b[:n + lag]
        elif lag > 0:
            x = serie_a[:n - lag]
            y = serie_b[lag:]
        else:
            x = serie_a
            y = serie_b

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        # Pearson is undefined for constant inputs.
        if np.std(x_valid) == 0 or np.std(y_valid) == 0:
            continue

        corr[idx] = pearsonr(x_valid, y_valid)[0]

    return lags, corr

if __name__ == "__main__":
    # Example usage or test cases can be added here
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    
    olr_xr = xr.open_dataset('olr.day.mean_1979_2000_30S30N.nc')
    
    anomalies_normal = remove_seasonal_cycle_harmonic(olr_xr['olr'].values,
                                               n_harmonics=4,
                                               year_period=365.25,
                                               method='normal')
    

    
    anomalies_xr = xr.DataArray(anomalies_normal,
                               coords=olr_xr['olr'].coords,
                               dims=olr_xr['olr'].dims)
    anomalies_xr.name = 'olr'
    anomalies_xr.attrs = olr_xr['olr'].attrs    
    anomalies_xr[0].plot()

#%%