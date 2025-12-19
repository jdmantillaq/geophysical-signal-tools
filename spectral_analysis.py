def bandpass_filter(series, low_period, high_period):
    '''
    Apply a bandpass filter to a time series using the Fourier transform.

    Parameters:
        series (array-like): Input time series data.
        low_period (float): Lower bound of the period (inclusive).
        high_period (float): Upper bound of the period (inclusive).

    Returns:
        np.ndarray: Filtered time series.
    '''
    import numpy as np
    sampling_interval = 1
    mean_value = np.mean(series)
    detrended_series = series - mean_value

    freqs = np.fft.fftfreq(len(detrended_series), sampling_interval)
    periods = 1 / freqs

    # Mask frequencies outside the desired period band
    filter_mask = (np.abs(periods) >= low_period) & (
        np.abs(periods) <= high_period)

    fourier_coeffs = np.fft.fft(detrended_series)
    fourier_coeffs[~filter_mask] = 0

    filtered_series = np.fft.ifft(fourier_coeffs).real
    filtered_series += mean_value

    return filtered_series


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
    print(f"  Data shape: {data.shape} (time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]})")
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
    
    print(f"  Valid spatial points: {n_valid:,}/{n_space:,} ({pct_valid:.1f}%)")
    
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
            data_anomalies_2d[:, valid_points] = data_valid - seasonal_component
            print(f"  ✓ Solved for all {n_valid:,} points simultaneously")
        else:
            nan_count = np.sum(np.isnan(data_valid))
            print(f"\n  Status: Sparse data ({nan_count:,} missing values) - point-wise solve")
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
                    print(f"    Progress: {i+1:,}/{data_valid.shape[1]:,} processed")
            
            print(f"  ✓ Solved for {processed:,} / {data_valid.shape[1]:,} points")
    
    # Reshape back to original shape
    data_anomalies = data_anomalies_2d.reshape(orig_shape)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'=' * 60}\n")
    
    return data_anomalies