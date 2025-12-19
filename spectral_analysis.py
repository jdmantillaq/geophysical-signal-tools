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
