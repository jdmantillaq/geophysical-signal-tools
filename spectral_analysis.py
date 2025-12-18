def fourier_bandpass_filter(series, low_period, high_period):
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


def compute_fourier_spectrum(time_series):
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


def plot_fourier_spectra(serie):
    """
    Plots the Fourier spectra of a time series.

    Args:
      series (array of int) - contains the measurements for each time step
    """
    import numpy as np
    import matplotlib.pyplot as plt
    serie = serie - serie.mean()

    # Compute the corresponding frequencies
    freq = np.fft.fftfreq(len(serie), 1)
    period = 1/freq
    fourier = np.fft.fft(serie)
    amplitud = np.abs(fourier)
    power = (amplitud ** 2)
    power_1 = (power/np.sum(power)) * np.var(serie)
    porcen_var = power_1 / np.var(serie) * 100.

    # Plot the magnitude of the FFT
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(period, porcen_var*2, color='k')
    ax.set_xscale('log', base=10)
    plt.xlabel('Period')
    plt.ylabel('Magnitude [variance]')
    plt.title('Fourier Spectra')
    plt.grid(True)
    plt.show()
