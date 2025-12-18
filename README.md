# Geophysical Signal Tools

Lightweight utilities for common geophysical time series and field analysis:
- Empirical Orthogonal Functions (EOF) for 3D fields (time, lat, lon)
- Basic Fourier spectral tools and a band‑pass filter for 1D series

## Modules

### `eof_analysis.py`
- **`eof_decomposition(data)`**: Computes EOFs from a 3D array or `xarray.DataArray` shaped as `(time, lat, lon)`.
	- **Input**: 3D data; NaN grid points are masked using the first time slice.
	- **Returns**: `(val_prop, vec_prop, eof, var_exp)`
		- `val_prop`: 1D eigenvalues
		- `vec_prop`: 2D eigenvectors in time space `(time, time)`
		- `eof`: 3D EOFs reshaped back to `(time, lat, lon)`
		- `var_exp`: 1D variance explained for each mode (percent)
- **`project_onto_eofs(data, vec_prop)`**: Projects the 3D data onto the provided eigenvectors to obtain mode time series/fields.
	- **Input**: 3D data `(time, lat, lon)`, `vec_prop` with shape `(time, time)`
	- **Returns**: Projected data with the same shape as the input.

Notes:
- The routines assume you provide data already preprocessed as desired (e.g., anomalies/detrended). No demeaning is applied inside `eof_decomposition`.
- NaN handling: grid cells that are NaN in the first time step are excluded from the analysis and restored on output.

### `spectral_analysis.py`
- **`bandpass_filter(series, low_period, high_period)`**: Bandpass filter a time series using Fourier transform and period bounds.
	- **Input**: 1D array‑like `series`; `low_period`, `high_period` in time units (assumes unit sampling interval).
	- **Returns**: Filtered `np.ndarray`.
- **`compute_power_spectrum(time_series)`**: Computes normalized power spectrum (% variance) via FFT.
	- **Returns**: `(periods, percent_variance)` arrays.
- **`plot_power_spectrum(serie)`**: Plot the power spectrum of a time series; returns the figure object.
	- **Returns**: `matplotlib.figure.Figure`.
- **`compute_harmonic_anomalies(data, n_harmonics=4, year_period=365.25)`**: Remove seasonal cycle via harmonic regression (vectorized).
	- **Input**: 3D data `(time, lat, lon)`, number of harmonics, period length (default: 365.25 days).
	- **Returns**: 3D anomalies with seasonal component removed.

## Quickstart

Using this repo as a simple source folder (no packaging required):

```python
# Option 1: If your script is in the same folder as the module files
from eof_analysis import eof_decomposition, project_onto_eofs
from spectral_analysis import bandpass_filter, compute_power_spectrum, plot_power_spectrum

# Option 2: Add the folder to sys.path at runtime
import sys
sys.path.append('/path/to/geophysical-signal-tools')
import eof_analysis as eof
```

### EOF example
```python
import numpy as np
from eof_analysis import eof_decomposition, project_onto_eofs

# Synthetic data: time=120, lat=36, lon=72
data = np.random.randn(120, 36, 72)
val_prop, vec_prop, eof, var_exp = eof_decomposition(data)

# Project back using eigenvectors (shape must be [time, time])
proj = project_onto_eofs(data, vec_prop)
```

### Spectral tools example
```python
import numpy as np
from spectral_analysis import (
		bandpass_filter,
		compute_power_spectrum,
		plot_power_spectrum,
)

t = np.arange(0, 512)
series = (
		np.sin(2*np.pi*t/32)          # 32‑step period
		+ 0.5*np.sin(2*np.pi*t/8)     # 8‑step period
		+ 0.2*np.random.randn(t.size) # noise
)

# Band‑pass between 10 and 40 steps
filtered = bandpass_filter(series, low_period=10, high_period=40)

# Spectrum (% variance by period)
periods, pct_var = compute_power_spectrum(series)

# Plot and get figure
fig = plot_power_spectrum(series)
```

## Requirements

- `numpy`
- `scipy` (for `eof_analysis` eigen decomposition)
- `matplotlib` (for plotting in `spectral_analysis`)
- `xarray` (optional: only if you pass a `DataArray` to EOF functions)

Install with pip:

```bash
pip install numpy scipy matplotlib xarray
```

## License

See the included LICENSE file for details.
