"""Trend analysis utilities for 1D series and gridded geophysical data.

This module provides helpers to estimate and interpret monotonic trends using
the Mann-Kendall test, and to optionally remove linear trends from time series.
It supports both:

- Single-vector inputs (e.g., one station or one grid-point time series).
- Gridded xarray inputs (e.g., DataArray with dimensions such as
    ``(time, lat, lon)``) via vectorized ``xarray.apply_ufunc``.

Main functions
--------------
- ``mk_test``: computes Mann-Kendall significance and linear slope for a 1D
    series while handling ties and NaNs.
- ``run_mk_on_grid``: applies ``mk_test`` over a gridded field along a chosen
    time dimension and returns an ``xarray.Dataset`` with trend diagnostics.
- ``detrend_1d_with_slope``: removes a linear trend from a 1D series while
    preserving NaNs.
- ``infer_time_step_info``: infers representative temporal step information
    (e.g., month/day/hour) from a datetime coordinate.

Output conventions
------------------
``run_mk_on_grid`` returns a dataset containing:

- ``trend_flag``: ``+1`` for significant positive trend, ``-1`` for
    significant negative trend, ``NaN`` for not significant.
- ``z_mk``: normalized Mann-Kendall Z statistic.
- ``slope``: linear least-squares trend per inferred time step.
- ``detrended`` (optional): input field with linear trend removed at each
    grid point when ``detrend=True``.
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import norm, linregress


def mk_test(x, eps=1e-4, alpha=0.01):
    """Mann-Kendall trend test with tie correction.

    Returns
    -------
    tuple
        (trend_flag, z_mk, slope)
        - trend_flag: +1.0 or -1.0 if trend is significant at ``alpha``,
          else ``np.nan``
        - z_mk: Mann-Kendall normalized statistic
        - slope: linear trend slope from least-squares regression
    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    x = np.asarray(x, dtype=float).ravel()
    valid = ~np.isnan(x)
    t_idx = np.arange(x.size, dtype=float)[valid]
    x = x[valid]
    n = x.size

    # Degenerate series: not enough data for trend inference.
    if n < 2:
        return np.nan, np.nan, np.nan

    # Compute Kendall S without allocating an n x n matrix.
    S = 0.0
    for i in range(n - 1):
        diff = x[i + 1:] - x[i]
        diff[np.fabs(diff) <= eps] = 0.0
        S += np.sign(diff).sum()

    # Tie groups for variance correction (values equal within eps).
    x_sorted = np.sort(x)
    tie_counts = []
    current_count = 1
    for i in range(1, n):
        if np.fabs(x_sorted[i] - x_sorted[i - 1]) <= eps:
            current_count += 1
        else:
            if current_count > 1:
                tie_counts.append(current_count)
            current_count = 1
    if current_count > 1:
        tie_counts.append(current_count)

    t = np.asarray(tie_counts, dtype=int)
    sin_empates = n * (n - 1) * (2 * n + 5)
    con_empates = (t * (t - 1) * (2 * t + 5)).sum() if t.size else 0
    varS = float(sin_empates - con_empates) / 18.0

    if varS <= 0:
        Zmk = 0.0
    elif S > 0:
        Zmk = (S - 1.0) / np.sqrt(varS)
    elif S < 0:
        Zmk = (S + 1.0) / np.sqrt(varS)
    else:
        Zmk = 0.0

    z_crit = norm.ppf(1.0 - alpha / 2.0)
    slope, _, _, _, _ = linregress(t_idx, x)

    if np.fabs(Zmk) >= z_crit:
        trend_flag = 1.0 if Zmk > 0 else -1.0
        return trend_flag, float(Zmk), float(slope)
    return np.nan, float(Zmk), float(slope)


def detrend_1d_with_slope(x, slope):
    """Remove a linear trend from a 1D series while preserving NaNs."""
    x = np.asarray(x, dtype=float).ravel()
    out = np.full(x.shape, np.nan, dtype=float)

    if np.isnan(slope):
        return out

    valid = ~np.isnan(x)
    if valid.sum() == 0:
        return out

    t = np.arange(x.size, dtype=float)
    # Intercept from valid points so detrended anomalies are centered.
    intercept = np.nanmean(x[valid] - slope * t[valid])
    out[valid] = x[valid] - (slope * t[valid] + intercept)
    return out


def infer_time_step_info(time_coord):
    """Infer representative time step and readable unit from a time coordinate.

    Returns
    -------
    tuple
        (delta_time: pd.Timedelta | None, unit_label: str | None,
         unit_for_slope: str | None)
    """
    time_index = pd.DatetimeIndex(time_coord.values)
    if time_index.size < 2:
        return None, None, None

    freq = pd.infer_freq(time_index)
    if freq is not None:
        f = freq.upper()
        if "MS" in f or "ME" in f or f == "M":
            delta_time = time_index[1] - time_index[0]
            return delta_time, "month", "month"
        if "D" in f:
            delta_time = time_index[1] - time_index[0]
            return delta_time, "day", "day"
        if "H" in f:
            delta_time = time_index[1] - time_index[0]
            return delta_time, "hour", "hour"

    diffs = np.diff(time_index.values).astype("timedelta64[ns]")
    if diffs.size == 0:
        return None, None, None

    delta_ns = int(np.median(diffs.astype("int64")))
    delta_time = pd.to_timedelta(delta_ns, unit="ns")
    hours = delta_time.total_seconds() / 3600.0

    if np.isclose(hours % (24.0 * 30.0), 0.0):
        return delta_time, "month", "month"
    if np.isclose(hours % 24.0, 0.0):
        return delta_time, "day", "day"
    return delta_time, "hour", "hour"


def run_mk_on_grid(data_array, eps=1e-4, alpha=0.01,
                   time_dim='time', detrend=False):
    """Run mk_test over a 3D field with dims (time, lat, lon).

    If ``detrend=True``, includes a ``detrended`` variable in the output
    dataset with the linear trend removed at each grid point.
    """
    trend_flag, z_mk, slope_step = xr.apply_ufunc(
        mk_test,
        data_array,
        input_core_dims=[[time_dim]],
        output_core_dims=[[], [], []],
        kwargs={"eps": eps, "alpha": alpha},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )

    in_units = data_array.attrs.get("units")
    in_long_name = data_array.attrs.get("long_name", data_array.name)

    delta_time, _, time_step_unit = infer_time_step_info(data_array[time_dim])
    if delta_time is not None and delta_time.total_seconds() > 0:
        seconds_per_decade = 365.25 * 24.0 * 3600.0 * 10.0
        steps_per_decade = seconds_per_decade / delta_time.total_seconds()
        slope = slope_step * steps_per_decade
        slope_units = f"{in_units} / decade" if in_units else "1 / decade"
        slope_desc = (
            "Least-squares slope"
            # f"(converted from {time_step_unit or time_dim} steps)"
        )
    else:
        slope = slope_step
        slope_units = (
            f"{in_units} / {time_step_unit}" if (in_units and time_step_unit)
            else (f"{in_units} / {time_dim}_step" if in_units else f"1 / {time_dim}_step")
        )
        slope_desc = f"Least-squares slope per {time_step_unit or time_dim} step"

    ds_out = xr.Dataset(
        {
            "trend_flag": trend_flag,
            "z_mk": z_mk,
            "slope": slope,
        }
    )

    ds_out["trend_flag"].attrs = {
        "long_name": "Mann-Kendall trend direction",
        "description":
            "+1 significant positive trend, -1 significant negative trend, NaN not significant",
        "units": "Positive/Negative",
    }
    ds_out["z_mk"].attrs = {
        "long_name": "Mann-Kendall Z statistic",
    }

    ds_out["slope"].attrs = {
        "long_name": "Linear trend slope",
        "description": slope_desc,
        "units": slope_units,
    }

    if detrend:
        detrended = xr.apply_ufunc(
            detrend_1d_with_slope,
            data_array,
            slope_step,
            input_core_dims=[[time_dim], []],
            output_core_dims=[[time_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Keep exact input dimension order (e.g., time, lat, lon).
        detrended = detrended.transpose(*data_array.dims)
        detrended.name = "detrended"
        detrended.attrs = {
            **data_array.attrs,
            "long_name": f"Detrended {in_long_name}" if in_long_name else "Detrended field",
            "description": "Input field with linear trend removed at each grid point",
        }
        if in_units:
            detrended.attrs["units"] = in_units
        ds_out["detrended"] = detrended

    return ds_out
