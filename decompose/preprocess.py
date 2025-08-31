import polars as pl
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import pywt
from scipy import signal
import matplotlib.pyplot as plt


def check_stationarity(series: pl.Series, significance_level: float = 0.05, window_length: int = 100000, stationarity_threshold: float = 0.95) -> dict:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test on consecutive subsequences.
    
    For long time series, this function applies the ADF test to all consecutive subsequences of specified length
    and considers the entire series stationary if at least stationarity_threshold proportion of subsequences are stationary.
    
    Args:
        series (pl.Series): The time series to test
        significance_level (float): Significance level for the test (default: 0.05)
        window_length (int): Length of consecutive subsequences to test (default: 100000)
        stationarity_threshold (float): Minimum proportion of stationary subsequences to consider 
                                       entire series stationary (default: 0.95)
    
    Returns:
        dict: Dictionary containing test results with keys:
            - 'is_stationary': bool indicating if series is stationary overall
            - 'stationary_ratio': proportion of stationary subsequences
            - 'total_windows': total number of windows tested
            - 'stationary_windows': number of stationary windows
            - 'window_results': list of individual window test results
            - 'interpretation': human-readable interpretation
    """
    # Remove null values and convert to numpy array
    clean_series = series.drop_nulls().to_numpy()
    
    if len(clean_series) < window_length:
        # For short series, use original single-window approach
        if len(clean_series) < 10:
            raise ValueError("Series too short for ADF test (minimum 10 observations required)")
        
        adf_result = adfuller(clean_series, autolag='AIC')
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        used_lag = adf_result[2]
        critical_values = adf_result[4]
        is_stationary = p_value < significance_level
        
        return {
            'is_stationary': is_stationary,
            'stationary_ratio': 1.0 if is_stationary else 0.0,
            'total_windows': 1,
            'stationary_windows': 1 if is_stationary else 0,
            'window_results': [{
                'start_idx': 0,
                'end_idx': len(clean_series),
                'is_stationary': is_stationary,
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'used_lag': used_lag
            }],
            'interpretation': _interpret_windowed_adf_results(1.0 if is_stationary else 0.0, 1, 1 if is_stationary else 0, stationarity_threshold)
        }
    
    # For long series, use windowing approach
    window_results = []
    stationary_count = 0
    total_windows = 0
    
    # Calculate total number of windows for progress tracking
    total_expected_windows = len(range(0, len(clean_series) - window_length + 1, window_length))
    print(f"Testing stationarity on {total_expected_windows} consecutive windows of length {window_length:,}")
    
    # Create consecutive windows
    for window_idx, start_idx in enumerate(range(0, len(clean_series) - window_length + 1, window_length)):
        end_idx = start_idx + window_length
        window_data = clean_series[start_idx:end_idx]
        
        # Print progress update
        if window_idx % max(1, total_expected_windows // 10) == 0 or window_idx == total_expected_windows - 1:
            progress_pct = (window_idx + 1) / total_expected_windows * 100
            print(f"Progress: {window_idx + 1}/{total_expected_windows} windows ({progress_pct:.1f}%) - Current ratio: {stationary_count}/{total_windows} stationary")
        
        # Perform ADF test on this window
        try:
            adf_result = adfuller(window_data, autolag='AIC')
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            used_lag = adf_result[2]
            critical_values = adf_result[4]
            is_window_stationary = p_value < significance_level
            
            window_results.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'is_stationary': is_window_stationary,
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'used_lag': used_lag
            })
            
            if is_window_stationary:
                stationary_count += 1
            total_windows += 1
            
        except Exception as e:
            # Skip windows that cause errors in ADF test
            window_results.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'is_stationary': False,
                'error': str(e)
            })
            total_windows += 1
    
    # Calculate overall stationarity
    stationary_ratio = stationary_count / total_windows if total_windows > 0 else 0.0
    is_overall_stationary = stationary_ratio >= stationarity_threshold
    
    return {
        'is_stationary': is_overall_stationary,
        'stationary_ratio': stationary_ratio,
        'total_windows': total_windows,
        'stationary_windows': stationary_count,
        'window_results': window_results,
        'interpretation': _interpret_windowed_adf_results(stationary_ratio, total_windows, stationary_count, stationarity_threshold)
    }


def _interpret_adf_results(adf_stat: float, p_value: float, critical_values: dict, alpha: float) -> str:
    """
    Provide a human-readable interpretation of ADF test results.
    
    Args:
        adf_stat: ADF test statistic
        p_value: p-value of the test
        critical_values: critical values dictionary
        alpha: significance level
    
    Returns:
        str: Interpretation of the test results
    """
    if p_value < alpha:
        interpretation = f"Series is stationary (p-value: {p_value:.4f} < {alpha}). "
        interpretation += "We reject the null hypothesis of a unit root."
    else:
        interpretation = f"Series is non-stationary (p-value: {p_value:.4f} >= {alpha}). "
        interpretation += "We fail to reject the null hypothesis of a unit root."
    
    # Add information about critical values
    interpretation += f"\nADF Statistic: {adf_stat:.4f}"
    interpretation += f"\nCritical Values: 1%: {critical_values['1%']:.4f}, "
    interpretation += f"5%: {critical_values['5%']:.4f}, 10%: {critical_values['10%']:.4f}"
    
    return interpretation


def _interpret_windowed_adf_results(stationary_ratio: float, total_windows: int, stationary_count: int, threshold: float) -> str:
    """
    Provide a human-readable interpretation of windowed ADF test results.
    
    Args:
        stationary_ratio: Proportion of stationary windows
        total_windows: Total number of windows tested
        stationary_count: Number of stationary windows
        threshold: Threshold for considering series stationary
    
    Returns:
        str: Interpretation of the windowed test results
    """
    interpretation = f"Windowed stationarity analysis: {stationary_count}/{total_windows} windows are stationary "
    interpretation += f"(ratio: {stationary_ratio:.3f}).\n"
    
    if stationary_ratio >= threshold:
        interpretation += f"Since {stationary_ratio:.3f} >= {threshold}, the entire time series is considered STATIONARY. "
        interpretation += "The majority of subsequences exhibit stationary behavior."
    else:
        interpretation += f"Since {stationary_ratio:.3f} < {threshold}, the entire time series is considered NON-STATIONARY. "
        interpretation += "Too many subsequences exhibit non-stationary behavior."
    
    interpretation += f"\nThreshold for stationarity: {threshold} (95% of windows must be stationary)"
    
    return interpretation


def wavelet_decompose(series: pl.Series, wavelet: str = 'db4', levels: int|None = None, seasonal_details: int|None = None) -> dict:
    """
    Decompose a time series into trend, seasonality, and residual components using wavelets.
    
    Args:
        series (pl.Series): The time series to decompose
        wavelet (str): Wavelet type to use (default: 'db4')
        levels (int): Number of decomposition levels (default: auto-calculated)
        seasonal_details (int): Number of detail levels to use for seasonality component.
                               If None, defaults to levels//2. If 0, all details go to residual.
    
    Returns:
        dict: Dictionary containing:
            - 'trend': pl.Series with trend component
            - 'seasonality': pl.Series with seasonal component  
            - 'residual': pl.Series with residual component
            - 'original': pl.Series with original data
    """
    # Remove null values and get clean data
    clean_series = series.drop_nulls()
    data = clean_series.to_numpy().copy()
    
    if len(data) < 32:
        raise ValueError("Series too short for wavelet decomposition (minimum 32 observations required)")
    
    # Auto-calculate levels if not provided
    if levels is None:
        levels = min(int(np.log2(len(data))) - 2, 6)  # Cap at 6 levels
    
    # Auto-calculate seasonal_details if not provided
    if seasonal_details is None:
        seasonal_details = max(1, levels // 2)
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=levels)
    
    # Reconstruct trend (approximation coefficients at lowest frequency)
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(trend_coeffs, wavelet)
    
    # Reconstruct seasonal component (specified number of detail coefficients)
    seasonal_coeffs = [np.zeros_like(coeffs[0])]
    for i in range(1, len(coeffs)):
        if i <= seasonal_details and seasonal_details > 0:
            seasonal_coeffs.append(coeffs[i])
        else:
            seasonal_coeffs.append(np.zeros_like(coeffs[i]))
    
    seasonality = pywt.waverec(seasonal_coeffs, wavelet)
    
    # Reconstruct residual (remaining detail coefficients)
    residual_coeffs = [np.zeros_like(coeffs[0])]
    for i in range(1, len(coeffs)):
        if i > seasonal_details or seasonal_details == 0:
            residual_coeffs.append(coeffs[i])
        else:
            residual_coeffs.append(np.zeros_like(coeffs[i]))
    
    residual = pywt.waverec(residual_coeffs, wavelet)
    
    # Handle length differences due to wavelet reconstruction
    target_length = len(data)
    trend = _adjust_length(trend, target_length)
    seasonality = _adjust_length(seasonality, target_length)
    residual = _adjust_length(residual, target_length)
    
    # Convert back to polars Series
    return {
        'trend': pl.Series(trend),
        'seasonality': pl.Series(seasonality),
        'residual': pl.Series(residual),
        'original': clean_series
    }


def _adjust_length(array: np.ndarray, target_length: int) -> np.ndarray:
    """
    Adjust array length to match target length by truncating or padding.
    
    Args:
        array: Input array
        target_length: Desired length
        
    Returns:
        np.ndarray: Array adjusted to target length
    """
    if len(array) == target_length:
        return array
    elif len(array) > target_length:
        # Truncate from both ends symmetrically
        excess = len(array) - target_length
        start = excess // 2
        return array[start:start + target_length]
    else:
        # Pad with zeros at the end
        padding = target_length - len(array)
        return np.pad(array, (0, padding), mode='constant', constant_values=0)


def plot_periodogram(series: pl.Series, sampling_rate: float = 1.0, title: str = "Periodogram", detrend: str = 'linear') -> None:
    """
    Plot the periodogram of a time series to analyze frequency components.
    
    The periodogram shows the power spectral density, revealing dominant frequencies
    and periodic patterns in the data.
    
    Args:
        series (pl.Series): The time series to analyze
        sampling_rate (float): Sampling rate of the data (default: 1.0)
        title (str): Title for the plot (default: "Periodogram")
        detrend (str): Detrending method ('linear', 'constant', or None) (default: 'linear')
    
    Returns:
        None: Displays the periodogram plot
    """
    # Remove null values and get clean data
    clean_series = series.drop_nulls()
    data = clean_series.to_numpy()
    
    if len(data) < 10:
        raise ValueError("Series too short for periodogram analysis (minimum 10 observations required)")
    
    # Compute periodogram using Welch's method for better frequency resolution
    frequencies, power_spectral_density = signal.periodogram(
        data, 
        fs=sampling_rate, 
        detrend=detrend,
        scaling='density'
    )
    
    # Create the plot
    plt.figure(figsize=(10, 3))
    
    # Plot on log scale for better visualization
    plt.semilogy(frequencies, power_spectral_density, linewidth=1)
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid(True, alpha=0.3)
    
    # Add some annotations for interpretation
    if len(frequencies) > 1:
        # Find dominant frequencies (peaks)
        peak_indices = signal.find_peaks(power_spectral_density, height=float(np.max(power_spectral_density)) * 0.1)[0]
        if len(peak_indices) > 0:
            # Annotate top 3 peaks
            top_peaks = peak_indices[np.argsort(power_spectral_density[peak_indices])[-3:]]
            for idx in top_peaks:
                plt.annotate(f'f={frequencies[idx]:.4f}', 
                           xy=(float(frequencies[idx]), float(power_spectral_density[idx])),
                           xytext=(10, 10), textcoords='offset points',
                           bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'yellow', 'alpha': 0.7},
                           arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3,rad=0'})
    
    plt.tight_layout()
    plt.show()
    
    # Print some basic statistics
    max_power_idx = int(np.argmax(power_spectral_density))
    dominant_freq = float(frequencies[max_power_idx])
    max_power = float(power_spectral_density[max_power_idx])
    
    print(f"Dominant frequency: {dominant_freq:.6f}")
    print(f"Maximum power: {max_power:.2e}")
    if dominant_freq > 0:
        period = 1.0 / dominant_freq
        print(f"Corresponding period: {period:.2f} time units")