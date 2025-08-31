import polars as pl
import numpy as np
from .fact import Fact
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd


class VolumeVolatilityCorrelation(Fact):
    def __init__(self, open_series: pl.Series, high_series: pl.Series, low_series: pl.Series, 
                 close_series: pl.Series, volume_series: pl.Series, logger, 
                 name='Volume-Volatility Correlation', datetime_series: pl.Series = None,
                 correlation_windows: list = None):
        """
        Initialize the VolumeVolatilityCorrelation fact with OHLC data for Garman-Klass volatility.
        
        Args:
            open_series (pl.Series): The opening price time series.
            high_series (pl.Series): The high price time series.
            low_series (pl.Series): The low price time series.
            close_series (pl.Series): The closing price time series.
            volume_series (pl.Series): The volume time series data.
            logger: Logger instance for logging.
            name (str): Name of the fact.
            datetime_series (pl.Series): Optional datetime series for x-axis labeling.
            correlation_windows (list): List of window sizes for rolling correlation. 
                                      Default: [50, 100, 250, 500]
        """
        # Validate all OHLC series
        ohlc_series = [open_series, high_series, low_series, close_series, volume_series]
        for i, series in enumerate(ohlc_series):
            if not isinstance(series, pl.Series):
                series_names = ['open_series', 'high_series', 'low_series', 'close_series', 'volume_series']
                raise TypeError(f"{series_names[i]} must be a polars Series.")
        
        # Check all series have same length
        lengths = [len(series) for series in ohlc_series]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All OHLC and volume series must have the same length.")
            
        self.open_series = open_series
        self.high_series = high_series
        self.low_series = low_series
        self.close_series = close_series
        self.volume_series = volume_series
        self.datetime_series = datetime_series
        self.correlation_windows = correlation_windows or [50, 100, 250, 500]
        
        # Initialize with close_series as the main series for compatibility
        super().__init__(name, close_series, logger, 'Time', 0)

    def compute_garman_klass_volatility(self):
        """
        Compute Garman-Klass volatility estimator using OHLC data.
        The Garman-Klass estimator provides a volatility estimate for each period.
        
        Formula: GK = 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
        where H=High, L=Low, C=Close, O=Open
        
        Returns:
            np.ndarray: Garman-Klass volatility series (one value per period)
        """
        self.logger.info("Computing Garman-Klass volatility estimator")
        
        # Create DataFrame with OHLC data for efficient computation
        df_ohlc = pl.DataFrame({
            'open': self.open_series,
            'high': self.high_series, 
            'low': self.low_series,
            'close': self.close_series
        })
        
        # Compute Garman-Klass estimator for each period
        # GK = 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
        gk_constant = 2 * np.log(2) - 1  # ≈ 0.3863
        
        gk_volatility = df_ohlc.with_columns([
            # First term: 0.5 * (ln(H/L))^2
            (0.5 * (pl.col('high') / pl.col('low')).log().pow(2)).alias('hl_term'),
            
            # Second term: (2*ln(2)-1) * (ln(C/O))^2  
            (gk_constant * (pl.col('close') / pl.col('open')).log().pow(2)).alias('co_term')
        ]).with_columns([
            # Combine terms: GK = hl_term - co_term
            (pl.col('hl_term') - pl.col('co_term')).alias('gk_volatility')
        ])['gk_volatility'].to_numpy()
        
        return gk_volatility

    def compute_rolling_correlations(self, volatility_series: np.ndarray, volume_array: np.ndarray):
        """
        Compute rolling correlations between volatility and volume for different window sizes
        using Polars' efficient rolling operations.
        
        Args:
            volatility_series (np.ndarray): Rolling volatility series
            volume_array (np.ndarray): Volume series as numpy array
            
        Returns:
            dict: Dictionary with window sizes as keys and correlation arrays as values
        """
        self.logger.info(f"Computing rolling correlations using Polars for windows: {self.correlation_windows}")
        
        # Create Polars DataFrame with volatility and volume columns
        df_corr = pl.DataFrame({
            'volatility': volatility_series,
            'volume': volume_array
        })
        
        correlations = {}
        
        for window in self.correlation_windows:
            if window > len(volatility_series):
                self.logger.warning(f"Window size {window} is larger than data length {len(volatility_series)}, skipping.")
                continue
            
            # Use Polars' rolling operations to compute correlation manually
            # Correlation = Cov(X,Y) / (Std(X) * Std(Y))
            # We'll use rolling statistics to compute this efficiently
            rolling_corr = df_corr.with_columns([
                # Rolling means
                pl.col('volatility').rolling_mean(window_size=window, min_periods=window).alias('vol_mean'),
                pl.col('volume').rolling_mean(window_size=window, min_periods=window).alias('vol_volume_mean'),
                
                # Rolling standard deviations
                pl.col('volatility').rolling_std(window_size=window, min_periods=window).alias('vol_std'),
                pl.col('volume').rolling_std(window_size=window, min_periods=window).alias('volume_std'),
            ]).with_columns([
                # Rolling covariance using the formula: E[XY] - E[X]E[Y]
                ((pl.col('volatility') * pl.col('volume')).rolling_mean(window_size=window, min_periods=window) 
                 - pl.col('vol_mean') * pl.col('vol_volume_mean')).alias('covariance')
            ]).with_columns([
                # Final correlation: Cov(X,Y) / (Std(X) * Std(Y))
                (pl.col('covariance') / (pl.col('vol_std') * pl.col('volume_std'))).alias('rolling_correlation')
            ])['rolling_correlation'].to_numpy()
            
            correlations[window] = rolling_corr
            
        return correlations

    def compute(self):
        """
        Compute the volume-volatility correlation analysis using Garman-Klass volatility.
        
        Returns:
            dict: Dictionary containing volatility and correlation results
        """
        self.logger.info("Computing volume-volatility correlation analysis with Garman-Klass volatility")
        
        # Compute Garman-Klass volatility
        gk_volatility = self.compute_garman_klass_volatility()
        
        # Get volume as numpy array (aligned with price series)
        volume_array = self.volume_series.drop_nulls().to_numpy()
        
        # Ensure lengths match (they should after drop_nulls)
        min_len = min(len(gk_volatility), len(volume_array))
        gk_volatility = gk_volatility[:min_len]
        volume_array = volume_array[:min_len]
        
        # Compute rolling correlations
        rolling_correlations = self.compute_rolling_correlations(gk_volatility, volume_array)
        
        return {
            'garman_klass_volatility': gk_volatility,
            'volume': volume_array,
            'rolling_correlations': rolling_correlations
        }

    def plot(self):
        """
        Plot the volume-volatility correlation analysis.
        """
        self.logger.info(f"Plotting {self.name}")
        
        # Compute the analysis
        results = self.compute()
        gk_volatility = results['garman_klass_volatility']
        volume_array = results['volume']
        rolling_correlations = results['rolling_correlations']
        
        # Prepare x-axis data
        data_length = len(gk_volatility)
        if self.datetime_series is not None:
            datetime_values = self.datetime_series.drop_nulls()
            if len(datetime_values) >= data_length:
                x_values = datetime_values[:data_length].to_list()
                use_datetime = True
            else:
                x_values = list(range(data_length))
                use_datetime = False
        else:
            x_values = list(range(data_length))
            use_datetime = False

        # Create subplots with 2-column layout
        n_correlations = len(rolling_correlations)
        total_plots = n_correlations + 1  # volatility + correlations (no volume)
        n_rows = (total_plots + 1) // 2  # Round up to get number of rows needed
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
        
        # Flatten axes array for easy indexing
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        # Plot 1: Garman-Klass Volatility
        axes_flat[0].plot(x_values, gk_volatility, color='blue', linewidth=0.8, alpha=0.7)
        axes_flat[0].set_title('Garman-Klass Volatility (per period)', fontsize=12)
        axes_flat[0].set_ylabel('GK Volatility')
        axes_flat[0].grid(True, alpha=0.3)
        
        if use_datetime:
            self._format_datetime_axis(axes_flat[0])

        # Plot correlations for each window (with moving average smoothing) 
        smoothing_window = 10000
        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (window, correlation) in enumerate(rolling_correlations.items()):
            ax_idx = i + 1  # Start from index 1 since we removed volume plot
            if ax_idx < len(axes_flat):
                color = colors[i % len(colors)]
                
                # Apply moving average to correlation
                correlation_ma = self._compute_moving_average(correlation, smoothing_window)
                
                axes_flat[ax_idx].plot(x_values, correlation_ma, color=color, linewidth=1.0, alpha=0.8)
                axes_flat[ax_idx].set_title(f'Rolling Correlation (Window = {window}, MA-{smoothing_window})', fontsize=12)
                axes_flat[ax_idx].set_ylabel('Correlation (MA)')
                axes_flat[ax_idx].grid(True, alpha=0.3)
                axes_flat[ax_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Add horizontal lines for significant correlation levels
                axes_flat[ax_idx].axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='±0.3')
                axes_flat[ax_idx].axhline(y=-0.3, color='gray', linestyle=':', alpha=0.5)
                axes_flat[ax_idx].axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='±0.5')
                axes_flat[ax_idx].axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)
                
                if use_datetime:
                    self._format_datetime_axis(axes_flat[ax_idx])
                
                # Calculate and display statistics (using smoothed data)
                valid_corr = correlation_ma[~np.isnan(correlation_ma)]
                if len(valid_corr) > 0:
                    mean_corr = np.mean(valid_corr)
                    std_corr = np.std(valid_corr)
                    axes_flat[ax_idx].text(0.02, 0.95, f'Mean: {mean_corr:.3f}\nStd: {std_corr:.3f}', 
                                    transform=axes_flat[ax_idx].transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots
        for i in range(total_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Set x-label for bottom row subplots
        bottom_row_start = (n_rows - 1) * 2
        for i in range(bottom_row_start, min(bottom_row_start + 2, total_plots)):
            if i < len(axes_flat):
                axes_flat[i].set_xlabel('Time')
        
        plt.tight_layout()
        plt.show()
        
        # Log summary statistics
        self.logger.info("\nVolume-Volatility Correlation Summary:")
        for window, correlation in rolling_correlations.items():
            valid_corr = correlation[~np.isnan(correlation)]
            if len(valid_corr) > 0:
                self.logger.info(f"Window {window}: Mean correlation = {np.mean(valid_corr):.4f}, "
                               f"Std = {np.std(valid_corr):.4f}, "
                               f"Max = {np.max(valid_corr):.4f}, "
                               f"Min = {np.min(valid_corr):.4f}")

    def _compute_moving_average(self, data, window):
        """
        Compute moving average for smoothing noisy data.
        
        Args:
            data (np.ndarray): Input data array
            window (int): Window size for moving average
            
        Returns:
            np.ndarray: Smoothed data using moving average
        """
        # Convert to Polars Series for efficient rolling mean computation
        if isinstance(data, np.ndarray):
            data_series = pl.Series(data)
        else:
            data_series = data
            
        # Compute rolling mean
        smoothed = data_series.rolling_mean(window_size=window, min_periods=1)
        
        return smoothed.to_numpy()
    
    def _format_datetime_axis(self, ax):
        """Helper method to format datetime axis with labels every two years"""
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Only show year
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # One label every two years
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Minor ticks every year
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)  # No rotation needed for years