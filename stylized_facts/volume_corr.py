import polars as pl
import numpy as np
from .fact import Fact
import matplotlib.pyplot as plt

class VolumeCorrelation(Fact):
    def __init__(self, series: pl.Series, logger, frame:pl.DataFrame, name = 'volatility clustering', window: int = 120, ma=15000):
        super().__init__(name, series, logger, 'Time', window)
        self.ma = ma
        self.frame = frame

    def compute_rolling_correlation(self, col1, col2, window_size=1000, ma=0):
        """
        Compute rolling correlation between two columns using Polars rolling_corr
        
        Args:
            df: Polars DataFrame
            col1: First column name
            col2: Second column name  
            window_size: Window size for rolling correlation
        
        Returns:
            Polars Series with rolling correlation values
        """
        rolling_corr = self.frame.select(
            pl.rolling_corr(pl.col(col1).abs(), pl.col(col2), window_size=window_size).alias("rolling_correlation")
        )["rolling_correlation"]
        if ma > 0:
            rolling_corr = rolling_corr.rolling_mean(window_size=ma)
        return rolling_corr

    def compute(self):
        """        Compute the  correlation between volume and absolute returns.
        Returns:
            np.ndarray: The volume correlation of the series.
        """
        self.logger.info(f"Computing rolling correlation with window size {self.window:,}...")
        rolling_corr = self.compute_rolling_correlation("c1_detrended_close", "volume", self.window, ma=self.ma)

        return rolling_corr.drop_nulls().to_numpy()

    def plot(self):
        valid_corr = self.compute()
        fig, axes = plt.subplots(1, 1, figsize=(8, 2))
        axes.plot(valid_corr.tolist(), linewidth=0.8, label=f'window={self.window:,}')
        axes.set_xlabel('Time')
        axes.set_ylabel('Correlation')
        axes.grid(True, alpha=0.3)
        axes.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        self.logger.info(f"  - Valid correlation points: {len(valid_corr):,}")
        self.logger.info(f"  - Min/Max: {valid_corr.min():.6f} / {valid_corr.max():.6f}")