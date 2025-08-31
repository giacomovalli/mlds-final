import polars as pl
import numpy as np
from .fact import Fact
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import scipy.stats as stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class Autocorrelation(Fact):
    def __init__(self, series: pl.Series, logger, window: int = 0, name = 'autocorrelation', K = 50):
        super().__init__(name, series, logger, 'Lag', window)
        self.K = K

        if K > window and window != 0:
            self.logger.error(f"Requested lag {K} exceeds window size {self.window}")
            raise ValueError(f"Requested lag {K} exceeds window size {self.window}")

    def compute(self) -> np.ndarray:
        """
        Compute autocorrelation for the given series.
        If window > 0, compute for sliding windows of that size.

        Returns:
            np.ndarray: Autocorrelation values (2D if windowed, 1D if not)
        """
        self.logger.info(f"Computing autocorrelation: window={self.window}, K={self.K}")
        #self.logger.info(f"Using {mp.cpu_count()} parallel processes")
        
        autocorr_matrix = []
        if self.window == 0:
            gamma_0 = self.series.var()
            if gamma_0 is None or gamma_0 == 0:
                self.logger.warning("Series has zero variance, returning zeros")
                return np.zeros(self.K)

            for k in range(1, self.K + 1):
                gamma_k = pl.cov(self.series, self.series.shift(k).rename("lagged"), eager=True).item()
                if gamma_k is None:
                    autocorr_matrix.append(0)
                else:
                    autocorr_k = gamma_k / gamma_0
                    autocorr_matrix.append(autocorr_k)
        else:
            for k in range(1, self.K + 1):
                df = pl.DataFrame([self.series.rename("orig"), self.series.shift(k).rename("lagged")])
                rolling_corr_df = df.with_columns(
                    rolling_corr=pl.rolling_corr("orig", "lagged", window_size=self.window)
)
                autocorr_matrix.append(rolling_corr_df["rolling_corr"].to_numpy()[self.window:].tolist())
                self.logger.info(f"Computed rolling autocorrelation for lag {k}") 

        self.logger.info(f"Shape: {np.array(autocorr_matrix).shape}") 
        return np.array(autocorr_matrix)

    def plot(self):
        """
        Compute and plot the stylized fact.
        """
        self.logger.info(f"Plotting {self.name}")
        result = self.compute()
        self.data = result

        # Create figure with subplots
        if result.ndim > 1:
            plt.figure(figsize=(18, 5))
            
            # First plot: boxplot
            plt.subplot(1, 3, 1)
            plt.boxplot([result[i, :][~np.isnan(result[i, :])] for i in range(result.shape[0])], 
                       label=[f"k={i+1}" for i in range(result.shape[0])])
            plt.title(self.name)
            plt.xlabel(self.x_label)
            plt.ylabel("Value")
            plt.grid(True)
            
            # Second plot: histogram of first row
            plt.subplot(1, 3, 2)
            first_row = result[0, :][~np.isnan(result[0, :])]
            plt.hist(first_row.tolist(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f"Distribution of k=1 Autocorrelation")
            plt.xlabel("Autocorrelation Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            
            # Third plot: QQ plot to check normality
            plt.subplot(1, 3, 3)
            stats.probplot(first_row, dist="norm", plot=plt)
            plt.title("Q-Q Plot vs Normal Distribution")
            plt.grid(True)
        else:
            # For 1D results, create a subplot with ACF, PACF
            fig, axes = plt.subplots(1, 1, figsize=(5, 3))
            
            # Diminishing autocorrelation plot
            x_values = range(1, len(result) + 1)
            axes.plot(x_values, result.tolist(), linestyle=':', marker='x')
            axes.set_title(self.name)
            axes.set_xlabel(self.x_label)
            axes.set_ylabel("Value")
            axes.grid(True)
            
            # ACF plot using statsmodels - downsample by factor of 15
            # Downsample by factor of 15: take mean of every 15 consecutive points
            #n_groups = len(self.series) // 15
            #downsampled_data = np.mean(self.series[:n_groups*15].to_numpy().reshape(-1, 15), axis=1)
            
            #fig_acf, ax_acf = plt.subplots(1, 1, figsize=(8, 4))
            #from statsmodels.graphics.tsaplots import plot_acf
            #plot_acf(downsampled_data, ax=ax_acf, lags=30, title='ACF of Downsampled Series (factor=15)')
            #ax_acf.set_xlabel('Lag (15-minute intervals)')
            #plt.tight_layout()
            #plt.show()
            
            # PACF plot using statsmodels
            #plot_pacf(self.series[4000000:5000000].to_numpy(), ax=axes[1, 0], lags=30, title='Partial Autocorrelation Function (PACF)')
            
            # Time series plot
            #axes[1, 1].plot(self.series.to_numpy(), linewidth=0.8)
            #axes[1, 1].set_title('Original Time Series')
            #axes[1, 1].set_xlabel('Time')
            #axes[1, 1].set_ylabel('Value')
            #axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
