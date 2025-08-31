import polars as pl
import numpy as np
from .fact import Fact
import arch
import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

class VolatilityClustering(Fact):
    def __init__(self, series: pl.Series, logger, name = 'volatility clustering', p: int = 1, q: int = 1, dist: str = 'normal', datetime_series: pl.Series = None):
        """
        Initialize the VolatilityClustering fact.
        Args:
            series (pl.Series): The time series data.
            logger: Logger instance for logging.
            name (str): Name of the fact.
            p (int): Order of the GARCH model for the autoregressive term.
            q (int): Order of the GARCH model for the moving average term.
            dist (str): Distribution for GARCH model.
            datetime_series (pl.Series): Optional datetime series for x-axis labeling.
        """
        self.p = p
        self.q = q
        if p < 1 or q < 1:
            raise ValueError("Both p and q must be at least 1 for GARCH model.")
        if not isinstance(series, pl.Series):
            raise TypeError("series must be a polars Series.")
        self.dist = dist
        self.datetime_series = datetime_series
        super().__init__(name, series, logger, 'Time', 0)

    def compute(self):
        self.logger.info(f"Computing volatility clustering")
        garch_model = arch_model(self.series.drop_nulls().to_numpy(), vol='Garch', p=self.p, q=self.q, dist=self.dist, mean='Zero')
        garch_fitted = garch_model.fit(disp='off')

        self.logger.info(f"GARCH({self.p},{self.q}) Model Results:")
        self.logger.info(garch_fitted.summary())
        return garch_fitted

    def plot(self):
        """
        Compute and plot the stylized fact.
        """
        self.logger.info(f"Plotting {self.name}")
        garch_fitted = self.compute()

        conditional_volatility = garch_fitted.conditional_volatility
        residuals = garch_fitted.resid

        # Create plots to demonstrate volatility clustering
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Prepare x-axis data
        if self.datetime_series is not None:
            # Filter datetime series to match the length of conditional_volatility
            # (GARCH drops nulls so lengths might differ)
            datetime_values = self.datetime_series.drop_nulls()
            if len(datetime_values) > len(conditional_volatility):
                datetime_values = datetime_values[-len(conditional_volatility):]
            elif len(datetime_values) < len(conditional_volatility):
                # If datetime is shorter, use the available datetime and fill with sequential values
                x_values = list(range(len(conditional_volatility)))
            else:
                x_values = datetime_values.to_list()
        else:
            x_values = list(range(len(conditional_volatility)))

        # Plot 1: Conditional volatility from GARCH
        axes[0].plot(x_values, conditional_volatility, color='red', linewidth=0.8)
        axes[0].set_title('GARCH(1,1) Conditional Volatility - Evidence of Clustering', fontsize=12)
        axes[0].set_ylabel('Volatility (%)')
        axes[0].grid(True, alpha=0.3)
        
        # Format x-axis for datetime if available
        if self.datetime_series is not None and isinstance(x_values[0] if x_values else None, (pd.Timestamp, np.datetime64)):
            axes[0].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Squared returns vs conditional variance
        squared_returns = self.series.drop_nulls()**2
        # Align lengths for plotting
        if len(squared_returns) > len(conditional_volatility):
            squared_returns = squared_returns[-len(conditional_volatility):]
        
        axes[1].plot(x_values, squared_returns, linewidth=0.5, alpha=0.6, label='Squared Returns')
        axes[1].plot(x_values, conditional_volatility**2, color='red', linewidth=0.8, label='GARCH Conditional Variance')
        axes[1].set_title('Squared Returns vs GARCH Conditional Variance', fontsize=12)
        axes[1].set_ylabel('Variance')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Format x-axis for datetime if available
        if self.datetime_series is not None and isinstance(x_values[0] if x_values else None, (pd.Timestamp, np.datetime64)):
            axes[1].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

        # Print key GARCH parameters to demonstrate volatility clustering
        self.logger.info("\nKey GARCH Parameters (Evidence of Volatility Clustering):")
        self.logger.info(f"α (alpha): {garch_fitted.params['alpha[1]']:.6f} - Impact of previous squared return")
        self.logger.info(f"β (beta):  {garch_fitted.params['beta[1]']:.6f} - Persistence of volatility")
        self.logger.info(f"Persistence (α + β): {garch_fitted.params['alpha[1]'] + garch_fitted.params['beta[1]']:.6f}")

        if garch_fitted.params['alpha[1]'] + garch_fitted.params['beta[1]'] > 0.95:
            self.logger.info("High persistence indicates strong volatility clustering!")