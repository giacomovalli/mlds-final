import polars as pl
import numpy as np
from .fact import Fact
import matplotlib.pyplot as plt
import seaborn as sns

class GainLossAsymmetry(Fact):
    def __init__(self, series: pl.Series, logger, name = 'Gain-Loss Asymmetry', threshold: float = 0.01):
        super().__init__(name, series, logger, 'Time', 0)
        self.threshold = threshold

        if self.threshold <= 0:
            raise ValueError("Threshold must be a positive number.")
    
    def _calculate_t_wait(self, prices: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the waiting times for gain/loss asymmetry in a financial time-series.

        This function implements the T_wait metric as defined in the paper "Modeling 
        financial time-series with generative adversarial networks". It measures the 
        number of time steps required for the cumulative log return to cross a 
        positive threshold (gain) or a negative threshold (loss) from each point 
        in the series.

        Args:
            prices (np.ndarray): A 1D NumPy array of asset prices.
            threshold (float): The positive threshold value (theta) to be tested. 
                            The function will test for both +threshold and -threshold.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                        - The first array contains the waiting times for gains.
                                        - The second array contains the waiting times for losses.
        """
        log_prices = np.log(prices)
        n = len(log_prices)
        t_wait_positive = []
        t_wait_negative = []

        # Iterate through each starting point in the time-series
        for t in range(n - 1):
                
            returns = log_prices[t + 1:] - log_prices[t]

            positive_crossings = np.where(returns >= threshold)[0]
            if positive_crossings.size > 0:
                t_prime_positive = positive_crossings[0] + 1
                t_wait_positive.append(t_prime_positive)

            # Find the first time t' where the return goes below the negative threshold
            negative_crossings = np.where(returns <= -threshold)[0]
            if negative_crossings.size > 0:
                t_prime_negative = negative_crossings[0] + 1
                t_wait_negative.append(t_prime_negative)
                
        return np.array(t_wait_positive), np.array(t_wait_negative)

    def compute(self):
        return self._calculate_t_wait(self.series[-100000:].to_numpy(), self.threshold)

    def plot(self):
        """
        Compute and plot the stylized fact.
        """
        self.logger.info(f"Plotting {self.name}")
        t_wait_positive, t_wait_negative = self.compute()

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(6, 3))

        # Plot using Kernel Density Estimation for a smooth distribution
        sns.kdeplot(t_wait_positive, ax=ax, color='red', label=f'Gain ($\\theta=+{self.threshold}$)', fill=True, alpha=0.1)
        sns.kdeplot(t_wait_negative, ax=ax, color='blue', label=f'Loss ($\\theta=-{self.threshold}$)', fill=True, alpha=0.1)

        ax.set_title('Gain/Loss Asymmetry: Distribution of Waiting Times')
        ax.set_xlabel("Return Time $t'$")
        ax.set_ylabel('Probability Density')
        ax.set_xscale('log')
        ax.set_xlim(left=1)
        ax.legend()
        plt.tight_layout()
        plt.show()
