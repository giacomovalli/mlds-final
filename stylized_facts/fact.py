import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Fact(ABC):
    def __init__(self, name: str, series: pl.Series, logger, x_label: str = "", window: int = 0):
        self.name = name
        self.series = series
        self.logger = logger
        self.x_label = x_label
        self.window = window
        self.data = np.array([])

    @abstractmethod
    def compute(self) -> np.ndarray:
        """
        Abstract method to compute the stylized fact.

        Returns:
            pl.Series: Result of the computation
        """
        pass

    def plot(self):
        """
        Compute and plot the stylized fact.
        """
        self.logger.info(f"Plotting {self.name}")
        result = self.compute()
        self.data = result

        plt.figure(figsize=(10, 6))
        if result.ndim == 1:
            plt.plot(result.tolist())
        else:
            # For 2D arrays, plot each row
            for i in range(result.shape[0]):
                arr = result[i, :]
                arr = arr[~np.isnan(arr)]
                plt.plot(arr.tolist(), alpha=0.7, label=f"k={i+1}")
            if result.shape[0] <= 10:
                plt.legend()
        plt.title(self.name)
        plt.xlabel(self.x_label)
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
