import polars as pl
import logging
import numpy as np
import random
import keras as keras
from decompose.preprocess import wavelet_decompose
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.scalers = {}  # Dictionary to store MinMaxScaler instances

    def load_financial_data(self, file_path: str) -> pl.DataFrame:
        """
        Load financial data from a CSV file into a Polars DataFrame.

        Args:
            file_path (str): Path to the CSV file containing financial data

        Returns:
            pl.DataFrame: DataFrame with columns: datetime, open, high, low, close, volume
        """
        self.logger.info(f"Loading financial data from {file_path}")
        df = pl.read_csv(
            file_path,
            has_header=False,
            new_columns=["datetime", "open", "high", "low", "close", "volume"]
        ).with_columns(
            pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )
        self.logger.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
        df_processed = self._preprocess(df)
        return df_processed

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocess the financial data by adding columns for price and volume changes.

        Args:
            df (pl.DataFrame): Input DataFrame with financial data

        Returns:
            pl.DataFrame: DataFrame with added change columns for open, high, low, close, and volume
        """
        # Drop rows with null values
        self.logger.info("Removing rows with null values")
        df_clean = df.drop_nulls()

        self.logger.info(f"After removing nulls: {df_clean.shape[0]} rows remaining")

        # Add detrended price series using wavelet decomposition
        self.logger.info("Computing trends and adding detrended price series")
        
        price_columns = ['open', 'high', 'low', 'close']
        detrended_columns = []
        trend_columns = []
        
        for col in price_columns:
            self.logger.info(f"Computing trend for {col}")
            # Perform wavelet decomposition to extract trend
            decomposition = wavelet_decompose(df_clean[col], levels=18, seasonal_details=0)
            trend = decomposition['trend']
            
            # Store trend component
            trend_columns.append(trend.alias(f'trend_{col}'))
            
            # Compute detrended series (original - trend)
            detrended = df_clean[col] - trend
            detrended_columns.append(detrended.alias(f'detrended_{col}'))
        
        # Add all trend and detrended columns to the dataframe
        df_clean = df_clean.with_columns(trend_columns + detrended_columns)

        self.logger.info("Preprocessing data: adding price and volume changes")
        df_clean = df_clean.with_columns([
            (pl.col("detrended_open") - pl.col("detrended_open").shift(1)).alias("c1_detrended_open"),
            (pl.col("detrended_high") - pl.col("detrended_high").shift(1)).alias("c1_detrended_high"),
            (pl.col("detrended_low") - pl.col("detrended_low").shift(1)).alias("c1_detrended_low"),
            (pl.col("detrended_close") - pl.col("detrended_close").shift(1)).alias("c1_detrended_close"),
            (pl.col("volume") - pl.col("volume").shift(1)).alias("c1_volume")
        ])

        if True:
            # Apply min-max scaling to c1_detrended columns using sklearn
            self.logger.info("Applying min-max scaling to c1_detrended columns")
            c1_detrended_columns = ['c1_detrended_open', 'c1_detrended_high', 'c1_detrended_low', 'c1_detrended_close']
            
            for col in c1_detrended_columns:
                # Extract column data as numpy array
                col_data = df_clean[col].to_numpy().reshape(-1, 1)
                col_min_orig = df_clean[col].min()
                col_max_orig = df_clean[col].max()
                
                # Create and fit MinMaxScaler for [0, 1] range
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(col_data).flatten()
                
                # Store the scaler for later inverse transformation
                self.scalers[col] = scaler
                
                # Replace column with scaled data
                df_clean = df_clean.with_columns([
                    pl.Series(name=col, values=scaled_data)
                ])
                self.logger.info(f"Scaled {col}: min={col_min_orig:.6f}, max={col_max_orig:.6f} -> [0, 1]")

        self.logger.info("Preprocessing completed")
        return df_clean

    def inverse_transform_series(self, c1_detrended_open, c1_detrended_high, c1_detrended_low, c1_detrended_close):
        """
        Transform the four scaled series back to their original scale.
        
        Args:
            c1_detrended_open (np.array): Scaled c1_detrended_open series (0-1 range)
            c1_detrended_high (np.array): Scaled c1_detrended_high series (0-1 range)
            c1_detrended_low (np.array): Scaled c1_detrended_low series (0-1 range)
            c1_detrended_close (np.array): Scaled c1_detrended_close series (0-1 range)
            
        Returns:
            tuple: Four arrays with data in original scale
        """
        if not self.scalers:
            raise ValueError("No scalers found. Make sure to call load_financial_data() first.")
        
        # Ensure inputs are numpy arrays and reshape for scaler
        c1_detrended_open = np.array(c1_detrended_open).reshape(-1, 1)
        c1_detrended_high = np.array(c1_detrended_high).reshape(-1, 1)
        c1_detrended_low = np.array(c1_detrended_low).reshape(-1, 1)
        c1_detrended_close = np.array(c1_detrended_close).reshape(-1, 1)
        
        # Inverse transform using the stored scalers
        original_open = self.scalers['c1_detrended_open'].inverse_transform(c1_detrended_open).flatten()
        original_high = self.scalers['c1_detrended_high'].inverse_transform(c1_detrended_high).flatten()
        original_low = self.scalers['c1_detrended_low'].inverse_transform(c1_detrended_low).flatten()
        original_close = self.scalers['c1_detrended_close'].inverse_transform(c1_detrended_close).flatten()
        
        self.logger.info("Successfully inverse transformed series to original scale")
        
        return original_open, original_high, original_low, original_close


class BatchedTimeseriesSequence(keras.utils.Sequence):
    def __init__(self, dataframe: pl.DataFrame, window: int, batch_size: int = 32, logger = None, shuffle=True):
        """
        Keras Sequence that takes a Polars DataFrame as input.
        
        Args:
            dataframe (pl.DataFrame): Input Polars DataFrame
            window (int): Window size for sequence data
            batch_size (int): Batch size for training
        """
        self.dataframe = dataframe
        self.window = window
        self.batch_size = batch_size
        
        # Extract all four detrended price series
        self.data_arrays = {
            'close': self.dataframe.select("c1_detrended_close").to_numpy().flatten(),
            'open': self.dataframe.select("c1_detrended_open").to_numpy().flatten(),
            'high': self.dataframe.select("c1_detrended_high").to_numpy().flatten(),
            'low': self.dataframe.select("c1_detrended_low").to_numpy().flatten()
        }
        
        self.n_samples = len(self.data_arrays['close']) - window + 1
        self.logger = logger
        self.shuffle = shuffle
        self.num_batches_all = int(np.ceil(self.n_samples / self.batch_size))
        self.indices = list(range(self.num_batches_all))
        
    def __len__(self):
        return self.num_batches_all
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        start_idx = real_idx * self.batch_size
        end_idx = min((real_idx + 1) * self.batch_size, self.n_samples)
        
        batch_size_actual = end_idx - start_idx
        # Shape: (batch_size, 4, window)
        batch_x = np.zeros((batch_size_actual, 4, self.window))
        
        for i, sample_idx in enumerate(range(start_idx, end_idx)):
            # Fill each channel with the corresponding detrended price series
            batch_x[i, 0] = self.data_arrays['close'][sample_idx:sample_idx + self.window]  # c1_detrended_close
            batch_x[i, 1] = self.data_arrays['open'][sample_idx:sample_idx + self.window]   # c1_detrended_open
            batch_x[i, 2] = self.data_arrays['high'][sample_idx:sample_idx + self.window]   # c1_detrended_high
            batch_x[i, 3] = self.data_arrays['low'][sample_idx:sample_idx + self.window]    # c1_detrended_low
        
        batch_y = np.zeros(batch_size_actual)
        
        # Transpose to match STAAR model expected shape: (batch_size, window, features)
        batch_x = np.transpose(batch_x, (0, 2, 1))
        
        return batch_x, batch_x
    
    def to_tf_dataset(self):
        """Convert this Sequence to a tf.data.Dataset with proper size information"""
        
        def generator():
            # This generator will be called once per epoch
            self.on_epoch_end()
            for i in range(len(self)):
                yield self[i]
        
        # Get output signature by getting a sample
        sample_x, sample_y = self[0]
        output_signature = (
            tf.TensorSpec(shape=(None, self.window, 4), dtype=tf.float32),  # (batch_size, window, features)
            tf.TensorSpec(shape=(None, self.window, 4), dtype=tf.float32),
            #tf.TensorSpec(shape=(None,), dtype=tf.float32)  # (batch_size,)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        # Set cardinality to help with progress tracking
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(self)))
        
        return dataset

    def on_epoch_end(self):
        """
        Reshuffle indices at the end of each epoch
        """
        #self.logger.info("Shuffling indices for the next epoch") if self.logger else None
        self.indices = list(range(self.num_batches_all))
        random.shuffle(self.indices)


class BatchedTimeseriesDataset:
    def __init__(self, dataframe: pl.DataFrame, window_size: int, logger=None):
        """
        TensorFlow Dataset factory for time series data from Polars DataFrame.
        
        Args:
            dataframe (pl.DataFrame): Input Polars DataFrame
            window_size (int): Window size for sequence data
            batch_size (int): Batch size for training
            logger: Logger instance
            shuffle (bool): Whether to shuffle the data
        """
        self.dataframe = dataframe
        self.window_size = window_size
        self.logger = logger
        
        # Extract all four detrended price series
        self.data_arrays = {
            'close': self.dataframe.select("c1_detrended_close").to_numpy().flatten(),
            'open': self.dataframe.select("c1_detrended_open").to_numpy().flatten(),
            'high': self.dataframe.select("c1_detrended_high").to_numpy().flatten(),
            'low': self.dataframe.select("c1_detrended_low").to_numpy().flatten()
        }
        
        self.n_samples = len(self.data_arrays['close']) - window_size + 1
        
        if logger:
            logger.info(f"BatchedTimeseriesDataset initialized with {self.n_samples} samples")
    
    def _generator(self):
        """Generator function that yields individual sequences"""
        indices = list(range(self.n_samples))
            
        for idx in indices:
            # Create single sequence (window_size, 4)
            sequence = np.zeros((self.window_size, 4), dtype=np.float32)
            sequence[:, 0] = self.data_arrays['close'][idx:idx + self.window_size]  # c1_detrended_close
            sequence[:, 1] = self.data_arrays['open'][idx:idx + self.window_size]   # c1_detrended_open
            sequence[:, 2] = self.data_arrays['high'][idx:idx + self.window_size]   # c1_detrended_high
            sequence[:, 3] = self.data_arrays['low'][idx:idx + self.window_size]    # c1_detrended_low
            
            yield sequence, sequence  # (input, target) both same for autoencoder
    
    def create_dataset(self):
        """Create and return a tf.data.Dataset"""
        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(self.window_size, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(self.window_size, 4), dtype=tf.float32)
        )
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=output_signature
        )
        
        # Note: Removed cardinality assertion to prevent multi-GPU sync issues
        
        return dataset
