import tensorflow as tf
import keras as keras
from keras import layers, Model
import os
from keras.models import load_model

class StaarModel:
    """
    Builds and contains all neural network components for the STAAR model.
    - Encoder
    - Decoder
    - Latent Discriminator
    - Statistical Discriminator
    """
    def __init__(self, logger, time_steps=120, features=4, latent_dim=32,
                 lstm_units=32, num_heads=4, key_dim=32, num_blocks=2,
                 conv_filters_1=64, conv_filters_2=128, kernel_size=5, dense_units=256):
        self.time_steps = time_steps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.num_blocks = num_blocks
        self.conv_filters_1 = conv_filters_1
        self.conv_filters_2 = conv_filters_2
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.logger = logger

        # Build all the component models upon initialization
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.latent_discriminator = self._build_latent_discriminator()
        self.stats_discriminator = self._build_stats_discriminator()

    def _build_encoder(self):
        inputs = layers.Input(shape=(self.time_steps, self.features))
        x = inputs

        # First 1D Conv layer with batch normalization
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=self.conv_filters_1, kernel_size=self.kernel_size, 
                         padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        # Second 1D Conv layer with batch normalization
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=self.conv_filters_2, kernel_size=self.kernel_size, 
                         padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        # Flatten and projection to latent space
        x = layers.Flatten()(x)
        x = layers.Dense(self.dense_units, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer for mean and Cholesky decomposition elements
        num_chol = int(self.latent_dim * (self.latent_dim + 1) / 2)
        outputs = layers.Dense(self.latent_dim + num_chol, name='encoder_outputs')(x)
        
        return Model(inputs, outputs, name='encoder')

    def _build_decoder(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        
        # Dense layers to expand from latent space
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(self.dense_units, activation='relu')(x)
        
        # Calculate the shape after pooling operations in encoder
        # time_steps -> pool/2 -> pool/2 = time_steps/4
        pooled_time_steps = self.time_steps // 4
        x = layers.Dense(pooled_time_steps * self.conv_filters_2, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Reshape((pooled_time_steps, self.conv_filters_2))(x)

        # First transpose conv (upsample)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=self.conv_filters_1, kernel_size=self.kernel_size, 
                         padding='same', activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Second transpose conv (upsample) 
        x = layers.UpSampling1D(size=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=32, kernel_size=self.kernel_size, 
                         padding='same', activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Ensure we get back to exact time_steps using a Lambda layer
        def crop_or_pad(x_input):
            current_shape = tf.shape(x_input)
            current_length = current_shape[1]
            if current_length > self.time_steps:
                return x_input[:, :self.time_steps, :]
            elif current_length < self.time_steps:
                padding = self.time_steps - current_length
                return tf.pad(x_input, [[0, 0], [0, padding], [0, 0]], 'CONSTANT')
            else:
                return x_input
                
        x = layers.Lambda(crop_or_pad, output_shape=(self.time_steps, 32))(x)

        # Output layers for mean and log variance
        mean = layers.Conv1D(filters=self.features, kernel_size=1, padding='same')(x)
        # Clamp log_var to prevent extreme values that could cause NaN
        log_var_raw = layers.Conv1D(filters=self.features, kernel_size=1, padding='same')(x)
        log_var = layers.Lambda(lambda x: tf.clip_by_value(x, -10.0, 10.0))(log_var_raw)
        outputs = layers.Concatenate()([mean, log_var])
        
        return Model(inputs, outputs, name='decoder')

    def _build_latent_discriminator(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs, name='latent_discriminator')

    def _build_stats_discriminator(self):
        inputs = layers.Input(shape=(self.time_steps, self.features))
        x = layers.Conv1D(filters=32, kernel_size=5, padding='causal', activation='relu')(inputs)
        x = layers.AvgPool1D(2)(x)
        x = layers.Conv1D(filters=64, kernel_size=5, padding='causal', activation='relu')(x)
        x = layers.AvgPool1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs, name='stats_discriminator')

    def summary(self):
        """Print summaries of all models"""
        self.logger.info("=== ENCODER ===")
        self.encoder.summary()
        self.logger.info("=== DECODER ===")
        self.decoder.summary()
        self.logger.info("=== LATENT DISCRIMINATOR ===")
        self.latent_discriminator.summary()
        self.logger.info("=== STATS DISCRIMINATOR ===")
        self.stats_discriminator.summary()

    def save_model(self, filepath: str, prefix: str):
        """Save all model components to disk"""
        
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # Save each model component
        encoder_path = os.path.join(filepath, f"cnn_{prefix}_encoder.keras")
        decoder_path = os.path.join(filepath, f"cnn_{prefix}_decoder.keras") 
        latent_disc_path = os.path.join(filepath, f"cnn_{prefix}_latent_discriminator.keras")
        stats_disc_path = os.path.join(filepath, f"cnn_{prefix}_stats_discriminator.keras")
        
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)
        self.latent_discriminator.save(latent_disc_path)
        self.stats_discriminator.save(stats_disc_path)
        
        if self.logger:
            self.logger.info(f"STAAR model saved to {filepath} with prefix '{prefix}'")
        else:
            print(f"STAAR model saved to {filepath} with prefix '{prefix}'")

    def load_model(self, filepath: str, prefix: str):
        """Load all model components from disk"""
        
        # Build paths for each component
        encoder_path = os.path.join(filepath, f"{prefix}_encoder.keras")
        decoder_path = os.path.join(filepath, f"{prefix}_decoder.keras")
        latent_disc_path = os.path.join(filepath, f"{prefix}_latent_discriminator.keras") 
        stats_disc_path = os.path.join(filepath, f"{prefix}_stats_discriminator.keras")
        
        # Check if all files exist
        if not all(os.path.exists(path) for path in [encoder_path, decoder_path, latent_disc_path, stats_disc_path]):
            raise FileNotFoundError(f"One or more model files not found for prefix '{prefix}' in {filepath}")
        
        # Load each model component
        self.encoder = load_model(encoder_path)
        self.decoder = load_model(decoder_path)
        self.latent_discriminator = load_model(latent_disc_path)
        self.stats_discriminator = load_model(stats_disc_path)
        
        self.logger.info(f"STAAR model loaded from {filepath} with prefix '{prefix}'")