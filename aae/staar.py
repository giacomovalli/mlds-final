import tensorflow as tf
import keras as keras
from keras import layers, Model
import os
from keras.models import load_model

@tf.function
@keras.saving.register_keras_serializable()
def scaled_tanh(x):
    """
    Custom activation function that behaves like tanh but returns values in range [-1, 2].
    
    This function applies tanh to scale inputs to [-1, 1], then linearly transforms 
    the result to map [-1, 1] to [-1, 2].
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with values in range [-1, 2]
    """
    tanh_x = tf.tanh(x)
    return 1.5 * tanh_x + 0.5

class StaarModel:
    """
    Builds and contains all neural network components for the STAAR model.
    - Encoder
    - Decoder
    - Latent Discriminator
    - Statistical Discriminator
    """
    def __init__(self, logger, time_steps=120, features=4, latent_dim=32,
                 lstm_units=32, num_heads=4, key_dim=32, num_blocks=2, output_features=4):
        self.time_steps = time_steps
        self.features = features
        self.output_features = output_features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.num_blocks = num_blocks
        self.logger = logger

        # Build all the component models upon initialization
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.latent_discriminator = self._build_latent_discriminator()
        self.stats_discriminator = self._build_stats_discriminator()

    def _build_encoder(self):
        inputs = layers.Input(shape=(self.time_steps, self.features))
        x = inputs
        # Up-sample to lstm_units dimensionality
        x = layers.Dense(self.lstm_units, activation='relu')(x)

        # --- New Block Structure ---
        for _ in range(self.num_blocks):
            # 1. Multi-Head Attention followed by Add & LayerNorm
            mha_input = x
            mha_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.key_dim
            )(query=mha_input, value=mha_input, key=mha_input)
            x = layers.LayerNormalization()(mha_input + mha_output)

            # 2. LSTM followed by LayerNorm
            lstm_input = x
            lstm_output = layers.LSTM(self.lstm_units, return_sequences=True)(lstm_input)
            x = layers.LayerNormalization()(lstm_output+lstm_input)
        
        # --- Flatten and Final Projection ---
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='tanh')(x)
        num_chol = int(self.latent_dim * (self.latent_dim + 1) / 2)
        outputs = layers.Dense(self.latent_dim + num_chol, name='encoder_outputs')(x)
        
        return Model(inputs, outputs, name='encoder')

    def _build_decoder_o(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.time_steps * self.lstm_units, activation='tanh')(inputs)
        x = layers.Reshape((self.time_steps, self.lstm_units))(x)
        orig_input = x

        # --- New Block Structure ---
        for _ in range(self.num_blocks):
            # 1. Multi-Head Attention followed by Add & LayerNorm
            #mha_input = x
            #mha_output = layers.MultiHeadAttention(
            #    num_heads=self.num_heads, key_dim=self.key_dim
            #)(query=mha_input, value=mha_input, key=mha_input)
            #x = layers.LayerNormalization()(mha_input + mha_output)

            # 2. LSTM followed by LayerNorm
            lstm_input = x
            lstm_output = layers.LSTM(self.lstm_units, return_sequences=True)(x)
            x = layers.LayerNormalization()(lstm_output + orig_input)

        mean = layers.TimeDistributed(layers.Dense(self.output_features))(x)
        log_var = layers.TimeDistributed(layers.Dense(self.output_features))(x)
        outputs = layers.Concatenate()([mean, log_var])
        return Model(inputs, outputs, name='decoder')

    def _build_decoder_conv(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.time_steps * self.lstm_units, activation='sigmoid')(inputs)
        x = layers.Reshape((self.time_steps, self.lstm_units))(x)
        
        # Add 1D convolution layers with pooling
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)

        mean = layers.TimeDistributed(layers.Dense(self.output_features))(x)
        log_var = layers.TimeDistributed(layers.Dense(self.output_features))(x)
        outputs = layers.Concatenate()([mean, log_var])
        return Model(inputs, outputs, name='decoder')

    def _build_decoder_ss(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.time_steps * self.lstm_units, activation='sigmoid')(inputs)
        x = layers.Dense(self.time_steps * self.output_features * 2, activation='relu')(x)
        mean_flat = layers.Dense(self.time_steps * self.output_features)(x)
        log_var_flat = layers.Dense(self.time_steps * self.output_features)(x)
        mean = layers.Reshape((self.time_steps, self.output_features))(mean_flat)
        log_var = layers.Reshape((self.time_steps, self.output_features))(log_var_flat)
        
        outputs = layers.Concatenate()([mean, log_var])
        return Model(inputs, outputs, name='decoder')

    def _build_decoder(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.time_steps * self.lstm_units, activation='tanh')(inputs)
        x = layers.Reshape((self.time_steps, self.lstm_units))(x)
        
        # Add 1D convolution layers with pooling
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        
        #orig_input = x
        # --- New Block Structure ---
        for _ in range(self.num_blocks//2):
            mha_input = x
            mha_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.key_dim
            )(query=mha_input, value=mha_input, key=mha_input)
            x = layers.LayerNormalization()(mha_input + mha_output)

            lstm_input = x
            lstm_output = layers.LSTM(self.lstm_units, return_sequences=True)(x)
            x = layers.LayerNormalization()(lstm_output + lstm_input)

        mean = layers.TimeDistributed(layers.Dense(self.output_features))(x)
        log_var = layers.TimeDistributed(layers.Dense(self.output_features))(x)
        outputs = layers.Concatenate()([mean, log_var])
        return Model(inputs, outputs, name='decoder')

    def _build_latent_discriminator(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation='tanh')(inputs)
        x = layers.Dense(32, activation='tanh')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs, name='latent_discriminator')

    def _build_stats_discriminator(self):
        inputs = layers.Input(shape=(self.time_steps, self.output_features))
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
        encoder_path = os.path.join(filepath, f"{prefix}_encoder.keras")
        decoder_path = os.path.join(filepath, f"{prefix}_decoder.keras") 
        latent_disc_path = os.path.join(filepath, f"{prefix}_latent_discriminator.keras")
        stats_disc_path = os.path.join(filepath, f"{prefix}_stats_discriminator.keras")
        
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

        print(encoder_path)
        
        # Check if all files exist
        if not all(os.path.exists(path) for path in [encoder_path, decoder_path, latent_disc_path, stats_disc_path]):
            raise FileNotFoundError(f"One or more model files not found for prefix '{prefix}' in {filepath}.")
        
        # Load each model component
        self.encoder = load_model(encoder_path)
        self.decoder = load_model(decoder_path)
        self.latent_discriminator = load_model(latent_disc_path)
        self.stats_discriminator = load_model(stats_disc_path)
        
        self.logger.info(f"STAAR model loaded from {filepath} with prefix '{prefix}'")
