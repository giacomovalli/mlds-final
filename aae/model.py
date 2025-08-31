from logging import Logger
import keras as keras
import tensorflow as tf
from keras import layers, models
import time
import numpy as np


class StylizedAAE:
    def __init__(self, window_size: int = 120, latent_dim: int = 64, logger:Logger=None, strategy=None, latent_prior_kde=None):
        """
        Adversarial Autoencoder with Stylized Facts score penalty
        
        Args:
            window_size (int): Length of input sequences (default: 120)
            latent_dim (int): Dimension of latent space (default: 64)
            logger: Logger instance for logging messages
            strategy: tf.distribute.Strategy for multi-GPU/CPU training
        """
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.input_shape = (4, window_size)  # 4 channels (OHLC), window_size timesteps
        self.logger = logger
        self.strategy = strategy
        
        # Build networks within strategy scope if provided
        if self.strategy:
            with self.strategy.scope():
                self.encoder = self._build_encoder()
                self.decoder = self._build_decoder()
                self.discriminator = self._build_discriminator()
                self.autoencoder = self._build_autoencoder()
        else:
            self.encoder = self._build_encoder()
            self.decoder = self._build_decoder()
            self.discriminator = self._build_discriminator()
            self.autoencoder = self._build_autoencoder()
    
    def _build_encoder(self):
        """Build encoder network that maps input to latent space"""
        inputs = layers.Input(shape=self.input_shape, name='encoder_input')
        
        # Reshape from (batch, channels, timesteps) to (batch, timesteps, channels) for attention
        x = layers.Permute((2, 1))(inputs)  # (batch, window_size, 4)
        
        # First block: Multi-head attention + LSTM + Layer norm
        attention_1 = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attention_1])  # Residual connection
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        
        # Second block: Multi-head attention + LSTM + Layer norm
        attention_2 = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attention_2])  # Residual connection
        x = layers.LSTM(64, return_sequences=False)(x)  # Final LSTM doesn't return sequences
        x = layers.LayerNormalization()(x)
        
        # Dense layer to convert to latent space
        latent = layers.Dense(self.latent_dim, activation='linear', name='latent_output')(x)
        
        return models.Model(inputs, latent, name='encoder')
    
    def _build_decoder(self):
        """Build decoder network that reconstructs input from latent space"""
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Dense layer to expand to 4 * window_size
        x = layers.Dense(4 * self.window_size, activation='relu')(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Reshape to (batch, window_size, 4) for attention layers
        x = layers.Reshape((self.window_size, 4))(x)
        
        # First block: Multi-head attention + LSTM + Layer norm
        attention_1 = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attention_1])  # Residual connection
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        
        # Second block: Multi-head attention + LSTM + Layer norm
        attention_2 = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attention_2])  # Residual connection
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        
        # Final dense layer to get back to 4 channels
        x = layers.Dense(4, activation='linear')(x)  # (batch, window_size, 4)
        
        # Convert from (batch, timesteps, channels) to (batch, channels, timesteps)
        x = layers.Permute((2, 1))(x)  # (batch, 4, window_size)
        
        return models.Model(latent_inputs, x, name='decoder')
    
    def _build_discriminator(self):
        """Build discriminator network to distinguish real vs fake latent representations"""
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='discriminator_input')
        
        x = layers.Dense(256, activation='relu')(latent_inputs)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output probability that input is from real prior distribution
        validity = layers.Dense(1, activation='sigmoid', name='validity_output')(x)
        
        return models.Model(latent_inputs, validity, name='discriminator')
    
    def _build_autoencoder(self):
        """Build complete autoencoder by connecting encoder and decoder"""
        inputs = layers.Input(shape=self.input_shape, name='autoencoder_input')
        
        # Encode
        latent = self.encoder(inputs)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return models.Model(inputs, reconstructed, name='autoencoder')
    
    def compile_models(self):
        """Compile all models with appropriate optimizers and losses"""
        
        # Compile autoencoder for reconstruction
        self.autoencoder.compile(
            loss='mse',
            metrics=['mae']
        )
        
        # Compile discriminator
        self.discriminator.compile(
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def get_models(self):
        """Return all component models"""
        return {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'discriminator': self.discriminator,
            'autoencoder': self.autoencoder
        }
    
    def summary(self):
        """Print summaries of all models"""
        self.logger.info("=== ENCODER ===")
        self.encoder.summary()
        self.logger.info("=== DECODER ===")
        self.decoder.summary()
        self.logger.info("=== DISCRIMINATOR ===")
        self.discriminator.summary()
        self.logger.info("=== AUTOENCODER ===")
        self.autoencoder.summary()
    
    def train(self, data_sequence, epochs=10, discriminator_lr=0.0005, autoencoder_lr=0.001, generator_lr=0.0005):
        """
        Train the Adversarial Autoencoder using custom training loop with optional distributed training.
        
        Args:
            data_sequence: BatchedTimeseriesSequence for training data
            epochs (int): Number of training epochs
            discriminator_lr (float): Learning rate for discriminator
            autoencoder_lr (float): Learning rate for autoencoder
            generator_lr (float): Learning rate for generator (default: 0.0005)
        """
        if self.strategy is None:
            return self._train_no_strategy(data_sequence, epochs, discriminator_lr, autoencoder_lr, generator_lr)
        else:
            return self._train_with_strategy(data_sequence, epochs, discriminator_lr, autoencoder_lr, generator_lr)
    
    def _train_no_strategy(self, data_sequence, epochs=10, discriminator_lr=0.0005, autoencoder_lr=0.001, generator_lr=0.0005):
        """
        Train the Adversarial Autoencoder without distributed strategy.
        
        Args:
            data_sequence: BatchedTimeseriesSequence for training data
            epochs (int): Number of training epochs
            discriminator_lr (float): Learning rate for discriminator
            autoencoder_lr (float): Learning rate for autoencoder
            generator_lr (float): Learning rate for generator (default: 0.0005)
        """
        # Create optimizers
        d_optimizer = keras.optimizers.Adam(learning_rate=discriminator_lr)
        ae_optimizer = keras.optimizers.Adam(learning_rate=autoencoder_lr)
        g_optimizer = keras.optimizers.Adam(learning_rate=generator_lr)
        
        # Loss functions
        reconstruction_loss_fn = keras.losses.MeanSquaredError()
        adversarial_loss_fn = keras.losses.BinaryCrossentropy()
        
        # Training metrics
        reconstruction_loss_metric = keras.metrics.Mean(name='reconstruction_loss')
        discriminator_loss_metric = keras.metrics.Mean(name='discriminator_loss')
        generator_loss_metric = keras.metrics.Mean(name='generator_loss')
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        # Initialize loss tracking
        epoch_losses = {
            'reconstruction_loss': [],
            'discriminator_loss': [],
            'generator_loss': []
        }
        
        # Performance tracking
        total_batches = len(data_sequence)
        self.logger.info(f"Total batches per epoch: {total_batches}")
        
        # Analyze data sequence characteristics
        sample_batch_x, sample_batch_y = data_sequence[0]
        batch_size_actual = sample_batch_x.shape[0]
        sequence_length = sample_batch_x.shape[2] if len(sample_batch_x.shape) > 2 else "N/A"
        
        self.logger.info(f"Batch size: {batch_size_actual}")
        self.logger.info(f"Input shape per sample: {sample_batch_x.shape[1:] if len(sample_batch_x.shape) > 1 else sample_batch_x.shape}")
        self.logger.info(f"Total samples per epoch: {total_batches * batch_size_actual}")
        
        # Timing variables
        setup_time = 0
        forward_time = 0
        backward_time = 0
        data_loading_time = 0
        
        # Define training step
        @tf.function
        def train_step(real_data):
            batch_size = tf.shape(real_data)[0]
            
            # Generate fake latent samples from prior (standard normal)
            real_latent = tf.random.normal((batch_size, self.latent_dim))
            
            # === Train Autoencoder (Reconstruction) ===
            # Only encoder and decoder should be trainable
            self.discriminator.trainable = False
            
            with tf.GradientTape() as ae_tape:
                encoded = self.encoder(real_data, training=True)
                reconstructed = self.decoder(encoded, training=True)
                reconstruction_loss = reconstruction_loss_fn(real_data, reconstructed)
                
            ae_gradients = ae_tape.gradient(reconstruction_loss, 
                                          self.encoder.trainable_variables + self.decoder.trainable_variables)
            ae_optimizer.apply_gradients(zip(ae_gradients, 
                                           self.encoder.trainable_variables + self.decoder.trainable_variables))
            
            # === Train Discriminator ===
            # Only discriminator should be trainable, encoder should be frozen
            self.encoder.trainable = False
            self.decoder.trainable = False
            self.discriminator.trainable = True
            
            with tf.GradientTape() as d_tape:
                fake_latent = self.encoder(real_data, training=False)
                
                real_pred = self.discriminator(real_latent, training=True)
                fake_pred = self.discriminator(fake_latent, training=True)
                
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))
                
                d_loss_real = adversarial_loss_fn(real_labels, real_pred)
                d_loss_fake = adversarial_loss_fn(fake_labels, fake_pred)
                d_loss = (d_loss_real + d_loss_fake) / 2
                
            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
            
            # === Train Generator (Encoder to fool discriminator) ===
            # Only encoder should be trainable, discriminator should be frozen
            self.encoder.trainable = True
            self.decoder.trainable = False
            self.discriminator.trainable = False
            
            with tf.GradientTape() as g_tape:
                fake_latent = self.encoder(real_data, training=True)
                fake_pred = self.discriminator(fake_latent, training=False)
                g_loss = adversarial_loss_fn(real_labels, fake_pred)
                
            g_gradients = g_tape.gradient(g_loss, self.encoder.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, self.encoder.trainable_variables))
            
            # Reset all models to trainable for next iteration
            self.encoder.trainable = True
            self.decoder.trainable = True
            self.discriminator.trainable = True
            
            return reconstruction_loss, d_loss, g_loss
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Reset metrics
            reconstruction_loss_metric.reset_state()
            discriminator_loss_metric.reset_state()
            generator_loss_metric.reset_state()
            
            # Reset timing for this epoch
            epoch_data_loading_time = float(0)
            epoch_forward_time = float(0)
            epoch_backward_time = float(0)

            # Training loop over batches
            for batch_idx in range(len(data_sequence)):
                batch_start_time = time.time()
                
                # Data loading timing
                data_load_start = time.time()
                real_data, _ = data_sequence[batch_idx]
                real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)
                data_load_time = time.time() - data_load_start
                epoch_data_loading_time += data_load_time
                
                # Training step timing
                train_start = time.time()
                reconstruction_loss, d_loss, g_loss = train_step(real_data)
                train_time = time.time() - train_start
                epoch_forward_time += train_time
                
                # Metrics update timing
                metrics_start = time.time()
                reconstruction_loss_metric.update_state(reconstruction_loss)
                discriminator_loss_metric.update_state(d_loss)
                generator_loss_metric.update_state(g_loss)
                metrics_time = time.time() - metrics_start
                epoch_backward_time += metrics_time
                
                # Log batch timing every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0:
                    batch_total_time = time.time() - batch_start_time
                    self.logger.info(f"  Batch {batch_idx}/{total_batches}: {batch_total_time:.3f}s "
                                   f"(data: {data_load_time:.3f}s, train: {train_time:.3f}s, metrics: {metrics_time:.3f}s)")
            
            # Store epoch losses
            epoch_recon_loss = float(reconstruction_loss_metric.result())
            epoch_disc_loss = float(discriminator_loss_metric.result())
            epoch_gen_loss = float(generator_loss_metric.result())
            
            epoch_losses['reconstruction_loss'].append(epoch_recon_loss)
            epoch_losses['discriminator_loss'].append(epoch_disc_loss)
            epoch_losses['generator_loss'].append(epoch_gen_loss)
            
            # Epoch timing summary
            epoch_total_time = time.time() - epoch_start_time
            avg_batch_time = epoch_total_time / total_batches
            
            # Print epoch results with timing
            self.logger.info(f"Losses - Reconstruction: {epoch_recon_loss:.4f}, Discriminator: {epoch_disc_loss:.4f}, Generator: {epoch_gen_loss:.4f}")
            self.logger.info(f"Epoch {epoch + 1} timing - Total: {epoch_total_time:.2f}s, Avg/batch: {avg_batch_time:.3f}s")
            self.logger.info(f"  Data loading: {epoch_data_loading_time:.2f}s ({epoch_data_loading_time/epoch_total_time*100:.1f}%)")
            self.logger.info(f"  Training: {epoch_forward_time:.2f}s ({epoch_forward_time/epoch_total_time*100:.1f}%)")
            self.logger.info(f"  Metrics: {epoch_backward_time:.2f}s ({epoch_backward_time/epoch_total_time*100:.1f}%)")
        
        self.logger.info("Training completed!")
        return epoch_losses
    
    def _train_with_strategy(self, data_sequence, epochs=10, discriminator_lr=0.0005, autoencoder_lr=0.001, generator_lr=0.0005):
        """
        Train the Adversarial Autoencoder with distributed strategy.
        
        Args:
            data_sequence: BatchedTimeseriesSequence for training data
            epochs (int): Number of training epochs
            discriminator_lr (float): Learning rate for discriminator
            autoencoder_lr (float): Learning rate for autoencoder
            generator_lr (float): Learning rate for generator (default: 0.0005)
        """
        
        # Create optimizers within strategy scope
        with self.strategy.scope():
            d_optimizer = keras.optimizers.Adam(learning_rate=discriminator_lr)
            ae_optimizer = keras.optimizers.Adam(learning_rate=autoencoder_lr)
            g_optimizer = keras.optimizers.Adam(learning_rate=generator_lr)
            
            # Loss functions
            reconstruction_loss_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            adversarial_loss_fn = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
        # Training metrics
        reconstruction_loss_metric = keras.metrics.Mean(name='reconstruction_loss')
        discriminator_loss_metric = keras.metrics.Mean(name='discriminator_loss')
        generator_loss_metric = keras.metrics.Mean(name='generator_loss')
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Using distributed strategy: {type(self.strategy).__name__}")
        
        # Initialize loss tracking
        epoch_losses = {
            'reconstruction_loss': [],
            'discriminator_loss': [],
            'generator_loss': []
        }
        
        # Performance tracking
        total_batches = len(data_sequence)
        self.logger.info(f"Total batches per epoch: {total_batches}")
        
        # Analyze data sequence characteristics
        sample_batch_x, sample_batch_y = data_sequence[0]
        batch_size_actual = sample_batch_x.shape[0]
        sequence_length = sample_batch_x.shape[2] if len(sample_batch_x.shape) > 2 else "N/A"
        
        self.logger.info(f"Batch size: {batch_size_actual}")
        self.logger.info(f"Input shape per sample: {sample_batch_x.shape[1:] if len(sample_batch_x.shape) > 1 else sample_batch_x.shape}")
        self.logger.info(f"Total samples per epoch: {total_batches * batch_size_actual}")
        
        # Timing variables
        setup_time = 0
        forward_time = 0
        backward_time = 0
        data_loading_time = 0
        
        # Define training step
        @tf.function
        def train_step(real_data):
            batch_size = tf.shape(real_data)[0]
            
            # Generate fake latent samples from prior (standard normal)
            real_latent = tf.random.normal((batch_size, self.latent_dim))
            
            # === Train Autoencoder (Reconstruction) ===
            # Only encoder and decoder should be trainable
            self.discriminator.trainable = False
            
            with tf.GradientTape() as ae_tape:
                encoded = self.encoder(real_data, training=True)
                reconstructed = self.decoder(encoded, training=True)
                reconstruction_loss = reconstruction_loss_fn(real_data, reconstructed)
                reconstruction_loss = tf.nn.compute_average_loss(
                    reconstruction_loss, 
                    global_batch_size=batch_size * self.strategy.num_replicas_in_sync
                )
                
            ae_gradients = ae_tape.gradient(reconstruction_loss, 
                                          self.encoder.trainable_variables + self.decoder.trainable_variables)
            ae_optimizer.apply_gradients(zip(ae_gradients, 
                                           self.encoder.trainable_variables + self.decoder.trainable_variables))
            
            # === Train Discriminator ===
            # Only discriminator should be trainable, encoder should be frozen
            self.encoder.trainable = False
            self.decoder.trainable = False
            self.discriminator.trainable = True
            
            with tf.GradientTape() as d_tape:
                fake_latent = self.encoder(real_data, training=False)
                
                real_pred = self.discriminator(real_latent, training=True)
                fake_pred = self.discriminator(fake_latent, training=True)
                
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))
                
                d_loss_real = adversarial_loss_fn(real_labels, real_pred)
                d_loss_fake = adversarial_loss_fn(fake_labels, fake_pred)
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss = tf.nn.compute_average_loss(
                    d_loss, 
                    global_batch_size=batch_size * self.strategy.num_replicas_in_sync
                )
                
            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
            
            # === Train Generator (Encoder to fool discriminator) ===
            # Only encoder should be trainable, discriminator should be frozen
            self.encoder.trainable = True
            self.decoder.trainable = False
            self.discriminator.trainable = False
            
            with tf.GradientTape() as g_tape:
                fake_latent = self.encoder(real_data, training=True)
                fake_pred = self.discriminator(fake_latent, training=False)
                g_loss = adversarial_loss_fn(real_labels, fake_pred)
                g_loss = tf.nn.compute_average_loss(
                    g_loss, 
                    global_batch_size=batch_size * self.strategy.num_replicas_in_sync
                )
                
            g_gradients = g_tape.gradient(g_loss, self.encoder.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, self.encoder.trainable_variables))
            
            # Reset all models to trainable for next iteration
            self.encoder.trainable = True
            self.decoder.trainable = True
            self.discriminator.trainable = True
            
            return reconstruction_loss, d_loss, g_loss
        
        # Create a distributed training step function
        def distributed_train_step(real_data):
            losses = self.strategy.run(train_step, args=(real_data,))
            # Reduce PerReplica values to scalars
            reconstruction_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, losses[0], axis=None)
            d_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, losses[1], axis=None)
            g_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, losses[2], axis=None)
            return reconstruction_loss, d_loss, g_loss
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Reset metrics
            reconstruction_loss_metric.reset_state()
            discriminator_loss_metric.reset_state()
            generator_loss_metric.reset_state()
            
            # Reset timing for this epoch
            epoch_data_loading_time = float(0)
            epoch_forward_time = float(0)
            epoch_backward_time = float(0)

            # Training loop over batches
            for batch_idx in range(len(data_sequence)):
                batch_start_time = time.time()
                
                # Data loading timing
                data_load_start = time.time()
                real_data, _ = data_sequence[batch_idx]
                real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)
                data_load_time = time.time() - data_load_start
                epoch_data_loading_time += data_load_time
                
                # Training step timing
                train_start = time.time()
                reconstruction_loss, d_loss, g_loss = distributed_train_step(real_data)
                train_time = time.time() - train_start
                epoch_forward_time += train_time
                
                # Metrics update timing
                metrics_start = time.time()
                reconstruction_loss_metric.update_state(reconstruction_loss)
                discriminator_loss_metric.update_state(d_loss)
                generator_loss_metric.update_state(g_loss)
                metrics_time = time.time() - metrics_start
                epoch_backward_time += metrics_time
                
                # Log batch timing every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0:
                    batch_total_time = time.time() - batch_start_time
                    self.logger.info(f"  Batch {batch_idx}/{total_batches}: {batch_total_time:.3f}s "
                                   f"(data: {data_load_time:.3f}s, train: {train_time:.3f}s, metrics: {metrics_time:.3f}s)")
            
            # Store epoch losses
            epoch_recon_loss = float(reconstruction_loss_metric.result())
            epoch_disc_loss = float(discriminator_loss_metric.result())
            epoch_gen_loss = float(generator_loss_metric.result())
            
            epoch_losses['reconstruction_loss'].append(epoch_recon_loss)
            epoch_losses['discriminator_loss'].append(epoch_disc_loss)
            epoch_losses['generator_loss'].append(epoch_gen_loss)
            
            # Epoch timing summary
            epoch_total_time = time.time() - epoch_start_time
            avg_batch_time = epoch_total_time / total_batches
            
            # Print epoch results with timing
            self.logger.info(f"Losses - Reconstruction: {epoch_recon_loss:.4f}, Discriminator: {epoch_disc_loss:.4f}, Generator: {epoch_gen_loss:.4f}")
            self.logger.info(f"Epoch {epoch + 1} timing - Total: {epoch_total_time:.2f}s, Avg/batch: {avg_batch_time:.3f}s")
            self.logger.info(f"  Data loading: {epoch_data_loading_time:.2f}s ({epoch_data_loading_time/epoch_total_time*100:.1f}%)")
            self.logger.info(f"  Training: {epoch_forward_time:.2f}s ({epoch_forward_time/epoch_total_time*100:.1f}%)")
            self.logger.info(f"  Metrics: {epoch_backward_time:.2f}s ({epoch_backward_time/epoch_total_time*100:.1f}%)")
        
        self.logger.info("Training completed!")
        return epoch_losses
    
    def save_model(self, filepath: str, prefix: str):
        """
        Save the complete AAE model components to a Keras file.
        
        Args:
            filepath (str): Path where the model should be saved (without extension)
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save each component
        self.encoder.save(f"{filepath}{prefix}_encoder.keras")
        self.decoder.save(f"{filepath}{prefix}_decoder.keras")
        self.discriminator.save(f"{filepath}{prefix}_discriminator.keras")
        self.autoencoder.save(f"{filepath}{prefix}_autoencoder.keras")
        
        self.logger.info("Model saved successfully:")
        self.logger.info(f"  - Encoder: {filepath}{prefix}_encoder.keras")
        self.logger.info(f"  - Decoder: {filepath}{prefix}_decoder.keras")
        self.logger.info(f"  - Discriminator: {filepath}{prefix}_discriminator.keras")
        self.logger.info(f"  - Autoencoder: {filepath}{prefix}_autoencoder.keras")
    
    def load_model(self, filepath: str, prefix: str):
        """
        Load a previously saved AAE model from Keras files, replacing the current model state.
        
        Args:
            filepath (str): Path where the model was saved (without extension)
        """
        import os
        
        # Check if all required files exist
        required_files = [
            f"{filepath}{prefix}_encoder.keras",
            f"{filepath}{prefix}_decoder.keras", 
            f"{filepath}{prefix}_discriminator.keras",
            f"{filepath}{prefix}_autoencoder.keras"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required model file not found: {file}")
        
        # Load the models
        encoder = keras.models.load_model(f"{filepath}{prefix}_encoder.keras")
        decoder = keras.models.load_model(f"{filepath}{prefix}_decoder.keras")
        discriminator = keras.models.load_model(f"{filepath}{prefix}_discriminator.keras")
        autoencoder = keras.models.load_model(f"{filepath}{prefix}_autoencoder.keras")
        
        # Extract model parameters from loaded models
        window_size = encoder.input_shape[2]  # (None, 4, window_size)
        latent_dim = encoder.output_shape[1]  # (None, latent_dim)
        
        # Update instance parameters
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.input_shape = (4, window_size)
        
        # Replace the models with loaded ones
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.autoencoder = autoencoder
        
        self.logger.info("Model loaded successfully from:")
        self.logger.info(f"  - Encoder: {filepath}{prefix}_encoder.keras")
        self.logger.info(f"  - Decoder: {filepath}{prefix}_decoder.keras")
        self.logger.info(f"  - Discriminator: {filepath}{prefix}_discriminator.keras")
        self.logger.info(f"  - Autoencoder: {filepath}{prefix}_autoencoder.keras")
        self.logger.info(f"Model parameters updated: window_size={window_size}, latent_dim={latent_dim}")