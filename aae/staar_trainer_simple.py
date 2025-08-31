import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers, Model
import math
import keras

class StaarModelTrainer(Model):
    """
    Manages the entire training process for the STAAR model.
    - Inherits from tf.keras.Model to leverage the .fit() API.
    - Contains all loss functions and the custom 5-phase train_step.
    - Supports distributed training with MirroredStrategy.
    """
    def __init__(self, staar_model, garch_params, loss_weights, strategy=None, clip_norm=1.0, exclude_stats_disc=True):
        super().__init__()
        self.model = staar_model
        self.latent_dim = self.model.latent_dim
        self.features = self.model.features
        self.garch_params = garch_params
        self.loss_weights = loss_weights
        self.strategy = strategy or tf.distribute.get_strategy()
        self.clip_norm = clip_norm
        self.exclude_stats_disc = exclude_stats_disc

        self.ae_optimizer = None
        self.latent_disc_optimizer = None
        self.gen_optimizer = None
        self.stats_disc_optimizer = None
        self.decoder_adv_optimizer = None
        self.adv_loss_fn = None

        self.log_2pi = tf.constant(tf.math.log(2. * math.pi))
        # Define the prior distribution
        self.prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim))
        
        # Batch counter for alternating phase execution
        self.batch_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

    def compile(self, ae_opt, lat_disc_opt, gen_opt, stat_disc_opt, dec_adv_opt):
        # Provide a dummy loss since we use custom train_step
        super().compile(loss='mse', run_eagerly=True)
        self.ae_optimizer = ae_opt
        self.latent_disc_optimizer = lat_disc_opt
        self.gen_optimizer = gen_opt
        self.stats_disc_optimizer = stat_disc_opt
        self.decoder_adv_optimizer = dec_adv_opt
        self.adv_loss_fn = keras.losses.BinaryCrossentropy()

    def call(self, inputs, training=None):
        """
        Forward pass through the model. Required for tf.keras.Model subclass.
        Returns the reconstruction for the given inputs.
        """
        # Get latent code from encoder (deterministic)
        z = self.model.encoder(inputs, training=training)
        # Decode to get reconstruction
        reconstruction = self.model.decoder(z, training=training)
        return reconstruction

    # --- Loss Functions ---
    def _mse_loss(self, x_true, x_pred):
        """
        Simple MSE reconstruction loss for standard autoencoder.
        """
        return tf.reduce_mean(tf.square(x_true - x_pred))

    @tf.function
    def _acl_loss_fn(self, y_pred, max_lag=50):
        first_sequence = y_pred[0, :, 0]
        last_values = y_pred[1:, -1, 0]
        series = tf.concat([first_sequence, last_values], axis=0)
        series_len = tf.shape(series)[0]
        series_centered = series - tf.reduce_mean(series)
        fft_len = 2 * series_len
        autocorr = tf.signal.rfft(series_centered, fft_length=[fft_len])
        autocorr = tf.math.real(tf.signal.irfft(tf.math.conj(autocorr) * autocorr, fft_length=[fft_len]))
        variance = tf.reduce_sum(tf.square(series_centered))
        autocorr = autocorr[:max_lag+1] / variance
        return tf.reduce_mean(tf.square(autocorr[1:]))

    @tf.function
    def _garch_nll_loss_fn(self, y_pred):
        nu = 4.0
        omega, alpha, beta = self.garch_params['omega'], self.garch_params['alpha'], self.garch_params['beta']
        first_sequence = y_pred[0, :, 0]
        last_values = y_pred[1:, -1, 0]
        residuals = tf.concat([first_sequence, last_values], axis=0)
        residuals = residuals - tf.reduce_mean(residuals)
        
        def garch_step(prev_sigma_sq, current_residual_sq):
            return omega + alpha * current_residual_sq + beta * prev_sigma_sq
        
        unconditional_variance = omega / (1.0 - alpha - beta)
        conditional_variances = tf.scan(garch_step, tf.square(residuals), initializer=unconditional_variance)
        
        scale = tf.sqrt(conditional_variances * (nu - 2.0) / nu)
        t_dist = tfp.distributions.StudentT(df=nu, loc=0.0, scale=scale)
        log_likelihood = t_dist.log_prob(residuals)
        return -tf.reduce_sum(log_likelihood)

    # --- Helper methods for @tf.function compilation ---
    @tf.function
    def _compute_ae_loss(self, x):
        """Compute autoencoder MSE loss with gradients"""
        with tf.GradientTape() as tape:
            reconstruction = self.call(x, training=True)
            recon_loss = self._mse_loss(x, reconstruction)
        ae_vars = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
        ae_grads = tape.gradient(recon_loss, ae_vars)
        return recon_loss, ae_grads

    @tf.function
    def _compute_latent_disc_loss(self, x, per_replica_batch_size):
        """Compute latent discriminator loss with gradients"""
        with tf.GradientTape() as tape:
            fake_z = self.model.encoder(x, training=True)
            real_z = self.prior.sample(per_replica_batch_size)
            disc_fake = self.model.latent_discriminator(fake_z, training=True)
            disc_real = self.model.latent_discriminator(real_z, training=True)
            lat_disc_loss = self.adv_loss_fn(tf.ones_like(disc_real), disc_real) + self.adv_loss_fn(tf.zeros_like(disc_fake), disc_fake)
        ld_grads = tape.gradient(lat_disc_loss, self.model.latent_discriminator.trainable_variables)
        return lat_disc_loss, ld_grads

    @tf.function
    def _compute_gen_loss(self, x):
        """Compute generator (encoder) loss with gradients"""
        with tf.GradientTape() as tape:
            fake_z = self.model.encoder(x, training=True)
            disc_fake = self.model.latent_discriminator(fake_z, training=False)
            gen_loss = self.adv_loss_fn(tf.ones_like(disc_fake), disc_fake)
        gen_grads = tape.gradient(gen_loss, self.model.encoder.trainable_variables)
        return gen_loss, gen_grads

    @tf.function
    def _compute_stats_disc_loss(self, x):
        """Compute stats discriminator loss with gradients"""
        with tf.GradientTape() as tape:
            z = self.model.encoder(x, training=False)
            generated_x = self.model.decoder(z, training=False)
            stats_disc_fake = self.model.stats_discriminator(generated_x, training=True)
            stats_disc_real = self.model.stats_discriminator(x, training=True)
            stats_disc_loss = self.adv_loss_fn(tf.ones_like(stats_disc_real), stats_disc_real) + self.adv_loss_fn(tf.zeros_like(stats_disc_fake), stats_disc_fake)
        sd_grads = tape.gradient(stats_disc_loss, self.model.stats_discriminator.trainable_variables)
        return stats_disc_loss, sd_grads

    @tf.function
    def _compute_decoder_adv_loss(self, x):
        """Compute decoder adversarial loss with gradients"""
        with tf.GradientTape() as tape:
            z = self.model.encoder(x, training=False)
            generated_x = self.model.decoder(z, training=True)
            stats_disc_fake = self.model.stats_discriminator(generated_x, training=False)
            
            decoder_adv_loss = self.adv_loss_fn(tf.ones_like(stats_disc_fake), stats_disc_fake)
            acl = self._acl_loss_fn(generated_x)
            garch = self._garch_nll_loss_fn(generated_x)
            
            total_decoder_loss = (self.loss_weights['adv'] * decoder_adv_loss +
                                  self.loss_weights['acl'] * acl +
                                  self.loss_weights['garch'] * garch)
        
        dec_grads = tape.gradient(total_decoder_loss, self.model.decoder.trainable_variables)
        return decoder_adv_loss, acl, garch, dec_grads

    # --- The 5-Phase Training Step ---
    def train_step(self, data):
        # Unpack data tuple (x, y) - we only need x for autoencoder training
        x, _ = data
        batch_size = tf.shape(x)[0]
        
        # Get per-replica batch size for distributed training
        per_replica_batch_size = batch_size // self.strategy.num_replicas_in_sync

        # PHASE 1: AUTOENCODER (MSE)
        recon_loss, ae_grads = self._compute_ae_loss(x)
        ae_vars = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
        if ae_grads is not None and self.ae_optimizer is not None:
            ae_grads, _ = tf.clip_by_global_norm(ae_grads, self.clip_norm)
            self.ae_optimizer.apply_gradients(zip(ae_grads, ae_vars))

        #if self.batch_counter % 2 == 0:
        # PHASE 2: LATENT DISCRIMINATOR
        lat_disc_loss, ld_grads = self._compute_latent_disc_loss(x, per_replica_batch_size)
        ld_grads, _ = tf.clip_by_global_norm(ld_grads, self.clip_norm)
        if self.latent_disc_optimizer is not None:
            self.latent_disc_optimizer.apply_gradients(zip(ld_grads, self.model.latent_discriminator.trainable_variables))

            # PHASE 3: LATENT GENERATOR (ENCODER)
        gen_loss, gen_grads = self._compute_gen_loss(x)
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, self.clip_norm)
        if self.gen_optimizer is not None:
            self.gen_optimizer.apply_gradients(zip(gen_grads, self.model.encoder.trainable_variables))
        #else:
        #    # Skip latent discriminator and generator phases on odd batches
        #    lat_disc_loss = tf.constant(0.0)
        #    gen_loss = tf.constant(0.0)

        # PHASE 4 & 5: STATS DISCRIMINATOR AND GENERATOR (conditionally executed)
        if not self.exclude_stats_disc:
            # PHASE 4: STATS DISCRIMINATOR
            stats_disc_loss, sd_grads = self._compute_stats_disc_loss(x)
            if sd_grads is not None and self.stats_disc_optimizer is not None:
                if self.clip_norm is not None:
                    sd_grads, _ = tf.clip_by_global_norm(sd_grads, self.clip_norm)
                self.stats_disc_optimizer.apply_gradients(zip(sd_grads, self.model.stats_discriminator.trainable_variables))

            # PHASE 5: STATS GENERATOR (DECODER)
            decoder_adv_loss, acl, garch, dec_grads = self._compute_decoder_adv_loss(x)
            if dec_grads is not None and self.decoder_adv_optimizer is not None:
                if self.clip_norm is not None:
                    dec_grads, _ = tf.clip_by_global_norm(dec_grads, self.clip_norm)
                self.decoder_adv_optimizer.apply_gradients(zip(dec_grads, self.model.decoder.trainable_variables))
        else:
            # Set dummy losses when phases are excluded
            stats_disc_loss = tf.constant(0.0)
            decoder_adv_loss = tf.constant(0.0)
            acl = tf.constant(0.0)
            garch = tf.constant(0.0)

        # Increment batch counter
        self.batch_counter.assign_add(1)

        # Return loss key for compiled model compatibility
        return {"loss": recon_loss, "recon": recon_loss, 
                "lat_disc": lat_disc_loss, "gen": gen_loss, 
                "stat_disc": stats_disc_loss, "dec_adv": decoder_adv_loss, "acl": acl, "garch": garch
                }

    def test_step(self, data):
        """
        Validation step - compute reconstruction loss on validation data.
        We only care about reconstruction quality, not adversarial losses.
        """
        # Unpack data tuple (x, y) - we only need x for reconstruction
        x, _ = data
        
        # Forward pass to get reconstruction
        reconstruction = self.call(x, training=False)
        
        # Compute MSE reconstruction loss
        recon_loss = self._mse_loss(x, reconstruction)
        
        # Return loss that matches the compiled loss shape
        # The 'loss' key is what the compiled model expects for the dummy MSE loss
        return {"loss": recon_loss, "recon": recon_loss, "mse": recon_loss}


def create_distributed_trainer(staar_model, garch_params, loss_weights, 
                              ae_opt_config, lat_disc_opt_config, gen_opt_config, 
                              stat_disc_opt_config, dec_adv_opt_config, exclude_stats_disc=True, clip_norm=None):
    """
    Creates a StaarModelTrainer with MirroredStrategy for distributed training.
    
    Args:
        staar_model: The STAAR model architecture
        garch_params: GARCH parameters dictionary
        loss_weights: Loss weights dictionary
        *_opt_config: Optimizer configuration dictionaries with 'class' and 'kwargs' keys
        
    Returns:
        Compiled StaarModelTrainer ready for distributed training
        
    Example:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            trainer = create_distributed_trainer(
                staar_model, garch_params, loss_weights,
                {'class': tf.keras.optimizers.Adam, 'kwargs': {'learning_rate': 1e-4}},
                {'class': tf.keras.optimizers.Adam, 'kwargs': {'learning_rate': 1e-4}},
                # ... other optimizer configs
            )
    """
    strategy = tf.distribute.get_strategy()
    
    # Create trainer within strategy scope
    trainer = StaarModelTrainer(staar_model, garch_params, loss_weights, strategy, exclude_stats_disc=exclude_stats_disc, clip_norm=clip_norm)
    
    # Create optimizers within strategy scope
    ae_opt = ae_opt_config['class'](**ae_opt_config['kwargs'])
    lat_disc_opt = lat_disc_opt_config['class'](**lat_disc_opt_config['kwargs'])
    gen_opt = gen_opt_config['class'](**gen_opt_config['kwargs'])
    stat_disc_opt = stat_disc_opt_config['class'](**stat_disc_opt_config['kwargs'])
    dec_adv_opt = dec_adv_opt_config['class'](**dec_adv_opt_config['kwargs'])
    
    # Compile trainer
    trainer.compile(ae_opt, lat_disc_opt, gen_opt, stat_disc_opt, dec_adv_opt)
    
    return trainer

