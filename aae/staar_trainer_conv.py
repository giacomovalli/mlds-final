import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, Model
import math

class StaarModelTrainer(Model):
    """
    Manages the entire training process for the STAAR model.
    - Inherits from tf.keras.Model to leverage the .fit() API.
    - Contains all loss functions and the custom 5-phase train_step.
    - Supports distributed training with MirroredStrategy.
    """
    def __init__(self, staar_model, garch_params, loss_weights, strategy=None, clip_norm=None, exclude_stats_disc=True):
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

        self.pi = tf.constant(math.pi)
        self.log_2pi = tf.constant(tf.math.log(2. * self.pi))
        self.prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim))
        
        self.batch_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

    def compile(self, ae_opt, lat_disc_opt, gen_opt, stat_disc_opt, dec_adv_opt):
        super().compile(loss='mse', run_eagerly=True)
        self.ae_optimizer = ae_opt
        self.latent_disc_optimizer = lat_disc_opt
        self.gen_optimizer = gen_opt
        self.stats_disc_optimizer = stat_disc_opt
        self.decoder_adv_optimizer = dec_adv_opt
        self.adv_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def call(self, inputs, training=None):
        """
        Forward pass through the model. Required for tf.keras.Model subclass.
        Returns the reconstruction for the given inputs.
        """
        # Get posterior distribution from encoder
        posterior = self._get_posterior(self.model.encoder(inputs, training=training))
        # Sample from posterior
        z = posterior.sample()
        # Decode to get reconstruction (means + log_vars concatenated)
        decoder_output = self.model.decoder(z, training=training)
        # Extract only the means (first half of the features dimension)
        means = decoder_output[..., :self.features]
        return means

    def _get_posterior(self, enc_out):
        mean = enc_out[:, :self.latent_dim]
        chol_elements = enc_out[:, self.latent_dim:]
        L = tfp.math.fill_triangular(chol_elements)
        L = tf.linalg.set_diag(L, tf.exp(tf.linalg.diag_part(L)))
        return tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=L)

    def _gaussian_nll_loss(self, x_true, dec_out):
        mean = dec_out[..., :self.features]
        log_var = dec_out[..., self.features:]
        squared_diff = tf.square(x_true - mean)
        inv_var = tf.exp(-log_var)
        log_likelihood = -0.5 * (self.log_2pi + log_var + squared_diff * inv_var)
        return -tf.reduce_mean(tf.reduce_sum(log_likelihood, axis=[1, 2]))

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

    @tf.function
    def _compute_ae_loss(self, x):
        """Compute autoencoder loss (ELBO) with gradients"""
        with tf.GradientTape() as tape:
            posterior = self._get_posterior(self.model.encoder(x, training=True))
            z = posterior.sample()
            decoder_output = self.model.decoder(z, training=True)
            recon_loss = self._gaussian_nll_loss(x, decoder_output)
            kl_loss = tf.reduce_mean(posterior.kl_divergence(self.prior))
            elbo_loss = recon_loss + kl_loss
        ae_vars = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
        ae_grads = tape.gradient(elbo_loss, ae_vars)
        return elbo_loss, recon_loss, kl_loss, ae_grads

    @tf.function
    def _compute_latent_disc_loss(self, x, per_replica_batch_size):
        """Compute latent discriminator loss with gradients"""
        with tf.GradientTape() as tape:
            fake_z = self._get_posterior(self.model.encoder(x, training=True)).sample()
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
            fake_z = self._get_posterior(self.model.encoder(x, training=True)).sample()
            disc_fake = self.model.latent_discriminator(fake_z, training=False)
            gen_loss = self.adv_loss_fn(tf.ones_like(disc_fake), disc_fake)
        gen_grads = tape.gradient(gen_loss, self.model.encoder.trainable_variables)
        return gen_loss, gen_grads

    @tf.function
    def _compute_stats_disc_loss(self, x):
        """Compute stats discriminator loss with gradients"""
        with tf.GradientTape() as tape:
            generated_x = self.call(x, training=False)
            stats_disc_fake = self.model.stats_discriminator(generated_x, training=True)
            stats_disc_real = self.model.stats_discriminator(x, training=True)
            stats_disc_loss = self.adv_loss_fn(tf.ones_like(stats_disc_real), stats_disc_real) + self.adv_loss_fn(tf.zeros_like(stats_disc_fake), stats_disc_fake)
        sd_grads = tape.gradient(stats_disc_loss, self.model.stats_discriminator.trainable_variables)
        return stats_disc_loss, sd_grads

    @tf.function
    def _compute_decoder_adv_loss(self, x):
        """Compute decoder adversarial loss with gradients"""
        with tf.GradientTape() as tape:
            generated_x = self.call(x, training=True)
            stats_disc_fake = self.model.stats_discriminator(generated_x, training=False)
            
            decoder_adv_loss = self.adv_loss_fn(tf.ones_like(stats_disc_fake), stats_disc_fake)
            acl = self._acl_loss_fn(generated_x)
            garch = self._garch_nll_loss_fn(generated_x)
            
            total_decoder_loss = (self.loss_weights['adv'] * decoder_adv_loss +
                                  self.loss_weights['acl'] * acl +
                                  self.loss_weights['garch'] * garch)
        
        dec_grads = tape.gradient(total_decoder_loss, self.model.decoder.trainable_variables)
        return decoder_adv_loss, acl, garch, dec_grads

    def train_step(self, data):
        x, _ = data
        batch_size = tf.shape(x)[0]
        
        per_replica_batch_size = batch_size // self.strategy.num_replicas_in_sync

        # PHASE 1: AUTOENCODER (ELBO)
        elbo_loss, recon_loss, kl_loss, ae_grads = self._compute_ae_loss(x)
        ae_vars = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
        if ae_grads is not None and self.ae_optimizer is not None:
            if self.clip_norm is not None:
                ae_grads, _ = tf.clip_by_global_norm(ae_grads, self.clip_norm)
            self.ae_optimizer.apply_gradients(zip(ae_grads, ae_vars))

        # PHASE 2: LATENT DISCRIMINATOR
        lat_disc_loss, ld_grads = self._compute_latent_disc_loss(x, per_replica_batch_size)
        if ld_grads is not None and self.latent_disc_optimizer is not None:
            if self.clip_norm is not None:
                ld_grads, _ = tf.clip_by_global_norm(ld_grads, self.clip_norm)
            self.latent_disc_optimizer.apply_gradients(zip(ld_grads, self.model.latent_discriminator.trainable_variables))

            # PHASE 3: LATENT GENERATOR (ENCODER)
        gen_loss, gen_grads = self._compute_gen_loss(x)
        if gen_grads is not None and self.gen_optimizer is not None:
            if self.clip_norm is not None:
                gen_grads, _ = tf.clip_by_global_norm(gen_grads, self.clip_norm)
            self.gen_optimizer.apply_gradients(zip(gen_grads, self.model.encoder.trainable_variables))

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
        return {"loss": elbo_loss, "elbo": elbo_loss, "kl": kl_loss, "recon": recon_loss, 
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
        posterior = self._get_posterior(self.model.encoder(x, training=False))
        z = posterior.sample()
        decoder_output = self.model.decoder(z, training=False)
        
        # Extract means from decoder output (first half of features dimension)
        means = decoder_output[..., :self.features]
        
        # Compute reconstruction loss (same as in training)
        recon_loss = self._gaussian_nll_loss(x, decoder_output)
        kl_loss = tf.reduce_mean(posterior.kl_divergence(self.prior))
        elbo_loss = recon_loss + kl_loss
        
        # Also compute simple MSE between input and reconstructed means
        mse_loss = tf.reduce_mean(tf.square(x - means))
        
        # Return loss that matches the compiled loss shape (y should be same shape as means)
        # The 'loss' key is what the compiled model expects for the dummy MSE loss
        return {"loss": mse_loss, "elbo": elbo_loss, "recon": recon_loss, "kl": kl_loss, "mse": mse_loss}


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