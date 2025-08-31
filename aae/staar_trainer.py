import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers, Model
import keras
import math
import sys

class StaarModelTrainer(Model):
    """
    Manages the entire training process for the STAAR model.
    - Inherits from tf.keras.Model to leverage the .fit() API.
    - Contains all loss functions and the custom 5-phase train_step.
    - Supports distributed training with MirroredStrategy.
    """
    def __init__(self, staar_model, garch_params, loss_weights, strategy=None, clip_norm=None, exclude_stats_disc=True, 
                 cholesky_epsilon=1e-4, save_path="saved_models", model_prefix="checkpoint", 
                 kl_anneal_epochs=10, kl_min_weight=0.0, kl_max_weight=1.0, free_bit_lambda=2.0, use_free_bit=True, l1_k=100, gk_vol_k=10, std_loss_k=10):
        super().__init__()
        self.model = staar_model
        self.latent_dim = self.model.latent_dim
        self.features = self.model.features
        self.output_features = self.model.output_features
        self.garch_params = garch_params
        self.loss_weights = loss_weights
        self.strategy = strategy or tf.distribute.get_strategy()
        self.clip_norm = clip_norm
        self.exclude_stats_disc = exclude_stats_disc
        self.cholesky_epsilon = cholesky_epsilon
        self.save_path = save_path
        self.model_prefix = model_prefix
        
        # KL annealing parameters
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_min_weight = kl_min_weight
        self.kl_max_weight = kl_max_weight
        
        # Free bit parameters
        self.free_bit_lambda = free_bit_lambda
        self.use_free_bit = tf.constant(use_free_bit)

        self.ae_optimizer = None
        self.latent_disc_optimizer = None
        self.gen_optimizer = None
        self.stats_disc_optimizer = None
        self.decoder_adv_optimizer = None
        self.adv_loss_fn = None
        self.l1_k = l1_k
        self.gk_vol_k = gk_vol_k
        self.std_loss_k = std_loss_k
        self.gk_vol_epsilon = tf.constant(1e-8)

        self.log_2pi = tf.constant(tf.math.log(2. * math.pi))
        self.prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim))
        
        self.batch_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.epoch_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.test_batch_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

    def compile(self, ae_opt, lat_disc_opt, gen_opt, stat_disc_opt, dec_adv_opt):
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
        # Get posterior distribution from encoder
        posterior = self._get_posterior(self.model.encoder(inputs, training=training))
        # Sample from posterior
        z = posterior.sample()
        # Decode to get reconstruction (means + log_vars concatenated)
        decoder_output = self.model.decoder(z, training=training)
        # Extract only the means (first half of the output_features dimension)
        means = decoder_output[..., :self.output_features]
        return means

    def _get_posterior(self, enc_out):
        mean = enc_out[:, :self.latent_dim]
        chol_elements = enc_out[:, self.latent_dim:]
        
        L = tfp.math.fill_triangular(chol_elements)
        
        # Ensure diagonal elements are always positive and not too small
        diag_raw = tf.linalg.diag_part(L)
        #diag_clamped = tf.clip_by_value(diag_raw, -4.0, 4.0)  # Limit log scale
        diag_positive = tf.exp(diag_raw) + self.cholesky_epsilon
        L = tf.linalg.set_diag(L, diag_positive)
        
        # determinant of L
        #diag_elements = tf.linalg.diag_part(L)
        #det_L = tf.exp(tf.reduce_sum(tf.math.log(diag_elements + 1e-8), axis=-1))  # log determinant
        #tf.print("determinant of L (sample):", det_L[0], summarize=10)
        
        return tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=L)
    
    def _get_kl_weight(self):
        """Calculate KL annealing weight using linear schedule"""
        current_epoch = tf.cast(self.epoch_counter, tf.float32)
        anneal_epochs = tf.cast(self.kl_anneal_epochs, tf.float32)
        
        # Calculate progress through annealing schedule (0.0 to 1.0)
        progress = tf.clip_by_value(current_epoch / (anneal_epochs + 1e-8), 0.0, 1.0)
        
        # Linear annealing: interpolate between min and max weight
        weight = self.kl_min_weight + progress * (self.kl_max_weight - self.kl_min_weight)
        
        return tf.clip_by_value(weight, self.kl_min_weight, self.kl_max_weight)

    def _compute_free_bit_kl(self, posterior):
        """
        Compute KL divergence with free bit regularization to prevent posterior collapse.
        For MultivariateNormalTriL, we approximate per-dimension KL using diagonal elements.
        
        Args:
            posterior: The posterior distribution from the encoder (MultivariateNormalTriL)
            
        Returns:
            KL loss with free bit regularization
        """
        # Get posterior parameters
        posterior_mean = posterior.mean()  # Shape: (batch_size, latent_dim)
        posterior_cov = posterior.covariance()  # Shape: (batch_size, latent_dim, latent_dim)
        
        # Extract diagonal variances for approximation
        posterior_variance = tf.linalg.diag_part(posterior_cov)  # Shape: (batch_size, latent_dim)
        
        # Approximate per-dimension KL using diagonal elements
        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        kl_per_dim = 0.5 * (posterior_variance + tf.square(posterior_mean) - 1.0 - tf.math.log(tf.maximum(posterior_variance, 1e-8)))
        
        # Apply free bit: max(KL_dim, free_bit_lambda) for each dimension
        free_bit_kl_per_dim = tf.maximum(kl_per_dim, self.free_bit_lambda)
        
        # Average over batch and sum over dimensions
        return tf.reduce_mean(tf.reduce_sum(free_bit_kl_per_dim, axis=1))

    def _gaussian_nll_loss(self, x_true, dec_out):
        mean = dec_out[..., :self.output_features]
        log_var = dec_out[..., self.output_features:]
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
    def _compute_gk_volatility_loss(self, x_real, x_reconstructed):
        """
        Compute Garman-Klass volatility loss between real and reconstructed sequences.
        
        Args:
            x_real: Real input sequences (batch_size, time_steps, 4) [close, open, high, low]
            x_reconstructed: Reconstructed sequences (batch_size, time_steps, 4) [close, open, high, low]
        
        Returns:
            Scalar tensor representing the absolute difference in Garman-Klass volatility
        """
        # Extract OHLC prices for real and reconstructed data
        # Features are ordered as: close, open, high, low (indices 0, 1, 2, 3)
        close_real = x_real[:, :, 0]  # (batch_size, time_steps)
        open_real = x_real[:, :, 1]   # (batch_size, time_steps)
        high_real = x_real[:, :, 2]   # (batch_size, time_steps)
        low_real = x_real[:, :, 3]    # (batch_size, time_steps)
        
        close_recon = x_reconstructed[:, :, 0]  # (batch_size, time_steps)
        open_recon = x_reconstructed[:, :, 1]   # (batch_size, time_steps)
        high_recon = x_reconstructed[:, :, 2]   # (batch_size, time_steps)
        low_recon = x_reconstructed[:, :, 3]    # (batch_size, time_steps)
        
        # Compute Garman-Klass volatility for each time step
        # GK formula: ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
        # Use epsilon from constructor to avoid log(0) issues
        
        # Real data Garman-Klass volatility
        ln_hc_real = tf.math.log(tf.abs(high_real) + self.gk_vol_epsilon) - tf.math.log(tf.abs(close_real) + self.gk_vol_epsilon)
        ln_ho_real = tf.math.log(tf.abs(high_real) + self.gk_vol_epsilon) - tf.math.log(tf.abs(open_real) + self.gk_vol_epsilon)
        ln_lc_real = tf.math.log(tf.abs(low_real) + self.gk_vol_epsilon) - tf.math.log(tf.abs(close_real) + self.gk_vol_epsilon)
        ln_lo_real = tf.math.log(tf.abs(low_real) + self.gk_vol_epsilon) - tf.math.log(tf.abs(open_real) + self.gk_vol_epsilon)
        gk_vol_real = ln_hc_real * ln_ho_real + ln_lc_real * ln_lo_real
        
        # Reconstructed data Garman-Klass volatility
        ln_hc_recon = tf.math.log(tf.abs(high_recon) + self.gk_vol_epsilon) - tf.math.log(tf.abs(close_recon) + self.gk_vol_epsilon)
        ln_ho_recon = tf.math.log(tf.abs(high_recon) + self.gk_vol_epsilon) - tf.math.log(tf.abs(open_recon) + self.gk_vol_epsilon)
        ln_lc_recon = tf.math.log(tf.abs(low_recon) + self.gk_vol_epsilon) - tf.math.log(tf.abs(close_recon) + self.gk_vol_epsilon)
        ln_lo_recon = tf.math.log(tf.abs(low_recon) + self.gk_vol_epsilon) - tf.math.log(tf.abs(open_recon) + self.gk_vol_epsilon)
        gk_vol_recon = ln_hc_recon * ln_ho_recon + ln_lc_recon * ln_lo_recon
        
        # Compute absolute difference between real and reconstructed GK volatilities
        gk_vol_diff = tf.abs(gk_vol_real - gk_vol_recon)
        
        # Return mean absolute difference across all time steps and batches
        return tf.reduce_mean(gk_vol_diff)

    @tf.function
    def _compute_ae_loss(self, x):
        """Compute autoencoder loss (ELBO) with gradients"""
        with tf.GradientTape() as tape:
            posterior = self._get_posterior(self.model.encoder(x, training=True))
            z = posterior.sample()
            decoder_output = self.model.decoder(z, training=True)
            recon_loss = self._gaussian_nll_loss(x, decoder_output)
            
            # Add L1 reconstruction loss (absolute difference)
            means = decoder_output[..., :self.output_features]
            l1_loss = tf.reduce_mean(tf.abs(x - means))
            
            # Add Garman-Klass volatility loss
            gk_vol_loss = self._compute_gk_volatility_loss(x, means)
            
            # Add standard deviation loss
            input_std = tf.reduce_mean(tf.math.reduce_std(x, axis=1), axis=1)  # Std over time, then mean over features
            recon_std = tf.reduce_mean(tf.math.reduce_std(means, axis=1), axis=1)  # Std over time, then mean over features
            std_diff = tf.abs(input_std - recon_std)  # Absolute difference
            std_loss = tf.reduce_mean(std_diff)  # Mean across batch
            
            # Compute KL loss with optional free bit
            kl_loss = tf.cond(
                self.use_free_bit,
                lambda: self._compute_free_bit_kl(posterior),
                lambda: tf.reduce_mean(posterior.kl_divergence(self.prior))
            )
            
            # Apply KL annealing weight
            kl_weight = self._get_kl_weight()
            weighted_kl_loss = kl_weight * kl_loss
            #elbo_loss = recon_loss + weighted_kl_loss + self.l1_k * l1_loss
            elbo_loss =  recon_loss + weighted_kl_loss + self.l1_k * l1_loss + self.gk_vol_k * gk_vol_loss + self.std_loss_k * std_loss
        ae_vars = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
        ae_grads = tape.gradient(elbo_loss, ae_vars)
        return elbo_loss, recon_loss, kl_loss, kl_weight, l1_loss, gk_vol_loss, std_loss, ae_grads

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
        elbo_loss, recon_loss, kl_loss, kl_weight, l1_loss, gk_vol_loss, std_loss, ae_grads = self._compute_ae_loss(x)
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
        return {"loss": elbo_loss, "elbo": elbo_loss, "kl": kl_loss, "kl_weight": kl_weight, "recon": recon_loss, "l1": l1_loss, "gk_vol": gk_vol_loss, "std": std_loss,
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
        
        # Extract means from decoder output (first half of output_features dimension)
        means = decoder_output[..., :self.output_features]
        
        # Compute reconstruction loss (same as in training)
        recon_loss = self._gaussian_nll_loss(x, decoder_output)
        
        # Add L1 reconstruction loss (same as in training)
        l1_loss = tf.reduce_mean(tf.abs(x - means))
        
        # Add Garman-Klass volatility loss (same as in training)
        gk_vol_loss = self._compute_gk_volatility_loss(x, means)
        
        # Add standard deviation loss (same as in training)
        input_std = tf.math.reduce_std(x, axis=[1, 2])  # Std across time and features for each batch
        recon_std = tf.math.reduce_std(means, axis=[1, 2])  # Std across time and features for each batch
        std_diff = tf.abs(input_std - recon_std)  # Absolute difference
        std_loss = tf.reduce_mean(std_diff)  # Mean across batch
        
        kl_loss = tf.reduce_mean(posterior.kl_divergence(self.prior))
        # Apply KL annealing weight for validation too
        kl_weight = self._get_kl_weight()
        weighted_kl_loss = kl_weight * kl_loss
        elbo_loss = recon_loss + weighted_kl_loss + self.l1_k * l1_loss + self.gk_vol_k * gk_vol_loss + self.std_loss_k * std_loss
        
        # Also compute simple MSE between input and reconstructed means
        mse_loss = tf.reduce_mean(tf.square(x - means))
        
        # Extract log_var and compute comprehensive variance statistics
        log_var = decoder_output[..., self.output_features:]
        variances = tf.exp(log_var)
        overall_avg_variance = tf.reduce_mean(variances)

        recon_std = tf.math.reduce_std(means)
        
        # Additional variance collapse detection metrics
        var_flat = tf.reshape(variances, [-1])
        decoder_var_min = tf.reduce_min(var_flat)
        decoder_var_max = tf.reduce_max(var_flat)
        decoder_var_std = tf.math.reduce_std(var_flat)
        # Percentage of variances below a small threshold (indicating collapse)
        small_var_threshold = 1e-4
        small_var_fraction = tf.reduce_mean(tf.cast(var_flat < small_var_threshold, tf.float32))
        
        # Compute posterior covariance matrix
        posterior_cov = posterior.covariance()
        # Get covariance statistics: mean diagonal (variance) and mean off-diagonal (covariance)
        posterior_mean_variance = tf.reduce_mean(tf.linalg.diag_part(posterior_cov))
        posterior_mean_covariance = tf.reduce_mean(posterior_cov - tf.linalg.diag(tf.linalg.diag_part(posterior_cov)))
        
        # Compute determinant of posterior covariance matrix
        posterior_det = tf.linalg.det(posterior_cov)
        posterior_mean_det = tf.reduce_mean(posterior_det)
        
        # Compute eigenvalues and count small ones (numerical instability detection)
        eigenvalues = tf.linalg.eigvals(posterior_cov)
        eigenvalues_real = tf.math.real(eigenvalues)  # Take real part in case of numerical errors
        small_eigenval_threshold = 1e-6
        small_eigenval_count = tf.reduce_sum(tf.cast(eigenvalues_real < small_eigenval_threshold, tf.float32), axis=1)
        mean_small_eigenval_count = tf.reduce_mean(small_eigenval_count)
        
        # Count diagonal elements (variances) smaller than 1e-6
        diagonal_variances = tf.linalg.diag_part(posterior_cov)
        small_variance_threshold = 1e-6
        small_diagonal_var_count = tf.reduce_sum(tf.cast(diagonal_variances < small_variance_threshold, tf.float32), axis=-1)
        mean_small_diagonal_var_count = tf.reduce_mean(small_diagonal_var_count)

        
        
        # Return loss that matches the compiled loss shape (y should be same shape as means)
        # The 'loss' key is what the compiled model expects for the dummy MSE loss
        result_dict = {"loss": mse_loss, "elbo": elbo_loss, "recon": recon_loss, "kl": kl_loss, "kl_weight": kl_weight, "l1": l1_loss, "gk_vol": gk_vol_loss, "std": std_loss, "mse": mse_loss, 
                       "avg_variance": overall_avg_variance, "posterior_mean_var": posterior_mean_variance,
                       "posterior_mean_cov": posterior_mean_covariance, "posterior_mean_det": posterior_mean_det,
                       "small_eigenval_count": mean_small_eigenval_count, "small_diagonal_var_count": mean_small_diagonal_var_count,
                       "decoder_var_min": decoder_var_min, "decoder_var_max": decoder_var_max, "decoder_var_std": decoder_var_std, 
                       "small_var_fraction": small_var_fraction, "recon_std": recon_std
                       }
        
        # Log the contents of the result dictionary only occasionally (every 100 test steps)
        tf.cond(
            tf.equal(tf.math.mod(self.test_batch_counter, 100), 0),
            lambda: [tf.print("Test step metrics:"), 
                    [tf.print(f"  {key}: {value}") for key, value in result_dict.items()]],
            lambda: []
        )
        
        # Increment test batch counter
        self.test_batch_counter.assign_add(1)
        
        return result_dict

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of each epoch - may not be called automatically"""
        self.logger.info(f"Starting epoch {epoch + 1}")
        self.epoch_counter.assign(epoch)
    
    def set_epoch(self, epoch):
        """Manually set the current epoch for KL annealing"""
        self.epoch_counter.assign(epoch)

    def generate(self, num_samples: int, batch_size: int = 1000):
        """
        Generate sequences from the STAAR model by sampling from the prior and decoding.
        
        Args:
            num_samples: Number of sequences to generate
            batch_size: Batch size for generation (to manage memory)
            
        Returns:
            np.array: Generated sequences with shape (num_samples, window_size, features)
        """
        import numpy as np
        generated_sequences = []
        
        # Generate in batches to manage memory
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            # Sample from the prior distribution
            z = self.prior.sample(current_batch_size)  # Shape: (batch_size, latent_dim)
            
            # Decode to get sequences
            decoder_output = self.model.decoder(z, training=False)
            
            # Extract means from decoder output (ignore variances for generation)
            means = decoder_output[..., :self.output_features]  # Shape: (batch_size, window, features)
            
            # Convert to numpy and add to collection
            means_np = means.numpy()
            for i in range(current_batch_size):
                generated_sequences.append(means_np[i])
        
        return np.array(generated_sequences)


def create_distributed_trainer(staar_model, garch_params, loss_weights, 
                              ae_opt_config, lat_disc_opt_config, gen_opt_config, 
                              stat_disc_opt_config, dec_adv_opt_config, exclude_stats_disc=True, clip_norm=None, cholesky_epsilon=1e-6,
                              kl_anneal_epochs=10, kl_min_weight=0.0, kl_max_weight=1.0, std_loss_k=10):
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
    trainer = StaarModelTrainer(staar_model, garch_params, loss_weights, strategy, exclude_stats_disc=exclude_stats_disc, clip_norm=clip_norm, cholesky_epsilon=cholesky_epsilon,
                               kl_anneal_epochs=kl_anneal_epochs, kl_min_weight=kl_min_weight, kl_max_weight=kl_max_weight, std_loss_k=std_loss_k)
    
    # Create optimizers within strategy scope
    ae_opt = ae_opt_config['class'](**ae_opt_config['kwargs'])
    lat_disc_opt = lat_disc_opt_config['class'](**lat_disc_opt_config['kwargs'])
    gen_opt = gen_opt_config['class'](**gen_opt_config['kwargs'])
    stat_disc_opt = stat_disc_opt_config['class'](**stat_disc_opt_config['kwargs'])
    dec_adv_opt = dec_adv_opt_config['class'](**dec_adv_opt_config['kwargs'])
    
    # Compile trainer
    trainer.compile(ae_opt, lat_disc_opt, gen_opt, stat_disc_opt, dec_adv_opt)
    
    return trainer


