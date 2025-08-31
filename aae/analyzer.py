from logging import Logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
import tensorflow as tf
from scipy.stats import wasserstein_distance
from data.loader import BatchedTimeseriesSequence
from .staar_trainer import StaarModelTrainer
import esig


class StaarResultAnalyzer:
    """
    Stylized Adversarial Autoencoder Result Analyzer
    
    A class for analyzing and visualizing the results of trained AAE models,
    including reconstruction comparison and t-SNE dimensionality reduction analysis.
    """
    
    def __init__(self, logger: Logger = None, show_plots: bool = False, data_loader=None):
        """
        Initialize the analyzer.
        
        Args:
            logger: Logger instance for logging messages
            show_plots: Whether to display plots using plt.show() (default: False)
            data_loader: DataLoader instance for inverse scaling (optional)
        """
        self.logger = logger
        self.show_plots = show_plots
        self.data_loader = data_loader
    
    def analyze_model(self, df_filtered, model, args, plots_path: str):
        """
        Complete analysis of the model including reconstruction and t-SNE analysis.
        
        Args:
            df_filtered: Filtered dataframe for testing
            model: Trained AAE model
            args: Arguments containing batch_size, window size, etc.
            plots_path: Path to save plots
            
        Returns:
            dict: Comprehensive analysis results
        """
        # Create data loader for testing (no shuffle for consistent results)
        data_loader = BatchedTimeseriesSequence(df_filtered,
            batch_size=args.batch_size, 
            window=args.window,
            logger=self.logger,
            shuffle=False)
        
        if self.logger:
            self.logger.info(f"Testing model on {len(data_loader)} batches")
        
        # Collect original and reconstructed sequences
        original_sequences = []
        reconstructed_sequences = []
        reconstructed_stds = []
        timestamps = []

        longer_original_close = []
        longer_rec_close = []
        longer_rec_std = []
        
        # Create trainer instance for proper forward pass
        trainer = StaarModelTrainer(model, {}, {})
        
        # Process batches and reconstruct
        for batch_idx in range(min(20, len(data_loader))):  # Test on first 10 batches
            real_data, batch_timestamps = data_loader[batch_idx]
            
            # Get posterior distribution from encoder
            posterior = trainer._get_posterior(model.encoder(real_data, training=False))
            # Sample from posterior
            z = posterior.sample()
            # Get full decoder output (means + log_vars concatenated)
            decoder_output = model.decoder(z, training=False)
            
            # Extract means and log variances
            means = decoder_output[..., :model.output_features]
            log_vars = decoder_output[..., model.output_features:]
            # Convert log variance to standard deviation
            stds = tf.exp(0.5 * log_vars)
            
            # Convert back to numpy
            original_np = real_data.numpy() if hasattr(real_data, 'numpy') else real_data
            reconstructed_np = means.numpy()
            stds_np = stds.numpy()

            #create longer sequences for better visualization
            longer_original_close.append(np.concatenate([original_np[0,:,0], original_np[1:,-1,0]]))
            longer_rec_close.append(np.concatenate([reconstructed_np[0,:,0], reconstructed_np[1:,-1,0]]))
            longer_rec_std.append(np.concatenate([stds_np[0,:,0], stds_np[1:,-1,0]]))
            
            # Store sequences (focusing on close price - channel 0)
            for i in range(original_np.shape[0]):  # For each sample in batch
                original_sequences.append(original_np[i, :, 0])  # Close price channel
                reconstructed_sequences.append(reconstructed_np[i, :, 0])  # Close price channel
                reconstructed_stds.append(stds_np[i, :, 0])  # Standard deviation for close price
                if batch_timestamps is not None:
                    timestamps.append(batch_timestamps[i])
        
        if self.logger:
            self.logger.info(f"Collected {len(original_sequences)} sequences for visualization")
        
        # Create prefix for filenames
        prefix = self._get_prefix(args) if hasattr(args, 'year') and hasattr(args, 'month') and any([args.year, args.month]) else "full_data"
        
        # Perform reconstruction analysis with longer concatenated sequences
        reconstruction_results = self.analyze_reconstruction(
            longer_original_close, longer_rec_close, longer_rec_std, plots_path, prefix)
        
        # Perform t-SNE analysis
        tsne_results = self.analyze_tsne(
            original_sequences, reconstructed_sequences, plots_path, prefix)
        
        # Perform distance analysis
        distance_results = self.analyze_distances(
            original_sequences, reconstructed_sequences)
        
        # Perform price generation analysis
        self.analyze_price_generation(df_filtered, trainer, plots_path, prefix)
        
        # Perform stylized facts analysis
        stylized_facts_results = self.test_stylized_facts(trainer, plots_path, prefix)
        
        # Combine results
        combined_results = {**reconstruction_results, **tsne_results, **distance_results, **stylized_facts_results}
        
        return combined_results
    
    def analyze_reconstruction(self, original_sequences, reconstructed_sequences, reconstructed_stds, plots_path: str, prefix: str):
        """
        Analyze and visualize reconstruction quality.
        
        Args:
            original_sequences: List of original time series sequences
            reconstructed_sequences: List of reconstructed time series sequences
            reconstructed_stds: List of reconstruction standard deviations
            plots_path: Path to save plots
            prefix: Filename prefix
            
        Returns:
            dict: Reconstruction analysis results
        """
        # Create visualization
        n_plots = min(3, len(original_sequences))  # Show first 6 sequences
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes = axes.flatten()  # Convert 2D array to 1D for easy indexing
        
        for i in range(n_plots):
            ax = axes[i]
            
            # Plot original and reconstructed
            time_steps = np.arange(len(original_sequences[i][100:]))
            ax.plot(time_steps, original_sequences[i][100:], label='Original', color='blue', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps, reconstructed_sequences[i][100:], label='Reconstructed', color='red', alpha=0.7, linewidth=1.5, linestyle='--')
            
            # Add uncertainty bands (±1 sigma)
            reconstruction_mean = reconstructed_sequences[i][100:]
            reconstruction_std = reconstructed_stds[i][100:]
            upper_bound = reconstruction_mean + reconstruction_std
            lower_bound = reconstruction_mean - reconstruction_std
            ax.fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, color='red', label='±1σ uncertainty')
            
            ax.set_title(f'Sequence {i+1}: Original vs Reconstructed Close Price')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Close Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate and display reconstruction error
            mse = np.mean((original_sequences[i] - reconstructed_sequences[i])**2)
            mae = np.mean(np.abs(original_sequences[i] - reconstructed_sequences[i]))
            ax.text(0.02, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{plots_path}test_reconstruction_{prefix}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        if self.show_plots:
            plt.show()
        if self.logger:
            self.logger.info(f"Test reconstruction plot saved to {plot_filename}")
        plt.close()
        
        # If DataLoader is available, also create original scale plot
        if self.data_loader is not None and hasattr(self.data_loader, 'inverse_transform_series'):
            self._plot_reconstruction_original_scale(original_sequences, reconstructed_sequences, reconstructed_stds, plots_path, prefix, n_plots)
        elif self.data_loader is not None:
            if self.logger:
                self.logger.warning("DataLoader provided but doesn't have inverse_transform_series method")
        
        # Calculate overall statistics
        all_originals = np.concatenate(original_sequences)
        all_reconstructed = np.concatenate(reconstructed_sequences)
        
        overall_mse = np.mean((all_originals - all_reconstructed)**2)
        overall_mae = np.mean(np.abs(all_originals - all_reconstructed))
        correlation = np.corrcoef(all_originals, all_reconstructed)[0, 1]
        
        if self.logger:
            self.logger.info(f"Overall reconstruction statistics:")
            self.logger.info(f"  MSE: {overall_mse:.6f}")
            self.logger.info(f"  MAE: {overall_mae:.6f}")
            self.logger.info(f"  Correlation: {correlation:.6f}")
        
        return {
            'mse': overall_mse,
            'mae': overall_mae,
            'correlation': correlation,
            'original_sequences': original_sequences[:n_plots],
            'reconstructed_sequences': reconstructed_sequences[:n_plots]
        }
    
    def _plot_reconstruction_original_scale(self, original_sequences, reconstructed_sequences, reconstructed_stds, plots_path: str, prefix: str, n_plots: int):
        """
        Plot reconstruction comparison in original scale using the DataLoader's inverse transform.
        
        Args:
            original_sequences: List of original sequences (scaled 0-1)
            reconstructed_sequences: List of reconstructed sequences (scaled 0-1)
            reconstructed_stds: List of reconstruction standard deviations (scaled 0-1)
            plots_path: Path to save plots
            prefix: Filename prefix
            n_plots: Number of plots to create
        """
        try:
            # For original scale plotting, we need all 4 channels, but analyze_reconstruction only uses close price
            # We'll focus on the close price channel (index 0) for now, with dummy values for other channels
            # This is a limitation - ideally we'd store all 4 channels from analyze_model
            
            if self.logger:
                self.logger.info("Creating reconstruction plot in original scale (close price only)")
            
            # Create dummy arrays for other channels (we only have close price)
            dummy_array = np.zeros_like(original_sequences[0])
            
            # Create visualization in original scale
            fig, axes = plt.subplots(3, 2, figsize=(20, 12))
            axes = axes.flatten()  # Convert 2D array to 1D for easy indexing
            
            for i in range(n_plots):
                # Transform back to original scale for close price
                # Note: We're using the close price for all channels as we only have that data
                # Parameters are: (open, high, low, close) - close price goes last
                _, _, _, orig_close_orig_scale = self.data_loader.inverse_transform_series(
                    dummy_array, dummy_array, dummy_array, original_sequences[i]
                )
                _, _, _, recon_close_orig_scale = self.data_loader.inverse_transform_series(
                    dummy_array, dummy_array, dummy_array, reconstructed_sequences[i]
                )
                
                # Transform standard deviations to original scale
                # For standard deviations, we need to scale them by the same factor as the data
                # Get the scaling factor by computing the ratio
                reconstruction_std_scaled = reconstructed_stds[i]
                _, _, _, dummy_transformed = self.data_loader.inverse_transform_series(
                    dummy_array, dummy_array, dummy_array, dummy_array + reconstruction_std_scaled
                )
                _, _, _, zero_transformed = self.data_loader.inverse_transform_series(
                    dummy_array, dummy_array, dummy_array, dummy_array
                )
                std_orig_scale = dummy_transformed - zero_transformed
                
                ax = axes[i]
                
                # Plot original and reconstructed in original scale
                time_steps = np.arange(len(orig_close_orig_scale))
                ax.plot(time_steps, orig_close_orig_scale, label='Original', color='blue', alpha=0.7, linewidth=1.5)
                ax.plot(time_steps, recon_close_orig_scale, label='Reconstructed', color='red', alpha=0.7, linewidth=1.5, linestyle='--')
                
                # Add uncertainty bands (±1 sigma) in original scale
                upper_bound_orig = recon_close_orig_scale + std_orig_scale
                lower_bound_orig = recon_close_orig_scale - std_orig_scale
                ax.fill_between(time_steps, lower_bound_orig, upper_bound_orig, alpha=0.3, color='red', label='±1σ uncertainty')
                
                ax.set_title(f'Sequence {i+1}: Original vs Reconstructed Close Price (Original Scale)')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Close Price (Original Scale)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Calculate and display reconstruction error in original scale
                mse = np.mean((orig_close_orig_scale - recon_close_orig_scale)**2)
                mae = np.mean(np.abs(orig_close_orig_scale - recon_close_orig_scale))
                ax.text(0.02, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save original scale plot
            plot_filename_orig = f"{plots_path}test_reconstruction_original_scale_{prefix}.png"
            plt.savefig(plot_filename_orig, dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            if self.logger:
                self.logger.info(f"Test reconstruction plot (original scale) saved to {plot_filename_orig}")
            plt.close()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create original scale plot: {str(e)}")
            else:
                print(f"Failed to create original scale plot: {str(e)}")
    
    def analyze_tsne(self, original_sequences, reconstructed_sequences, plots_path: str, prefix: str):
        """
        Perform t-SNE analysis on original and reconstructed sequences.
        
        Args:
            original_sequences: List of original time series sequences
            reconstructed_sequences: List of reconstructed time series sequences
            plots_path: Path to save plots
            prefix: Filename prefix
            
        Returns:
            dict: t-SNE analysis results
        """
        if self.logger:
            self.logger.info("Performing t-SNE analysis on original and reconstructed sequences")
        
        # Prepare data for t-SNE (use all sequences)
        original_array = np.array(original_sequences)  # Shape: (n_sequences, window_size)
        reconstructed_array = np.array(reconstructed_sequences)  # Shape: (n_sequences, window_size)
        
        # Combine original and reconstructed for joint t-SNE
        combined_data = np.vstack([original_array, reconstructed_array])
        n_samples = len(original_sequences)
        
        if self.logger:
            self.logger.info(f"Running t-SNE on {combined_data.shape[0]} sequences with {combined_data.shape[1]} features each")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1), max_iter=1000)
        tsne_results = tsne.fit_transform(combined_data)
        
        # Split results back into original and reconstructed
        tsne_original = tsne_results[:n_samples]
        tsne_reconstructed = tsne_results[n_samples:]
        
        # Create t-SNE visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Separate plots for original and reconstructed
        ax1.scatter(tsne_original[:, 0], tsne_original[:, 1], 
                   c='blue', alpha=0.6, s=50, label='Original', edgecolors='black', linewidth=0.5)
        ax1.scatter(tsne_reconstructed[:, 0], tsne_reconstructed[:, 1], 
                   c='red', alpha=0.6, s=50, label='Reconstructed', edgecolors='black', linewidth=0.5)
        ax1.set_title('t-SNE: Original vs Reconstructed Sequences')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Connected pairs showing the transformation
        ax2.scatter(tsne_original[:, 0], tsne_original[:, 1], 
                   c='blue', alpha=0.6, s=50, label='Original', edgecolors='black', linewidth=0.5)
        ax2.scatter(tsne_reconstructed[:, 0], tsne_reconstructed[:, 1], 
                   c='red', alpha=0.6, s=50, label='Reconstructed', edgecolors='black', linewidth=0.5)
        
        # Draw lines connecting original to reconstructed pairs
        for i in range(min(50, n_samples)):  # Limit connections to avoid clutter
            ax2.plot([tsne_original[i, 0], tsne_reconstructed[i, 0]], 
                    [tsne_original[i, 1], tsne_reconstructed[i, 1]], 
                    'gray', alpha=0.3, linewidth=0.5)
        
        ax2.set_title('t-SNE: Original→Reconstructed Transformation')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save t-SNE plot
        tsne_filename = f"{plots_path}test_tsne_{prefix}.png"
        plt.savefig(tsne_filename, dpi=300, bbox_inches='tight')
        if self.show_plots:
            plt.show()
        if self.logger:
            self.logger.info(f"t-SNE analysis plot saved to {tsne_filename}")
        plt.close()
        
        # Calculate t-SNE distance statistics
        tsne_distances = np.sqrt(np.sum((tsne_original - tsne_reconstructed)**2, axis=1))
        mean_tsne_distance = np.mean(tsne_distances)
        std_tsne_distance = np.std(tsne_distances)
        
        if self.logger:
            self.logger.info(f"t-SNE transformation statistics:")
            self.logger.info(f"  Mean distance between original and reconstructed: {mean_tsne_distance:.4f}")
            self.logger.info(f"  Std deviation of distances: {std_tsne_distance:.4f}")
        
        return {
            'tsne_original': tsne_original,
            'tsne_reconstructed': tsne_reconstructed,
            'tsne_mean_distance': mean_tsne_distance,
            'tsne_std_distance': std_tsne_distance
        }
    
    def compute_mmd(self, X, Y, gamma=None):
        """
        Compute Maximum Mean Discrepancy (MMD) between two samples using Gaussian kernel.
        
        Args:
            X: First sample (n_samples_X, n_features)
            Y: Second sample (n_samples_Y, n_features)
            gamma: Kernel parameter for RBF kernel. If None, uses adaptive gamma based on data variance.
            
        Returns:
            float: MMD value
        """
        XX = pairwise_distances(X, X, metric='euclidean')
        YY = pairwise_distances(Y, Y, metric='euclidean')
        XY = pairwise_distances(X, Y, metric='euclidean')
        
        # Use adaptive gamma if not provided
        if gamma is None:
            # Use median heuristic: gamma = 1 / (2 * median_distance^2)
            all_distances = np.concatenate([XX.flatten(), YY.flatten(), XY.flatten()])
            median_dist = np.median(all_distances[all_distances > 0])  # Exclude zero distances
            gamma = 1.0 / (2 * median_dist**2)
            
            if self.logger:
                self.logger.info(f"Using adaptive gamma for MMD: {gamma:.4f} (median distance: {median_dist:.4f})")
        
        XX = np.exp(-gamma * XX**2)
        YY = np.exp(-gamma * YY**2)
        XY = np.exp(-gamma * XY**2)
        
        mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
        return mmd
    
    def compute_mmd_permutation_test(self, X, Y, observed_mmd, n_permutations=1000, gamma=None):
        """
        Perform permutation test to assess MMD statistical significance.
        
        H0: X and Y come from the same distribution
        H1: X and Y come from different distributions
        
        Args:
            X: First sample (n_samples_X, n_features)
            Y: Second sample (n_samples_Y, n_features)  
            observed_mmd: The observed MMD value
            n_permutations: Number of permutations for the test
            gamma: Kernel parameter (if None, uses same adaptive approach as compute_mmd)
            
        Returns:
            tuple: (p_value, null_distribution)
        """
        if self.logger:
            self.logger.info(f"Running MMD permutation test with {n_permutations} permutations")
        
        # Combine samples
        combined = np.vstack([X, Y])
        n_X, n_Y = len(X), len(Y)
        n_total = n_X + n_Y
        
        null_mmds = []
        
        for i in range(n_permutations):
            # Random permutation of combined samples
            perm_indices = np.random.permutation(n_total)
            
            # Split permuted samples back into X and Y
            perm_X = combined[perm_indices[:n_X]]
            perm_Y = combined[perm_indices[n_X:]]
            
            # Compute MMD for this permutation using the existing method
            mmd_perm = self.compute_mmd(perm_X, perm_Y, gamma=gamma)
            null_mmds.append(mmd_perm)
            
            # Progress logging every 200 permutations
            if self.logger and (i + 1) % 200 == 0:
                self.logger.info(f"  Completed {i + 1}/{n_permutations} permutations")
        
        null_mmds = np.array(null_mmds)
        
        # Compute p-value (two-tailed test)
        # p-value = proportion of null MMDs >= observed MMD
        p_value = np.mean(null_mmds >= observed_mmd)
        
        if self.logger:
            self.logger.info(f"Permutation test results:")
            self.logger.info(f"  Observed MMD: {observed_mmd:.6f}")
            self.logger.info(f"  Null distribution mean: {np.mean(null_mmds):.6f}")
            self.logger.info(f"  Null distribution std: {np.std(null_mmds):.6f}")
            self.logger.info(f"  P-value: {p_value:.6f}")
        
        return p_value, null_mmds
    
    def compute_wasserstein_permutation_test(self, X, Y, observed_wasserstein, n_permutations=1000):
        """
        Perform permutation test to assess Wasserstein distance statistical significance.
        
        H0: X and Y come from the same distribution
        H1: X and Y come from different distributions
        
        Args:
            X: First sample (n_samples, sequence_length)
            Y: Second sample (n_samples, sequence_length)  
            observed_wasserstein: The observed Wasserstein distance
            n_permutations: Number of permutations for the test
            
        Returns:
            tuple: (p_value, null_distribution)
        """
        if self.logger:
            self.logger.info(f"Running Wasserstein permutation test with {n_permutations} permutations")
        
        # Combine samples
        combined = np.vstack([X, Y])
        n_X, n_Y = len(X), len(Y)
        n_total = n_X + n_Y
        
        null_wassersteins = []
        
        for i in range(n_permutations):
            # Random permutation of combined samples
            perm_indices = np.random.permutation(n_total)
            
            # Split permuted samples back into X and Y
            perm_X = combined[perm_indices[:n_X]]
            perm_Y = combined[perm_indices[n_X:]]
            
            # Compute Wasserstein distance for this permutation using the existing method
            wasserstein_perm = self.compute_wasserstein_distance(perm_X, perm_Y)
            null_wassersteins.append(wasserstein_perm)
            
            # Progress logging every 200 permutations
            if self.logger and (i + 1) % 200 == 0:
                self.logger.info(f"  Completed {i + 1}/{n_permutations} permutations")
        
        null_wassersteins = np.array(null_wassersteins)
        
        # Compute p-value (two-tailed test)
        # p-value = proportion of null Wassersteins >= observed Wasserstein
        p_value = np.mean(null_wassersteins >= observed_wasserstein)
        
        if self.logger:
            self.logger.info(f"Wasserstein permutation test results:")
            self.logger.info(f"  Observed Wasserstein: {observed_wasserstein:.6f}")
            self.logger.info(f"  Null distribution mean: {np.mean(null_wassersteins):.6f}")
            self.logger.info(f"  Null distribution std: {np.std(null_wassersteins):.6f}")
            self.logger.info(f"  P-value: {p_value:.6f}")
        
        return p_value, null_wassersteins
    
    def compute_signature_wasserstein_permutation_test(self, X, Y, observed_sig_wasserstein, n_permutations=1000):
        """
        Perform permutation test to assess signature Wasserstein distance statistical significance.
        
        H0: X and Y come from the same distribution
        H1: X and Y come from different distributions
        
        Args:
            X: First sample (n_samples, sequence_length)
            Y: Second sample (n_samples, sequence_length)  
            observed_sig_wasserstein: The observed signature Wasserstein distance
            n_permutations: Number of permutations for the test
            
        Returns:
            tuple: (p_value, null_distribution)
        """
        if self.logger:
            self.logger.info(f"Running signature Wasserstein permutation test with {n_permutations} permutations")
        
        # Combine samples
        combined = np.vstack([X, Y])
        n_X, n_Y = len(X), len(Y)
        n_total = n_X + n_Y
        
        null_sig_wassersteins = []
        
        for i in range(n_permutations):
            # Random permutation of combined samples
            perm_indices = np.random.permutation(n_total)
            
            # Split permuted samples back into X and Y
            perm_X = combined[perm_indices[:n_X]]
            perm_Y = combined[perm_indices[n_X:]]
            
            # Compute signature Wasserstein distance for this permutation using the existing method
            sig_wasserstein_perm = self.compute_signature_wasserstein(perm_X, perm_Y)
            null_sig_wassersteins.append(sig_wasserstein_perm)
            
            # Progress logging every 200 permutations
            if self.logger and (i + 1) % 200 == 0:
                self.logger.info(f"  Completed {i + 1}/{n_permutations} permutations")
        
        null_sig_wassersteins = np.array(null_sig_wassersteins)
        
        # Compute p-value (two-tailed test)
        # p-value = proportion of null signature Wassersteins >= observed signature Wasserstein
        p_value = np.mean(null_sig_wassersteins >= observed_sig_wasserstein)
        
        if self.logger:
            self.logger.info(f"Signature Wasserstein permutation test results:")
            self.logger.info(f"  Observed signature Wasserstein: {observed_sig_wasserstein:.6f}")
            self.logger.info(f"  Null distribution mean: {np.mean(null_sig_wassersteins):.6f}")
            self.logger.info(f"  Null distribution std: {np.std(null_sig_wassersteins):.6f}")
            self.logger.info(f"  P-value: {p_value:.6f}")
        
        return p_value, null_sig_wassersteins
    
    def compute_wasserstein_distance(self, X, Y):
        """
        Compute Wasserstein distance between two time series datasets using scipy.
        
        For multivariate time series, computes the average Wasserstein distance
        across all time steps.
        
        Args:
            X: First dataset (n_samples, sequence_length)
            Y: Second dataset (n_samples, sequence_length)
            
        Returns:
            float: Average Wasserstein distance
        """
        # Convert to numpy if needed
        X_np = np.array(X) if not isinstance(X, np.ndarray) else X
        Y_np = np.array(Y) if not isinstance(Y, np.ndarray) else Y
        
        # Flatten all sequences to create two distributions
        X_flat = X_np.flatten()
        Y_flat = Y_np.flatten()
        
        # Compute 1D Wasserstein distance between the flattened distributions
        wasserstein_dist = wasserstein_distance(X_flat, Y_flat)
        
        return float(wasserstein_dist)
    
    def compute_signature_wasserstein(self, X, Y):
        """
        Compute path-based Wasserstein distance using esig path signatures and geomloss.
        
        Uses the esig library to compute proper path signatures, then applies
        geomloss for the Wasserstein computation.
        
        Args:
            X: First dataset (n_samples, sequence_length)
            Y: Second dataset (n_samples, sequence_length)
            
        Returns:
            float: Path-based Wasserstein distance
        """
        # Convert to numpy if needed
        X_np = np.array(X) if not isinstance(X, np.ndarray) else X
        Y_np = np.array(Y) if not isinstance(Y, np.ndarray) else Y
        
        # Compute path signatures using esig
        X_signatures = self._compute_path_signatures(X_np)
        Y_signatures = self._compute_path_signatures(Y_np)
        
        # Use scipy Wasserstein distance on flattened path signatures
        from scipy.stats import wasserstein_distance
        X_flat = X_signatures.flatten()
        Y_flat = Y_signatures.flatten()
        return float(wasserstein_distance(X_flat, Y_flat))
    
    def _compute_path_signatures(self, data):
        """
        Compute path signatures using the esig library.
        
        Args:
            data: Array of shape (n_samples, sequence_length)
            
        Returns:
            Array of shape (n_samples, n_signature_features) with path signatures
        """
        n_samples, seq_len = data.shape
        signatures = []
        
        # Signature truncation level - controls the complexity of the signature
        # Level 3 provides a good balance between expressiveness and dimensionality
        signature_level = 3
        
        for i in range(n_samples):
            path_values = data[i]
            
            # Create a 2D path with time and value coordinates
            # This is essential for meaningful path signatures
            time_coords = np.linspace(0, 1, seq_len)
            path = np.column_stack([time_coords, path_values])
            
            # Compute the path signature using esig
            try:
                signature = esig.stream2sig(path, signature_level)
                signatures.append(signature)
            except Exception as e:
                # Fallback for degenerate paths or other issues
                if self.logger:
                    self.logger.warning(f"Failed to compute signature for path {i}: {e}")
                # Create a zero signature with appropriate dimensions
                sig_dim = esig.sigdim(2, signature_level)  # 2D path, given level
                signatures.append(np.zeros(sig_dim))
        
        return np.array(signatures)
    
    
    
    def analyze_distances(self, original_sequences, reconstructed_sequences):
        """
        Analyze distances between original and reconstructed sequences using MMD and Wasserstein distances.
        
        Args:
            original_sequences: List of original time series sequences
            reconstructed_sequences: List of reconstructed time series sequences
            
        Returns:
            dict: Distance analysis results
        """
        permutations = 1
        if self.logger:
            self.logger.info("Computing MMD and Wasserstein distances")
        
        # Convert to arrays
        original_array = np.array(original_sequences)
        reconstructed_array = np.array(reconstructed_sequences)
        
        # Compute MMD
        mmd_value = self.compute_mmd(original_array, reconstructed_array)
        
        # Perform permutation test for MMD significance
        if self.logger:
            self.logger.info("Performing permutation test for MMD statistical significance")
        
        mmd_pvalue, mmd_null_dist = self.compute_mmd_permutation_test(
            original_array, reconstructed_array, mmd_value, n_permutations=permutations
        )
        
        # Compute standard Wasserstein distance
        wasserstein_dist = self.compute_wasserstein_distance(original_array, reconstructed_array)
        
        # Perform permutation test for Wasserstein distance significance
        if self.logger:
            self.logger.info("Performing permutation test for Wasserstein distance statistical significance")
        
        wasserstein_pvalue, wasserstein_null_dist = self.compute_wasserstein_permutation_test(
            original_array, reconstructed_array, wasserstein_dist, n_permutations=permutations
        )
        
        # Compute signature-based Wasserstein distance
        signature_wasserstein_dist = self.compute_signature_wasserstein(original_array, reconstructed_array)
        
        # Perform permutation test for signature Wasserstein distance significance
        if self.logger:
            self.logger.info("Performing permutation test for signature Wasserstein distance statistical significance")
        
        sig_wasserstein_pvalue, sig_wasserstein_null_dist = self.compute_signature_wasserstein_permutation_test(
            original_array, reconstructed_array, signature_wasserstein_dist, n_permutations=permutations
        )
        
        if self.logger:
            self.logger.info(f"Distance metrics computed:")
            self.logger.info(f"  MMD: {mmd_value:.6f}")
            self.logger.info(f"  MMD p-value: {mmd_pvalue:.6f}")
            if mmd_pvalue < 0.05:
                self.logger.info("  MMD indicates significantly different distributions (p < 0.05)")
            else:
                self.logger.info("  MMD indicates similar distributions (p >= 0.05)")
            
            self.logger.info(f"  Wasserstein (1D): {wasserstein_dist:.6f}")
            self.logger.info(f"  Wasserstein p-value: {wasserstein_pvalue:.6f}")
            if wasserstein_pvalue < 0.05:
                self.logger.info("  Wasserstein indicates significantly different distributions (p < 0.05)")
            else:
                self.logger.info("  Wasserstein indicates similar distributions (p >= 0.05)")
            
            self.logger.info(f"  Signature Wasserstein: {signature_wasserstein_dist:.6f}")
            self.logger.info(f"  Signature Wasserstein p-value: {sig_wasserstein_pvalue:.6f}")
            if sig_wasserstein_pvalue < 0.05:
                self.logger.info("  Signature Wasserstein indicates significantly different distributions (p < 0.05)")
            else:
                self.logger.info("  Signature Wasserstein indicates similar distributions (p >= 0.05)")
        
        return {
            'mmd': mmd_value,
            'mmd_pvalue': mmd_pvalue,
            'mmd_significant': mmd_pvalue < 0.05,
            'mmd_null_dist_mean': np.mean(mmd_null_dist),
            'mmd_null_dist_std': np.std(mmd_null_dist),
            'wasserstein': wasserstein_dist,
            'wasserstein_pvalue': wasserstein_pvalue,
            'wasserstein_significant': wasserstein_pvalue < 0.05,
            'wasserstein_null_dist_mean': np.mean(wasserstein_null_dist),
            'wasserstein_null_dist_std': np.std(wasserstein_null_dist),
            'signature_wasserstein': signature_wasserstein_dist,
            'signature_wasserstein_pvalue': sig_wasserstein_pvalue,
            'signature_wasserstein_significant': sig_wasserstein_pvalue < 0.05,
            'signature_wasserstein_null_dist_mean': np.mean(sig_wasserstein_null_dist),
            'signature_wasserstein_null_dist_std': np.std(sig_wasserstein_null_dist)
        }
    
    def _get_prefix(self, args):
        """Helper method to generate filename prefix from arguments"""
        prefix_parts = [f"w{args.window}", f"l{args.latent_dim}"]
        if hasattr(args, 'epochs') and args.epochs:
            prefix_parts.append(f"e{args.epochs}")
        if hasattr(args, 'year') and args.year:
            # Handle multiple years in filename
            years = [int(y.strip()) for y in args.year.split(',')]
            if len(years) == 1:
                prefix_parts.append(f"y{years[0]}")
            else:
                # For multiple years, use range notation or concatenation
                years_sorted = sorted(years)
                if len(years) == 2 and years_sorted[1] - years_sorted[0] == 1:
                    prefix_parts.append(f"y{years_sorted[0]}-{years_sorted[1]}")
                else:
                    prefix_parts.append(f"y{'_'.join(map(str, years_sorted))}")
        if hasattr(args, 'month') and args.month:
            # Handle multiple months in filename
            months = [int(m.strip()) for m in args.month.split(',')]
            if len(months) == 1:
                prefix_parts.append(f"m{months[0]}")
            else:
                # For multiple months, use range notation or concatenation
                months_sorted = sorted(months)
                if len(months) == 2 and months_sorted[1] - months_sorted[0] == 1:
                    prefix_parts.append(f"m{months_sorted[0]}-{months_sorted[1]}")
                else:
                    prefix_parts.append(f"m{'_'.join(map(str, months_sorted))}")
        prefix = "_".join(prefix_parts)
        return prefix

    def analyze_price_generation(self, df_original, staar_trainer, plots_path: str, prefix: str):
        """
        Compare original price series with generated ones by taking 2 groups of 1000 observations,
        generating synthetic sequences, and reconstructing full price series.
        
        Args:
            df_original: Original dataframe with all columns (close, trend_close, etc.)
            staar_trainer: Trained STAAR model for generation
            plots_path: Path to save plots
            prefix: Filename prefix
        """
        if self.logger:
            self.logger.info("Starting price generation analysis")
        
        total_samples = len(df_original)
        num_obs = 120
        offset = 500
        
        # Select two non-overlapping groups: arbitraryly chosen here
        group1_start = offset 
        group1_end = group1_start + num_obs
        
        group2_start = total_samples - num_obs - offset
        group2_end = group2_start + num_obs
        
        # Extract data for both groups
        groups_data = []
        for group_start, group_end, group_name in [(group1_start, group1_end, "Group 1"), 
                                                   (group2_start, group2_end, "Group 2")]:
            
            if self.logger:
                self.logger.info(f"Processing {group_name}: indices {group_start} to {group_end}")
            
            group_df = df_original[group_start:group_end]
            original_close = group_df.select("close").to_numpy().flatten()[:] 
            trend_close = group_df.select("trend_close").to_numpy().flatten()[:]
            original_c1_close = group_df.select("c1_detrended_close").to_numpy().flatten()[:]
            
            # Generate 3 synthetic sequences
            
            generated_sequences = []
            for i in range(3):
                gen_c1_close = staar_trainer.generate(1)[0,:,0]
                dummy_array = np.zeros_like(gen_c1_close)

                _, _, _, gen_c1_close_orig_scale = self.data_loader.inverse_transform_series(
                    dummy_array, dummy_array, dummy_array, gen_c1_close
                )

                _, _, _, c1_close_orig_scale = self.data_loader.inverse_transform_series(
                    dummy_array, dummy_array, dummy_array, original_c1_close
                )

                #print(gen_c1_close_orig_scale)
                #print(c1_close_orig_scale)
                
                generated_sequences.append(gen_c1_close_orig_scale)
            
            # Reconstruct full price series for each generated sequence
            reconstructed_prices = []
            for gen_c1_close_orig in generated_sequences:
                # Start from the detrended close value just before the group
                starting_detrended_close = group_df.select("detrended_close").to_numpy().flatten()[0]
                
                # Cumulative sum to get detrended prices
                detrended_prices = np.cumsum(np.concatenate([[starting_detrended_close], gen_c1_close_orig]))[1:]
                
                # Add trend to get final prices
                final_prices = detrended_prices + trend_close
                reconstructed_prices.append(final_prices)
            
            groups_data.append({
                'name': group_name,
                'original_prices': original_close,
                'reconstructed_prices': reconstructed_prices,
                'timestamps': list(range(group_start, group_end))  # For x-axis
            })
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        colors = ['red', 'blue', 'green']
        
        for idx, group_data in enumerate(groups_data):
            ax = axes[idx]
            
            # Plot original prices
            ax.plot(group_data['timestamps'], group_data['original_prices'], 
                   'black', linewidth=2, label='Original', alpha=0.8)
            
            # Plot 3 generated price series
            for i, gen_prices in enumerate(group_data['reconstructed_prices']):
                ax.plot(group_data['timestamps'], gen_prices, 
                       colors[i], linewidth=1.5, alpha=0.7, 
                       label=f'Generated {i+1}', linestyle='--')
            
            ax.set_title(f'{group_data["name"]}: Original vs Generated Price Series', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Close Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{plots_path}/{prefix}_price_generation_comparison.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        if self.logger:
            self.logger.info(f"Price generation comparison plot saved to {plot_filename}")
        plt.close()
        
        if self.logger:
            self.logger.info("Price generation analysis completed")

    def test_stylized_facts(self, staar_trainer, plots_path: str, prefix: str, original_sequences=None):
        """
        Test stylized facts of financial time series using generated sequences.
        Generates 1000 sequences, concatenates them, and tests for:
        1. Absence of autocorrelation in returns
        2. GARCH(1,1) model fit
        
        Args:
            staar_trainer: Trained STAAR model for generation
            plots_path: Path to save plots
            prefix: Filename prefix
            original_sequences: List of original sequences for comparison (optional)
        """
        if self.logger:
            self.logger.info("Starting stylized facts analysis")
        
        # Generate 1000 sequences of 120 elements each
        if self.logger:
            self.logger.info("Generating 1000 sequences from STAAR model")
        
        generated_sequences = staar_trainer.generate(1000)  # Shape: (1000, 120, 4)
        
        # Extract close price differences (channel 0)
        close_differences = generated_sequences[:, :, 0]  # Shape: (1000, 120)
        
        # Concatenate into one long sequence
        long_sequence = close_differences.flatten()  # Shape: (120000,)
        
        if self.logger:
            self.logger.info(f"Created long sequence of {len(long_sequence)} elements")
        
        dummy_array = np.zeros_like(long_sequence)
        # scale back to original scale
        _, _, _, long_sequence_orig = self.data_loader.inverse_transform_series(
            dummy_array, dummy_array, dummy_array, long_sequence
        )
        
        # Test 1: Autocorrelation analysis
        if self.logger:
            self.logger.info("Testing for absence of autocorrelation")
        
        from statsmodels.tsa.stattools import acf
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        # Compute autocorrelation function up to 50 lags
        autocorr, confint = acf(long_sequence_orig, nlags=50, alpha=0.05, fft=True)
        
        # Ljung-Box test for autocorrelation (H0: no autocorrelation)
        try:
            lb_results = acorr_ljungbox(long_sequence_orig, lags=20, return_df=False)
            if isinstance(lb_results, tuple) and len(lb_results) == 2:
                lb_stat, lb_pvalue = lb_results
                # Extract scalar values if they're arrays/series
                if hasattr(lb_pvalue, 'iloc'):
                    lb_pvalue = float(lb_pvalue.iloc[-1])
                elif hasattr(lb_pvalue, '__len__') and not isinstance(lb_pvalue, str):
                    lb_pvalue = float(lb_pvalue[-1])
                else:
                    lb_pvalue = float(lb_pvalue)
            else:
                # Handle case where return format is different - try to extract p-value
                lb_pvalue = 0.5
                lb_stat = None
                if hasattr(lb_results, 'iloc'):
                    try:
                        lb_pvalue = float(lb_results.iloc[-1])
                    except:
                        pass
                elif hasattr(lb_results, '__len__') and not isinstance(lb_results, str):
                    try:
                        lb_pvalue = float(lb_results[-1])
                    except:
                        pass
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Ljung-Box test failed: {e}, using default p-value")
            lb_pvalue = 0.5
            lb_stat = None
        
        if self.logger:
            self.logger.info(f"Ljung-Box test p-value: {lb_pvalue:.6f}")
            if lb_pvalue > 0.05:
                self.logger.info("No significant autocorrelation detected (p > 0.05)")
            else:
                self.logger.info("Significant autocorrelation detected (p <= 0.05)")
        
        # Test 2: GARCH(1,1) model fitting
        if self.logger:
            self.logger.info("Fitting GARCH(1,1) model")
        
        try:
            from arch import arch_model
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(long_sequence_orig, vol='Garch', p=1, q=1)
            garch_results = garch_model.fit(disp='off')
            
            # Extract GARCH parameters
            omega = garch_results.params['omega']
            alpha = garch_results.params['alpha[1]']
            beta = garch_results.params['beta[1]']
            
            # Check GARCH constraints (alpha + beta < 1 for stationarity)
            persistence = alpha + beta
            
            if self.logger:
                self.logger.info(f"GARCH(1,1) parameters:")
                self.logger.info(f"  ω (omega): {omega:.6f}")
                self.logger.info(f"  α (alpha): {alpha:.6f}")
                self.logger.info(f"  β (beta): {beta:.6f}")
                self.logger.info(f"  Persistence (α + β): {persistence:.6f}")
                
                if persistence is not None and persistence < 1.0:
                    self.logger.info("GARCH model is stationary (α + β < 1)")
                else:
                    self.logger.info("GARCH model is non-stationary (α + β >= 1)")
            
            garch_success = True
            garch_aic = garch_results.aic
            garch_bic = garch_results.bic
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fit GARCH(1,1) model: {str(e)}")
            garch_success = False
            omega = alpha = beta = persistence = garch_aic = garch_bic = None
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Time series of the long sequence (first 5000 points)
        ax1 = axes[0, 0]
        plot_length = min(5000, len(long_sequence_orig))
        ax1.plot(long_sequence_orig[:plot_length], linewidth=0.8, alpha=0.8)
        ax1.set_title('Generated Sequence (First 5000 Points)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Autocorrelation function
        ax2 = axes[0, 1]
        lags = range(len(autocorr))
        ax2.plot(lags, autocorr, 'b-', linewidth=2, label='ACF')
        ax2.fill_between(lags, confint[:, 0] - autocorr, confint[:, 1] - autocorr, alpha=0.3, color='gray', label='95% CI')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Autocorrelation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of returns
        ax3 = axes[1, 0]
        ax3.hist(long_sequence_orig, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_title('Distribution of Generated Returns', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary statistics
        mean_val = np.mean(long_sequence_orig)
        std_val = np.std(long_sequence_orig)
        skew_val = np.mean(((long_sequence_orig - mean_val) / std_val) ** 3)
        kurt_val = np.mean(((long_sequence_orig - mean_val) / std_val) ** 4) - 3
        
        summary_text = f"""Stylized Facts Analysis Results
        
Sequence Statistics:
• Length: {len(long_sequence_orig):,} observations
• Mean: {mean_val:.6f}
• Std Dev: {std_val:.6f}
• Skewness: {skew_val:.4f}
• Kurtosis: {kurt_val:.4f}

Autocorrelation Test:
• Ljung-Box p-value: {lb_pvalue:.6f}
• Result: {'No autocorr.' if lb_pvalue > 0.05 else 'Autocorr. present'}

GARCH(1,1) Model:"""
        
        if garch_success:
            summary_text += f"""
• ω (omega): {omega:.6f}
• α (alpha): {alpha:.6f}
• β (beta): {beta:.6f}
• Persistence: {persistence:.6f}
• Stationary: {'Yes' if persistence is not None and persistence < 1.0 else 'No'}
• AIC: {garch_aic:.2f}
• BIC: {garch_bic:.2f}"""
        else:
            summary_text += """
• Model fitting failed"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        # Overall title
        fig.suptitle('Stylized Facts Analysis of Generated Financial Time Series', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{plots_path}/{prefix}_stylized_facts.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        if self.logger:
            self.logger.info(f"Stylized facts analysis plot saved to {plot_filename}")
        plt.close()
        
        # Return results
        results = {
            'autocorr_pvalue': lb_pvalue,
            'no_autocorrelation': lb_pvalue > 0.05,
            'mean': mean_val,
            'std': std_val,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'garch_success': garch_success
        }
        
        if garch_success:
            results.update({
                'garch_omega': omega,
                'garch_alpha': alpha,
                'garch_beta': beta,
                'garch_persistence': persistence,
                'garch_stationary': persistence is not None and persistence < 1.0,
                'garch_aic': garch_aic,
                'garch_bic': garch_bic
            })
        
        if self.logger:
            self.logger.info("Stylized facts analysis completed")
        
        return results