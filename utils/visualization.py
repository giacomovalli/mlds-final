import matplotlib.pyplot as plt
import os


def plot_training_losses(training_losses, output_folder, filename_prefix, logger=None):
    """
    Plot and save comprehensive training losses for STAAR model training.
    
    Args:
        training_losses (dict): Dictionary containing loss histories with keys like:
            - 'elbo_loss': ELBO loss
            - 'reconstruction_loss': Reconstruction loss
            - 'kl_loss': KL divergence loss
            - 'latent_discriminator_loss': Latent discriminator loss
            - 'generator_loss': Generator loss
            - 'stats_discriminator_loss': Stats discriminator loss
            - 'decoder_adversarial_loss': Decoder adversarial loss
            - 'autocorrelation_loss': Autocorrelation loss
            - 'garch_loss': GARCH loss
            - 'val_*': Validation losses
        output_folder (str): Directory where to save the plot files
        filename_prefix (str): Prefix for the saved plot filenames
        logger: Optional logger instance for logging messages
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find the number of epochs from any non-empty loss list
    epochs_range = None
    for key, values in training_losses.items():
        if values and len(values) > 0:
            epochs_range = range(1, len(values) + 1)
            break
    
    if epochs_range is None:
        if logger:
            logger.warning("No training loss data found to plot")
        return
    
    # Define the plots we want to create
    plot_configs = [
        # Main losses subplot
        {
            'title': 'Main Training Losses',
            'losses': [
                ('elbo_loss', 'ELBO Loss', 'blue'),
                ('reconstruction_loss', 'Reconstruction', 'green'),
                ('kl_loss', 'KL Divergence', 'orange')
            ]
        },
        # Adversarial losses subplot
        {
            'title': 'Adversarial Losses',
            'losses': [
                ('latent_discriminator_loss', 'Latent Discriminator', 'red'),
                ('generator_loss', 'Generator', 'purple'),
                ('stats_discriminator_loss', 'Stats Discriminator', 'brown'),
                ('decoder_adversarial_loss', 'Decoder Adversarial', 'pink')
            ]
        },
        # Stylized facts losses subplot
        {
            'title': 'Stylized Facts Losses',
            'losses': [
                ('autocorrelation_loss', 'Autocorrelation', 'cyan'),
                ('garch_loss', 'GARCH', 'magenta')
            ]
        },
        # Validation losses subplot
        {
            'title': 'Validation Losses',
            'losses': [
                ('val_elbo_loss', 'Val ELBO', 'blue', '--'),
                ('val_reconstruction_loss', 'Val Reconstruction', 'green', '--'),
                ('val_kl_loss', 'Val KL', 'orange', '--'),
                ('val_mse_loss', 'Val MSE', 'red', '--')
            ]
        }
    ]
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        ax.set_title(config['title'], fontsize=14, fontweight='bold')
        
        plotted_any = False
        for loss_config in config['losses']:
            loss_key = loss_config[0]
            label = loss_config[1] 
            color = loss_config[2]
            linestyle = loss_config[3] if len(loss_config) > 3 else '-'
            
            if loss_key in training_losses and training_losses[loss_key]:
                ax.plot(epochs_range, training_losses[loss_key], 
                       color=color, linestyle=linestyle, linewidth=2, label=label)
                plotted_any = True
        
        if plotted_any:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_filename = os.path.join(output_folder, f"{filename_prefix}_comprehensive_losses.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate plot comparing training vs validation losses
    val_losses = {k: v for k, v in training_losses.items() if k.startswith('val_') and v}
    if val_losses:
        plt.figure(figsize=(15, 5))
        
        # ELBO comparison
        plt.subplot(1, 3, 1)
        if training_losses.get('elbo_loss'):
            plt.plot(epochs_range, training_losses['elbo_loss'], 'b-', linewidth=2, label='Training ELBO')
        if training_losses.get('val_elbo_loss'):
            plt.plot(epochs_range, training_losses['val_elbo_loss'], 'b--', linewidth=2, label='Validation ELBO')
        plt.title('ELBO Loss: Training vs Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Reconstruction comparison
        plt.subplot(1, 3, 2)
        if training_losses.get('reconstruction_loss'):
            plt.plot(epochs_range, training_losses['reconstruction_loss'], 'g-', linewidth=2, label='Training Reconstruction')
        if training_losses.get('val_reconstruction_loss'):
            plt.plot(epochs_range, training_losses['val_reconstruction_loss'], 'g--', linewidth=2, label='Validation Reconstruction')
        plt.title('Reconstruction Loss: Training vs Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # KL comparison
        plt.subplot(1, 3, 3)
        if training_losses.get('kl_loss'):
            plt.plot(epochs_range, training_losses['kl_loss'], 'orange', linewidth=2, label='Training KL')
        if training_losses.get('val_kl_loss'):
            plt.plot(epochs_range, training_losses['val_kl_loss'], 'orange', linestyle='--', linewidth=2, label='Validation KL')
        plt.title('KL Divergence: Training vs Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the validation comparison plot
        val_plot_filename = os.path.join(output_folder, f"{filename_prefix}_training_vs_validation.png")
        plt.savefig(val_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"Training vs validation plot saved to: {val_plot_filename}")
    
    if logger:
        logger.info(f"Comprehensive training loss plot saved to: {plot_filename}")