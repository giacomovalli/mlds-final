import tensorflow as tf
import keras as keras


class ModelCheckpointCallback(keras.callbacks.Callback):
    """
    Custom callback to save the STAAR model every N epochs.
    
    This callback saves the model at regular intervals to prevent loss of progress
    during long training runs.
    
    Args:
        staar_model: The STAAR model instance to save
        save_path (str): Directory path where models will be saved
        model_prefix (str): Prefix for saved model files
        save_frequency (int): Save model every N epochs (default: 10)
        logger: Logger instance for logging save messages
    """
    
    def __init__(self, staar_model, save_path="saved_models", model_prefix="checkpoint", save_frequency=10, logger=None):
        super().__init__()
        self.staar_model = staar_model
        self.save_path = save_path
        self.model_prefix = model_prefix
        self.save_frequency = save_frequency
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to check if model should be saved.
        
        Args:
            epoch: Integer index of the epoch
            logs: Dictionary containing metrics results for this epoch
        """
        # Save model every save_frequency epochs
        if (epoch + 1) % self.save_frequency == 0:
            checkpoint_prefix = f"{self.model_prefix}_{epoch + 1}"
            try:
                self.staar_model.save_model(self.save_path, checkpoint_prefix)
                if self.logger:
                    self.logger.info(f"Model saved successfully as {checkpoint_prefix}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to save model at epoch {epoch + 1}: {str(e)}")



class EarlyStoppingCallback(keras.callbacks.Callback):
    """
    Custom early stopping callback that monitors both training and validation loss.
    
    Stops training when neither training loss nor validation loss improve for a given
    number of epochs (patience). This prevents overfitting and saves computational resources.
    
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped
        min_delta (float): Minimum change in monitored quantity to qualify as improvement
        logger: Logger instance for logging messages
    """
    
    def __init__(self, patience=5, min_delta=1e-4, logger=None):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger
        
        # Track best losses and patience counters
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to check for improvement.
        
        Args:
            epoch: Integer index of the epoch
            logs: Dictionary containing metrics results for this epoch
        """
        if logs is None:
            return
            
        current_train_loss = logs.get('loss')
        current_val_loss = logs.get('val_loss')
        
        if current_train_loss is None or current_val_loss is None:
            return
            
        # Check for improvement in either training or validation loss
        train_improved = current_train_loss < (self.best_train_loss - self.min_delta)
        val_improved = current_val_loss < (self.best_val_loss - self.min_delta)
        
        if train_improved or val_improved:
            # At least one loss improved
            if train_improved:
                self.best_train_loss = current_train_loss
            if val_improved:
                self.best_val_loss = current_val_loss
            self.patience_counter = 0
        else:
            # No improvement in either loss
            self.patience_counter += 1
            if self.logger:
                self.logger.info(f"Epoch {epoch + 1}: No improvement for {self.patience_counter}/{self.patience} epochs")
            
            if self.patience_counter >= self.patience:
                self.stopped_epoch = epoch + 1
                if self.logger:
                    self.logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}")
                    self.logger.info(f"Best train loss: {self.best_train_loss:.6f}, Best val loss: {self.best_val_loss:.6f}")
                self.model.stop_training = True
                
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.stopped_epoch > 0 and self.logger:
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch} due to lack of improvement")


class NaNValidationCallback(keras.callbacks.Callback):
    """
    Custom callback to stop training when validation loss becomes NaN or Inf.
    
    This callback monitors the validation loss after each epoch and stops training
    immediately if the validation loss becomes NaN or infinite, preventing wasted
    computational resources on unstable training runs.
    
    Args:
        logger: Logger instance for logging error messages
    """
    
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to check validation loss.
        
        Args:
            epoch: Integer index of the epoch
            logs: Dictionary containing metrics results for this epoch
        """
        if logs is None:
            return
        
        # Check for NaN or Inf in validation loss
        val_loss = logs.get('val_loss')
        if val_loss is not None and (tf.math.is_nan(val_loss) or tf.math.is_inf(val_loss)):
            self.logger.error(f"Validation loss became NaN or Inf at epoch {epoch + 1}. Stopping training.")
            self.model.stop_training = True


class NegativeReconstructionStopCallback(keras.callbacks.Callback):
    """
    Custom callback to stop training when reconstruction loss is negative for consecutive epochs.
    
    This callback monitors the reconstruction loss and stops training if it remains negative
    for a specified number of consecutive epochs, indicating potential model instability.
    
    Args:
        consecutive_epochs (int): Number of consecutive epochs with negative reconstruction loss 
                                to trigger early stopping (default: 3)
        logger: Logger instance for logging messages
    """
    
    def __init__(self, consecutive_epochs=3, logger=None):
        super().__init__()
        self.consecutive_epochs = consecutive_epochs
        self.logger = logger
        self.negative_count = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to check reconstruction loss.
        
        Args:
            epoch: Integer index of the epoch
            logs: Dictionary containing metrics results for this epoch
        """
        if logs is None:
            return
            
        # Look for reconstruction loss in logs (could be named differently)
        recon_loss = logs.get('recon') or logs.get('val_recon')
        
        if recon_loss is not None:
            if recon_loss < 0:
                self.negative_count += 1
                
                if self.negative_count >= self.consecutive_epochs:
                    self.stopped_epoch = epoch + 1
                    if self.logger:
                        self.logger.error(f"Stopping training: Reconstruction loss has been negative for "
                                        f"{self.consecutive_epochs} consecutive epochs (current: {recon_loss:.6f})")
                    self.model.stop_training = True
            else:
                self.negative_count = 0
                
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.stopped_epoch > 0 and self.logger:
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch} due to persistent negative reconstruction loss")

class KLAnnealingCallback(tf.keras.callbacks.Callback):
    """Callback to update the epoch counter for KL annealing"""
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
    
    def on_epoch_begin(self, epoch, logs=None):
        self.trainer.set_epoch(epoch)