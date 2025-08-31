from data.loader import BatchedTimeseriesSequence
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class StaarTester:
    
    def __init__(self, dataset:BatchedTimeseriesSequence, 
                staar_trainer,
                window_size: int, n_features: int = 4, 
                 lstm_units_1: int = 64, lstm_units_2: int = 32,
                 dense_units: int = 16, dropout_rate: float = 0.2, logger=None):
        """
        Initialize StaarTester with a classification network.
        
        Args:
            window_size: Length of input sequences (time steps)
            n_features: Number of features per time step (default 4 for OHLC data)
            lstm_units_1: Number of units in first LSTM layer
            lstm_units_2: Number of units in second LSTM layer  
            dense_units: Number of units in the dense layer before output
            dropout_rate: Dropout rate for regularization
        """
        self.model = self.create_classification_network(
            window_size=window_size,
            n_features=n_features,
            lstm_units_1=lstm_units_1,
            lstm_units_2=lstm_units_2,
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )
        self.dataset = dataset
        self.staar_trainer = staar_trainer
        self.logger = logger
        logger.info(self.model.summary())
        self.X, self.y = self._prepare_data()

    
    def create_classification_network(self, window_size: int, n_features: int = 4, 
                                    lstm_units_1: int = 64, lstm_units_2: int = 32,
                                    dense_units: int = 16, dropout_rate: float = 0.2):
        """
        Create a classification network
        
        Args:
            window_size: Length of input sequences (time steps)
            n_features: Number of features per time step (default 4 for OHLC data)
            lstm_units_1: Number of units in first LSTM layer
            lstm_units_2: Number of units in second LSTM layer  
            dense_units: Number of units in the dense layer before output
            dropout_rate: Dropout rate for regularization
            
        Returns:
            keras.Model: Compiled classification model
        """
        # Input layer - expecting sequences from BatchedTimeseriesSequence
        # Shape: (batch_size, window_size, n_features) - already in correct format for LSTM
        inputs = keras.Input(shape=(window_size, n_features), name='sequence_input')
        
        # Input is already in the correct shape for LSTM layers
        x = inputs
        
        # First LSTM layer - return sequences for the second LSTM
        x = layers.LSTM(
            units=lstm_units_1,
            return_sequences=True,
        #    dropout=dropout_rate,
        #    recurrent_dropout=dropout_rate,
            name='lstm_1'
        )(x)
        
        # Layer normalization after first LSTM
        x = layers.LayerNormalization(name='layer_norm_1')(x)
        
        # Second LSTM layer - return only the final output
        x = layers.LSTM(
            units=lstm_units_2,
            return_sequences=False,
            #dropout=dropout_rate,
            #recurrent_dropout=dropout_rate,
            name='lstm_2'
        )(x)
        
        # Layer normalization after second LSTM
        x = layers.LayerNormalization(name='layer_norm_2')(x)
        
        # Dense layer
        x = layers.Dense(
            units=dense_units,
            activation='relu',
            name='dense_hidden'
        )(x)
        
        # Dropout for regularization
        x = layers.Dropout(dropout_rate, name='dropout')(x)
        
        # Final output layer with sigmoid activation for binary classification
        outputs = layers.Dense(
            units=1,
            activation='sigmoid',
            name='classification_output'
        )(x)
        
        # Create and compile the model
        model = keras.Model(inputs=inputs, outputs=outputs, name='staar_classifier')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                'precision', 
                'recall',
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.FalseNegatives(name='fn')
            ]
        )
        
        return model
    
    def _prepare_data(self):
        """
        Pull 50,000 sequences from self.dataset and generate 50,000 from staar_model.
        Creates a combined dataset for training the discriminator.
        
        Returns:
            tuple: (X, y) where X is array of sequences and y is array of labels 
                   (1s for real data, 0s for generated data)
        """
        target_sequences = 100000
        
        # 1. Collect real sequences from dataset
        real_sequences = []
        
        # Iterate through the dataset once
        for batch_idx in range(len(self.dataset)):
            # Get batch from dataset
            batch_x, _ = self.dataset[batch_idx]  # batch_x shape: (batch_size, window, features)
            
            # Add all sequences from this batch
            for i in range(batch_x.shape[0]):
                real_sequences.append(batch_x[i])
                
                # Stop if we've collected enough sequences
                if len(real_sequences) >= target_sequences:
                    break
            
            # Stop if we've collected enough sequences
            if len(real_sequences) >= target_sequences:
                break
        
        # Take only the first 50,000 sequences (or all if dataset is smaller)
        real_sequences = real_sequences[:target_sequences]
        
        # 2. Generate fake sequences from STAAR model
        generated_sequences = self.staar_trainer.generate(target_sequences)
        self.logger.info(f"Generated {len(generated_sequences)} sequences from STAAR model")
        
        # 3. Combine real and generated sequences
        X_real = np.array(real_sequences)  # Shape: (n_real, window, features)
        X_generated = np.array(generated_sequences)  # Shape: (50000, window, features)
        
        # Combine all sequences
        X = np.concatenate([X_real, X_generated], axis=0)
        
        # 4. Create labels: 1s for real data, 0s for generated data
        y_real = np.ones(len(real_sequences), dtype=np.float32)
        y_generated = np.zeros(len(generated_sequences), dtype=np.float32)
        
        # Combine labels
        y = np.concatenate([y_real, y_generated])
        
        return X, y
    
    def train(self, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2, 
              test_split: float = 0.2, shuffle: bool = True, verbose: int = 1):
        """
        Train the classification network using prepared data.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation (from remaining after test split)
            test_split: Fraction of data to use for testing
            shuffle: Whether to shuffle data before splitting
            verbose: Verbosity level for training
            
        Returns:
            dict: Training history and test results
        """
        # Shuffle data if requested
        if shuffle:
            indices = np.random.permutation(len(self.X))
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
        else:
            X_shuffled = self.X
            y_shuffled = self.y
        
        # Calculate split indices
        n_samples = len(X_shuffled)
        test_size = int(n_samples * test_split)
        remaining_size = n_samples - test_size
        val_size = int(remaining_size * validation_split)
        train_size = remaining_size - val_size
        
        # Split data: train/val/test
        X_train = X_shuffled[:train_size]
        y_train = y_shuffled[:train_size]
        
        X_val = X_shuffled[train_size:train_size + val_size]
        y_val = y_shuffled[train_size:train_size + val_size]
        
        X_test = X_shuffled[train_size + val_size:]
        y_test = y_shuffled[train_size + val_size:]
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Train labels - Real: {np.sum(y_train):.0f}, Generated: {len(y_train) - np.sum(y_train):.0f}")
        
        # Create TensorFlow datasets with optimized pipeline
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size)
        
        # Set up callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        print("Starting training...")
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = self.model.evaluate(test_dataset, verbose=0)
        
        # Debug: Get actual predictions to see what the model is predicting
        print("\nGetting sample predictions...")
        sample_predictions = self.model.predict(test_dataset.take(1), verbose=0)
        print(f"Sample predictions shape: {sample_predictions.shape}")
        print(f"Sample predictions (first 10): {sample_predictions[:10].flatten()}")
        print(f"Sample predictions stats - min: {sample_predictions.min():.4f}, max: {sample_predictions.max():.4f}, mean: {sample_predictions.mean():.4f}")

        results = {
            'loss': test_results[0],
            'accuracy': test_results[1],
            'precision': test_results[2],
            'recall': test_results[3],
            'tp': test_results[4],
            'tn': test_results[5],
            'fp': test_results[6],
            'fn': test_results[7]
        }

        self.logger.info(results)
        return {'history': history, 'test_results': results}
    
    def visualize_results(self, training_results, save_path="plots/discriminator_results.png"):
        """
        Create visualizations of training history and test results.
        
        Args:
            training_results: Dictionary containing 'history' and 'test_results'
            save_path: Path to save the visualization
        """
        history = training_results['history']
        test_results = training_results['test_results']
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots - simplified layout
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. TP/TN/FP/FN progression through training
        ax1 = fig.add_subplot(gs[0, :])
        epochs = range(1, len(history.history['tp']) + 1)
        ax1.plot(epochs, history.history['tp'], 'g-', label='True Positives', linewidth=2)
        ax1.plot(epochs, history.history['tn'], 'b-', label='True Negatives', linewidth=2)
        ax1.plot(epochs, history.history['fp'], 'r-', label='False Positives', linewidth=2)
        ax1.plot(epochs, history.history['fn'], 'orange', label='False Negatives', linewidth=2)
        ax1.set_title('Confusion Matrix Components During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[1, 0])
        tp, tn, fp, fn = test_results['tp'], test_results['tn'], test_results['fp'], test_results['fn']
        # Convert to integers to avoid formatting issues
        confusion_matrix = np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted: Generated', 'Predicted: Real'],
                   yticklabels=['Actual: Generated', 'Actual: Real'],
                   ax=ax2, cbar_kws={'label': 'Count'})
        ax2.set_title('Final Test Confusion Matrix', fontsize=14, fontweight='bold')
        
        # 3. Generation Quality Assessment
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        # Calculate additional metrics (convert to int for calculations)
        tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
        f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Quality assessment
        accuracy = test_results['accuracy']
        
        # Create text summary
        summary_text = f"""Results Summary
        
Test Metrics:
• Accuracy: {accuracy:.4f}
• Precision: {test_results['precision']:.4f}
• Recall: {test_results['recall']:.4f}
• F1-Score: {f1_score:.4f}
• Specificity: {specificity:.4f}

Confusion Matrix:
• True Positives: {tp:.0f}
• True Negatives: {tn:.0f}
• False Positives: {fp:.0f}
• False Negatives: {fn:.0f}
        """
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        # Overall title
        fig.suptitle('STAAR data discriminator Results', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.logger:
            self.logger.info(f"Results visualization saved to {save_path}")
        
        plt.show()
        plt.close()
    