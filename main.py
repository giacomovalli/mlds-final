from data.loader import DataLoader, BatchedTimeseriesSequence
from utils.logger import setup_logger
from utils.visualization import plot_training_losses
from stylized_facts.autocorrelation import Autocorrelation
from decompose.preprocess import check_stationarity, wavelet_decompose, plot_periodogram
import polars as pl
import matplotlib.pyplot as plt
from aae.analyzer import StaarResultAnalyzer
import argparse
import tensorflow as tf
import keras
from data.distribution import fit_kde_distribution
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from stylized_facts.gain_loss_asymmetry import GainLossAsymmetry
from stylized_facts.volatility_clustering import VolatilityClustering
from stylized_facts.volume_vol_corr import VolumeVolatilityCorrelation
from aae.staar import StaarModel
from aae.staar_trainer import StaarModelTrainer, create_distributed_trainer
from aae.staar_tester import StaarTester
from aae.callbacks import NaNValidationCallback, EarlyStoppingCallback, NegativeReconstructionStopCallback, ModelCheckpointCallback, KLAnnealingCallback

saved_models_path = "saved_models/"
training_plots_path = "plots/"
dataset_path = "dataset/ES_full_1min_continuous_absolute_adjusted.txt"

AE_LEARNING_RATE = 2e-4
LAT_DISC_LEARNING_RATE = 0.5e-4
GEN_LEARNING_RATE = 0.75e-4
STAT_DISC_LEARNING_RATE = 5e-5
DEC_ADV_LEARNING_RATE = 5e-5

FEATURES = 4
OUTPUT_FEATURES = 4
LSTM_UNITS = 32
NUM_HEADS = 4
KEY_DIM = 16
NUM_BLOCKS = 2
EXCLUDE_STATS_DISC = False  # Exclude stats discriminator by default
GRADIENT_CLIPPING_NORM = 1.0  # Enable gradient clipping to prevent NaN
KL_ANNEALING_EPOCHS=10
KL_MAX_WEIGHT=1
KL_MIN_WEIGHT=0
L1_K=10
GK_VOL_K=0
STD_LOSS_K=10

garch_params = {'omega': 0.0137, 'alpha': 0.2, 'beta': 0.78}
loss_weights = {'adv': 0.5, 'acl': 0.25, 'garch': 0.25}

def _get_training_strategy(logger):
    strategy = None
    if len(tf.config.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    elif len(tf.config.list_physical_devices('GPU')) == 0:
        # Multi-CPU strategy
        strategy = tf.distribute.experimental.CentralStorageStrategy()
        logger.info("Using CentralStorageStrategy for CPU training")
    else:
        logger.info("Using single device training")
    #return tf.distribute.MirroredStrategy()
    return strategy

def _get_prefix(args):
    # Create filename prefix from parameters
    prefix_parts = [f"w{args.window}", f"l{args.latent_dim}", f"e{args.epochs}"]
    if args.year:
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
    if args.month:
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

def summary_model(df_filtered, args, logger):
    """Display model summary - temporarily disabled due to TensorFlow compatibility issues"""
    model = StaarModel(
        logger,
        time_steps=args.window,
        features=FEATURES,
        output_features=OUTPUT_FEATURES,
        latent_dim=args.latent_dim,
        lstm_units=LSTM_UNITS,
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        num_blocks=NUM_BLOCKS
    )
    logger.info(model.summary())

def test_model(df_filtered, model, args, logger, loader):
    """Test the loaded AAE model using the StaarResultAnalyzer"""
    analyzer = StaarResultAnalyzer(logger=logger, show_plots=True, data_loader=loader)
    results = analyzer.analyze_model(df_filtered, model, args, training_plots_path)
    return results

def train_model(df_filtered, df_validation, args, logger, loaded_model=None):
    """Train the AAE model with the given parameters"""
    strategy = _get_training_strategy(logger)

    logger.info(f"total number of samples: {len(df_filtered)} and number of batchs: {len(df_filtered) // args.batch_size + 1}")
    logger.info(f"validation samples: {len(df_validation)} and number of validation batchs: {len(df_validation) // args.batch_size + 1}")
    
    data_loader = BatchedTimeseriesSequence(df_filtered,
        batch_size=args.batch_size, 
        window=args.window,
        logger=logger,
        shuffle=True)
    #train_dataset_factory = BatchedTimeseriesDataset(df_filtered, window_size=args.window, logger=logger)
    #train_dataset = train_dataset_factory.create_dataset()
    #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    #validation_dataset_factory = BatchedTimeseriesDataset(df_validation, window_size=args.window, logger=logger)
    #validation_dataset = validation_dataset_factory.create_dataset()
    #validation_dataset = validation_dataset.batch(args.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    validation_loader = BatchedTimeseriesSequence(df_validation,
        batch_size=args.batch_size, 
        window=args.window,
        logger=logger,
        shuffle=False)

    if strategy:
        with strategy.scope():
            model = StaarModel(
                logger,
                time_steps=args.window,
                features=FEATURES,
                output_features=OUTPUT_FEATURES,
                latent_dim=args.latent_dim,
                lstm_units=LSTM_UNITS,
                num_heads=NUM_HEADS,
                key_dim=KEY_DIM,
                num_blocks=NUM_BLOCKS
            )
            
            logger.info("Creating distributed trainer")
            trainer = create_distributed_trainer(
                staar_model=model,
                garch_params=garch_params,
                loss_weights=loss_weights,
                ae_opt_config={'class': keras.optimizers.Adam, 'kwargs': {'learning_rate': AE_LEARNING_RATE}},
                lat_disc_opt_config={'class': keras.optimizers.Adam, 'kwargs': {'learning_rate': LAT_DISC_LEARNING_RATE}},
                gen_opt_config={'class': keras.optimizers.Adam, 'kwargs': {'learning_rate': GEN_LEARNING_RATE}},
                stat_disc_opt_config={'class': keras.optimizers.Adam, 'kwargs': {'learning_rate': STAT_DISC_LEARNING_RATE}},
                dec_adv_opt_config={'class': keras.optimizers.Adam, 'kwargs': {'learning_rate': DEC_ADV_LEARNING_RATE}},
                exclude_stats_disc=EXCLUDE_STATS_DISC,
                clip_norm=GRADIENT_CLIPPING_NORM
            )
    else:
        if loaded_model:
            model = loaded_model
            logger.info("Using provided loaded model for training")
        else:
             model = StaarModel(
                logger,
                time_steps=args.window,
                features=FEATURES,
                output_features=OUTPUT_FEATURES,
                latent_dim=args.latent_dim,
                lstm_units=LSTM_UNITS,
                num_heads=NUM_HEADS,
                key_dim=KEY_DIM,
                num_blocks=NUM_BLOCKS
            )
        
        trainer = StaarModelTrainer(
            staar_model=model,
            garch_params=garch_params,
            loss_weights=loss_weights,
            exclude_stats_disc=EXCLUDE_STATS_DISC,
            clip_norm=GRADIENT_CLIPPING_NORM,
            kl_anneal_epochs=KL_ANNEALING_EPOCHS,
            kl_max_weight=KL_MAX_WEIGHT,
            kl_min_weight=KL_MIN_WEIGHT,
            l1_k=L1_K,
            gk_vol_k=GK_VOL_K,
            std_loss_k=STD_LOSS_K
        )
        
        logger.info("Compiling AAE model")
        trainer.compile(
            ae_opt=keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
            lat_disc_opt=keras.optimizers.Adam(learning_rate=LAT_DISC_LEARNING_RATE),
            gen_opt=keras.optimizers.Adam(learning_rate=GEN_LEARNING_RATE),
            stat_disc_opt=keras.optimizers.Adam(learning_rate=STAT_DISC_LEARNING_RATE),
            dec_adv_opt=keras.optimizers.Adam(learning_rate=DEC_ADV_LEARNING_RATE)
        )

    logger.info(model.summary())
    
    logger.info("Starting AAE training")
    tf_dataset = data_loader.to_tf_dataset()
    validation_dataset = validation_loader.to_tf_dataset()
    
    # Distribute datasets if using multi-device strategy
    if strategy and strategy.num_replicas_in_sync > 1:
        # Ensure dataset is evenly divisible across replicas to prevent OUT_OF_RANGE errors
        num_replicas = strategy.num_replicas_in_sync
        
        # Calculate steps per epoch for even distribution
        train_steps = len(data_loader) // num_replicas * num_replicas
        val_steps = len(validation_loader) // num_replicas * num_replicas
        
        # Take only the evenly divisible portion of the dataset
        tf_dataset = tf_dataset.take(train_steps)
        validation_dataset = validation_dataset.take(val_steps)
        
        logger.info(f"Adjusted dataset sizes: train={train_steps}, val={val_steps} for {num_replicas} GPUs")
        
        # Add prefetch and optimization for multi-GPU
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
        
        tf_dataset = strategy.experimental_distribute_dataset(tf_dataset)
        validation_dataset = strategy.experimental_distribute_dataset(validation_dataset)
        
        logger.info(f"Distributed datasets across {strategy.num_replicas_in_sync} GPUs")
    else:
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    
    nan_callback = NaNValidationCallback(logger)
    early_stop_callback = EarlyStoppingCallback(patience=10, logger=logger)
    negative_recon_callback = NegativeReconstructionStopCallback(consecutive_epochs=3, logger=logger)
    kl_callback = KLAnnealingCallback(trainer)
    checkpoint_callback = ModelCheckpointCallback(
        staar_model=model, 
        save_path=saved_models_path, 
        model_prefix=f"cp_{_get_prefix(args)}", 
        save_frequency=10, 
        logger=logger
    )
    history = trainer.fit(tf_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[nan_callback, early_stop_callback, checkpoint_callback, kl_callback])
    logger.info("AAE training completed")
    
    prefix = _get_prefix(args)
    model.save_model(saved_models_path, prefix)

    logger.info(f"Available history keys: {list(history.history.keys())}")
    
    # Extract and format training losses for plotting
    training_losses = {
        # Main training losses
        'elbo_loss': history.history.get('elbo', []),
        'reconstruction_loss': history.history.get('recon', []),
        'kl_loss': history.history.get('kl', []),
        'latent_discriminator_loss': history.history.get('lat_disc', []),
        'generator_loss': history.history.get('gen', []),
        'stats_discriminator_loss': history.history.get('stat_disc', []),
        'decoder_adversarial_loss': history.history.get('dec_adv', []),
        'autocorrelation_loss': history.history.get('acl', []),
        'garch_loss': history.history.get('garch', []),
        
        # Validation losses (keys must match what test_step returns)
        'val_elbo_loss': history.history.get('val_elbo', []),
        'val_reconstruction_loss': history.history.get('val_recon', []),
        'val_kl_loss': history.history.get('val_kl', []),
        'val_mse_loss': history.history.get('val_mse', [])
    }
    
    # Plot training losses
    plot_training_losses(training_losses, training_plots_path, prefix, logger)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Financial data analysis application')
    parser.add_argument('--train', action='store_true', help='Run model training')
    parser.add_argument('--summary', action='store_true', help='Model summary')
    parser.add_argument('--year', type=str, help='Filter data by year(s) - single year (e.g., 2023) or comma-separated list (e.g., 2022,2023,2024)')
    parser.add_argument('--month', type=str, help='Filter data by month(s) - single month (e.g., 6) or comma-separated list (e.g., 1,6,12)')
    parser.add_argument('--window', type=int, default=120, help='Window size for sequence data (default: 120)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training (default: 100)')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of latent space (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--kl-scaling-epochs', type=int, default=0, help='Number of epochs to apply KL scaling (0 disables)')
    parser.add_argument('--kl-scaling-factor', type=float, default=1.0, help='KL scaling factor to apply for the specified epochs (default: 1.0)')
    parser.add_argument('--load-model', type=str, help='Load a saved model using the given prefix (e.g., w120_l64_y2023)')
    parser.add_argument('--test', action='store_true', help='Test loaded model on data and plot reconstruction')
    parser.add_argument('--fit-kde', type=str, help='Fit KDE model to specified column and create plots')
    parser.add_argument('--wavelets', action='store_true', help='Peridogram and wavelet decomposition analysis of close price series')
    parser.add_argument('--stationarity', action='store_true', help='Check stationarity of close price series and its changes')
    parser.add_argument('--autocorrelation', action='store_true', help='Create autocorrelation analysis for close price changes')
    parser.add_argument('--gainloss', action='store_true', help='Plot gain/loss asymmetry for close price series')
    parser.add_argument('--volatility_clustering', action='store_true', help='Plot volatility clustering using GARCH model on close price changes')
    parser.add_argument('--volume_volatility_corr', action='store_true', help='Analyze volume-volatility correlation with different rolling windows')
    parser.add_argument('--test-generation', action='store_true', help='Test model generation quality using a discriminator (requires --load-model)')
    parser.add_argument('--ppp', action='store_true', help='for random tests (requires --load-model)')
    args = parser.parse_args()
    logger = setup_logger(__name__)
    logger.info("Starting financial data analysis application")

    loader = DataLoader(logger)
    df = loader.load_financial_data(dataset_path)

    logger.info("Dataset preview:")
    logger.info(df.head(10))

    df_filtered = df
    
    # Apply year filter if specified
    if args.year:
        # Parse comma-separated years
        years = [int(y.strip()) for y in args.year.split(',')]
        df_filtered = df_filtered.filter(pl.col("datetime").dt.year().is_in(years))
        logger.info(f"Filtered data by years: {', '.join(map(str, years))}")
    
    # Apply month filter if specified
    if args.month:
        # Parse comma-separated months
        months = [int(m.strip()) for m in args.month.split(',')]
        df_filtered = df_filtered.filter(pl.col("datetime").dt.month().is_in(months))
        logger.info(f"Filtered data by months: {', '.join(map(str, months))}")

    #df_filtered = df.filter(
    #    (pl.col("datetime").dt.month() <= 3) & 
    #    (pl.col("datetime").dt.year() == 2024)
    #)
    
    df_filtered = df_filtered.drop_nulls()
    logger.info(df_filtered.describe())
    
    df_validation = df.filter(
        (pl.col("datetime").dt.year() == 2024) & 
        (pl.col("datetime").dt.month() == 4)
    ).drop_nulls()
    
    # Load model if specified
    loaded_model = None
    if getattr(args, 'load_model', None):
        loaded_model = StaarModel(
            logger,
            time_steps=args.window,
            features=FEATURES,
            output_features=OUTPUT_FEATURES,
            latent_dim=args.latent_dim,
            lstm_units=LSTM_UNITS,
            num_heads=NUM_HEADS,
            key_dim=KEY_DIM,
            num_blocks=NUM_BLOCKS
        )
        loaded_model.load_model(saved_models_path, args.load_model)
        logger.info("Model loaded successfully")
    
    if args.summary:
        summary_model(df_filtered, args, logger)
    
    elif args.train:
        model = train_model(df_filtered, df_validation, args, logger, loaded_model)
    
    elif args.test:
        if loaded_model is None:
            logger.error("No model loaded. Use --load-model to specify a model to test.")
            return

        df_test = df.filter(
            (pl.col("datetime").dt.month() == 1) &
            (pl.col("datetime").dt.year() == 2024) 
        )
        test_results = test_model(df_test, loaded_model, args, logger, loader)
        logger.info("Model testing completed")
    
    elif getattr(args, 'fit_kde', None):
        column_name = args.fit_kde
        # Create prefix for this analysis
        prefix = _get_prefix(args) if any([args.year, args.month]) else "full_data"
        results = fit_kde_distribution(df_filtered, column_name, training_plots_path, prefix, logger, sample_size=2000000)
        logger.info("KDE analysis completed")

    elif args.wavelets:

        # Periodogram analysis of close price series
        logger.info("Performing periodogram analysis of close price series")
        plot_periodogram(df["close"], sampling_rate=1.0, title="Periodogram of Close Price Series")
        logger.info("Periodogram analysis completed")

        # Wavelet decomposition of close price series
        logger.info("Performing wavelet decomposition of close price series")
        decomposition = wavelet_decompose(df["close"], levels=19, seasonal_details = 0)
        
        # Plot decomposition results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        
        # Original series
        ax1.plot(decomposition['original'].to_list(), label='Original Close Price', color='black', linewidth=1)
        ax1.set_title('Original Close Price Series')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Decomposed components
        ax2.plot(decomposition['trend'].to_list(), label='Trend', alpha=0.8, linewidth=2)
        ax2.plot(decomposition['seasonality'].to_list(), label='Seasonality', alpha=0.7, linewidth=1)
        ax2.plot(decomposition['residual'].to_list(), label='Residual', alpha=0.6, linewidth=1)
        ax2.set_title('Wavelet Decomposition Components')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Wavelet decomposition completed")

    elif args.stationarity:

        # Check stationarity of close price series
        logger.info("Checking stationarity of close price series")
        stationarity_result = check_stationarity(df["close"])
        logger.info(f"Stationarity test results: {stationarity_result['interpretation']}")
        
        # Check stationarity of close price changes
        logger.info("Checking stationarity of close price changes")
        stationarity_result_changes = check_stationarity(df["c1_detrended_close"])
        logger.info(f"Stationarity test results for changes: {stationarity_result_changes['interpretation']}")

    elif args.autocorrelation:
        # Create autocorrelation analysis for close price changes
        autocorr = Autocorrelation(
            df_filtered["c1_detrended_close"],
            logger,
            window=0,
            name='Autocorrelation of Close Price Changes',
            K=50)
        autocorr.plot()

        #autocorr = Autocorrelation(
        #    df["c1_detrended_close"],
        #    logger,
        #    window=120,
        #    name='Autocorrelation of Close Price Changes - 120 minutes rolling window',
        #    K=20)
        #autocorr.plot()

    elif args.gainloss:
        gain_loss_asymmetry = GainLossAsymmetry(
            series=df_filtered["close"],
            logger=logger,
            name='Gain Loss Asymmetry',
            threshold=0.01,
        )
        gain_loss_asymmetry.plot()

    elif args.volatility_clustering:
        volatility_clustering = VolatilityClustering(
            series=df_filtered["c1_detrended_close"],
            logger=logger,
            name='Volatility Clustering',
            p=1,  # GARCH(1,1)
            q=1,
            dist='t',
            datetime_series=df_filtered["datetime"]
        )
        volatility_clustering.plot()

    elif args.volume_volatility_corr:
        volume_vol_corr = VolumeVolatilityCorrelation(
            open_series=df_filtered["open"],
            high_series=df_filtered["high"],
            low_series=df_filtered["low"],
            close_series=df_filtered["close"],
            volume_series=df_filtered["volume"],
            logger=logger,
            name='Volume-Volatility Correlation Analysis (Garman-Klass)',
            datetime_series=df_filtered["datetime"],
            correlation_windows=[1440]
        )
        volume_vol_corr.plot()
    
    elif getattr(args, 'test_generation', False):
        if loaded_model is None:
            logger.error("No model loaded. Use --load-model to specify a model to test generation quality.")
            return
        
        logger.info("Starting generation quality testing with discriminator")
        
        # Create data loader for discriminator training
        data_loader = BatchedTimeseriesSequence(df_filtered,
            batch_size=args.batch_size, 
            window=args.window,
            logger=logger,
            shuffle=True)
        
        # Create STAAR trainer from loaded model
        staar_trainer = StaarModelTrainer(
            staar_model=loaded_model,
            garch_params=garch_params,
            loss_weights=loss_weights
        )
        
        # Create STAAR tester
        logger.info("Creating discriminator and preparing training data...")
        staar_tester = StaarTester(
            dataset=data_loader,
            staar_trainer=staar_trainer,
            window_size=args.window,
            n_features=FEATURES,  # OHLC data
            logger=logger
        )
        
        # Train the discriminator
        logger.info(f"Training discriminator on {len(staar_tester.X)} total sequences")
        training_results = staar_tester.train(
            epochs=20,
            batch_size=args.batch_size,
            verbose=1
        )
        
        # Create visualization
        staar_tester.visualize_results(training_results, "plots/discriminator_results.png")

    elif args.ppp:
        trainer = StaarModelTrainer(loaded_model, {}, {})
        data_loader = BatchedTimeseriesSequence(df_filtered,
            batch_size=args.batch_size, 
            window=args.window,
            logger=logger,
            shuffle=True)

        gen_c1_close = trainer.generate(1)[0,:,0]
        x, _ = next(iter(data_loader))
        recon = trainer.call(x)[0,:,0]
        
        original_sequence = x[0,:,0]
        logger.info(gen_c1_close)
        logger.info(original_sequence)
        logger.info(recon)

        dummy_array = np.zeros_like(gen_c1_close)
        _, _, _, original_sequence_scale = loader.inverse_transform_series(
            dummy_array, dummy_array, dummy_array, original_sequence
        )
        _, _, _, gen_c1_close_scale = loader.inverse_transform_series(
            dummy_array, dummy_array, dummy_array, gen_c1_close
        )
        _, _, _, recon_scale = loader.inverse_transform_series(
            dummy_array, dummy_array, dummy_array, recon
        )

        logger.info(np.std(original_sequence_scale))
        logger.info(np.std(gen_c1_close_scale))
        logger.info(np.std(recon_scale))

        plt.plot(original_sequence_scale, label='Original Close Price Changes', color='blue')
        #plt.plot(gen_c1_close_scale*0.1, label='generated', color='red')
        plt.plot(recon_scale*0.1, label='generated', color='green')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    main()
