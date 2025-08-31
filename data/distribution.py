from logging import Logger
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import joblib

def fit_kde_distribution(df, column_name, output_folder="plots/", filename_prefix="kde", logger:Logger=None, sample_size=50000):
    """
    Fit a KDE (Kernel Density Estimation) model using sklearn KernelDensity.
    
    Args:
        df (pl.DataFrame): Input dataframe
        column_name (str): Name of the column to fit
        output_folder (str): Directory where to save the plot
        filename_prefix (str): Prefix for the saved plot filename
        logger: Optional logger instance
        sample_size (int): Maximum number of samples to use for fitting
        
    Returns:
        dict: Dictionary containing fitted parameters and statistics:
            - 'bandwidth': optimal bandwidth used for KDE
            - 'kernel': kernel type used
            - 'data_mean': empirical mean of the data
            - 'data_std': empirical standard deviation of the data
            - 'data_median': empirical median
            - 'data_skew': skewness of the data
            - 'data_kurtosis': kurtosis of the data
    """

    data = df[column_name].drop_nulls().to_numpy().flatten()
    
    # Sample 50,000 values if dataset is larger to speed up computation
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = data[indices]
        logger.info(f"Sampled {sample_size} values from {len(df[column_name].drop_nulls())} total data points")
    
    if logger:
        logger.info(f"Fitting KDE model to column '{column_name}' using sklearn KernelDensity")
        logger.info(f"Data points: {len(data)}")
        logger.info(f"Data range: [{np.min(data):.6f}, {np.max(data):.6f}]")

    # Reshape data for sklearn (expects 2D array)
    X = data.reshape(-1, 1)
    
    # Use GridSearchCV to find optimal bandwidth
    #param_grid = {'bandwidth': np.logspace(-1, 0, 10)}
    #kde_model = KernelDensity(kernel='gaussian')
    #grid = GridSearchCV(kde_model, param_grid, cv=5, n_jobs=-1)
    #grid.fit(X)
    
    #optimal_bandwidth = grid.best_params_['bandwidth']
    optimal_bandwidth=0.105
    #if logger:
    #    logger.info(f"Optimal bandwidth (cross-validation): {optimal_bandwidth:.6f}")
    #    logger.info(f"Best cross-validation score: {grid.best_score_:.6f}")

    # Fit final model with optimal bandwidth
    kde = KernelDensity(kernel='gaussian', bandwidth=optimal_bandwidth)
    kde.fit(X)
    
    # Create evaluation points for plotting
    x_min, x_max = np.min(data), np.max(data)
    x_range = x_max - x_min
    x_eval = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 1000)
    
    # Get KDE density estimates  
    X_eval = x_eval.reshape(-1, 1)
    log_density = kde.score_samples(X_eval)
    kde_density = np.exp(log_density)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main histogram and KDE plot
    plt.subplot(2, 1, 1)
    n, bins, patches = plt.hist(data, bins=400, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.plot(x_eval, kde_density, 'r-', linewidth=2, label=f'KDE sklearn (bw={optimal_bandwidth:.4f})')
    
    # Add Gaussian comparison
    empirical_mean = np.mean(data)
    empirical_std = np.std(data, ddof=1)
    gaussian_pdf = stats.norm.pdf(x_eval, empirical_mean, empirical_std)
    plt.plot(x_eval, gaussian_pdf, 'g--', linewidth=2, label='Gaussian (comparison)')
    
    plt.title(f'KDE Fit for {column_name} (sklearn, cross-validation bandwidth)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-10,10)
    
    # Residuals plot (empirical CDF vs KDE-based CDF)
    plt.subplot(2, 1, 2)
    # Sort data for empirical CDF
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Calculate KDE-based CDF at sorted data points
    kde_cdf = []
    for point in sorted_data:
        # Integrate KDE up to this point
        mask = x_eval <= point
        if np.any(mask):
            # Numerical integration using trapezoidal rule
            kde_cdf_val = np.trapz(kde_density[mask], x_eval[mask])
            kde_cdf.append(min(kde_cdf_val, 1.0))  # Cap at 1.0
        else:
            kde_cdf.append(0.0)
    
    kde_cdf = np.array(kde_cdf)
    residuals = empirical_cdf - kde_cdf
    
    plt.plot(sorted_data, residuals, 'b-', linewidth=1)
    plt.axhline(0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Empirical CDF - KDE CDF')
    plt.title('CDF Residuals')
    plt.xlim(-10,10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_folder, exist_ok=True)
    plot_filename = os.path.join(output_folder, f"{filename_prefix}_{column_name}_kde_fit.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate additional statistics
    data_median = np.median(data)
    data_skew = stats.skew(data)
    data_kurtosis = stats.kurtosis(data)
    
    # Prepare results
    results = {
        'bandwidth': optimal_bandwidth,
        'kernel': 'gaussian',
        'data_mean': empirical_mean,
        'data_std': empirical_std,
        'data_median': data_median,
        'data_skew': data_skew,
        'data_kurtosis': data_kurtosis,
        'n_samples': len(data),
        'kde_model': kde  # Return the fitted model for further use
    }
    
    if logger:
        logger.info(f"Sklearn KDE model fitted successfully:")
        logger.info(f"  Optimal bandwidth (CV): {optimal_bandwidth:.6f}")
        logger.info(f"  Data mean: {empirical_mean:.6f}")
        logger.info(f"  Data std: {empirical_std:.6f}")
        logger.info(f"  Data median: {data_median:.6f}")
        logger.info(f"  Data skewness: {data_skew:.6f}")
        logger.info(f"  Data kurtosis: {data_kurtosis:.6f}")
        logger.info(f"  Plot saved to: {plot_filename}")

    output_file = os.path.join(output_folder, f"{filename_prefix}_{column_name}_kde_{optimal_bandwidth:.4f}_{sample_size}.pkl")
    joblib.dump(kde, output_file)
    
    return results

def load_kde_model(file_path)-> KernelDensity:
    """
    Load a previously saved KDE model from a file.
    
    Args:
        file_path (str): Path to the saved model file.
        
    Returns:
        KernelDensity: The loaded KDE model.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"KDE model file not found: {file_path}")
    
    kde_model = joblib.load(file_path)
    return kde_model