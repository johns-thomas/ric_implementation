import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    # Define the ranges for the synthetic data
    memory_allocation = np.random.choice([128, 256, 512, 1024, 2048], num_samples)
    timeout_setting = np.random.uniform(1, 300, num_samples)
    concurrency_level = np.random.randint(1, 100, num_samples)
    
    # Generate synthetic performance metrics
    execution_time = np.random.normal(loc=100, scale=20, size=num_samples)  # in milliseconds
    error_rate = np.random.beta(a=2, b=20, size=num_samples)  # low error rates
    cold_start_frequency = np.random.poisson(lam=0.1, size=num_samples)  # rare cold starts
    cost = (memory_allocation / 1024) * execution_time * 0.00001667  # Simplified cost model
    
    # Create a DataFrame
    synthetic_data = pd.DataFrame({
        'memory_allocation': memory_allocation,
        'timeout_setting': timeout_setting,
        'concurrency_level': concurrency_level,
        'average_execution_time': execution_time,
        'error_rate': error_rate,
        'cold_start_frequency': cold_start_frequency,
        'cost': cost
    })
    
    return synthetic_data

#synthetic_data = generate_synthetic_data()
