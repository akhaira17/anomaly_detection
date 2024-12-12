# data_generation.py
import numpy as np
import pandas as pd

def generate_synthetic_data(size=1000, random_state=42):
    np.random.seed(random_state)
    
    # Numerical features
    our_cents = np.random.normal(loc=100, scale=10, size=size)
    cp_cents = our_cents + np.random.normal(loc=0, scale=5, size=size)  # Slight variations
    
    # Introduce anomalies
    n_anomalies = int(size * 0.01)  # 1% anomalies
    anomaly_indices = np.random.choice(size, n_anomalies, replace=False)
    our_cents[anomaly_indices] += np.random.normal(loc=50, scale=10, size=n_anomalies)  # Large deviation
    
    notional = np.random.uniform(10000, 1000000, size=size)
    impact_dollars = np.abs(our_cents - cp_cents) * notional / 100.0
    
    # Categorical features
    product_types = ['swaption', 'vanilla_swap', 'option', 'forward']
    counterparties = ['CP_A', 'CP_B', 'CP_C', 'CP_D']
    
    product_type = np.random.choice(product_types, size=size)
    counterparty = np.random.choice(counterparties, size=size)
    
    # Create DataFrame
    data = pd.DataFrame({
        'trade_id': range(1, size + 1),
        'our_cents': our_cents,
        'cp_cents': cp_cents,
        'notional': notional,
        'impact_dollars': impact_dollars,
        'product_type': product_type,
        'counterparty': counterparty
    })
    
    return data