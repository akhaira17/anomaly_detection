# eif_model.py
import numpy as np
import pandas as pd
from eif import iForest
import logging

def run_eif(df, feature_columns, eif_params):
    """
    Apply Extended Isolation Forest (EIF) for anomaly detection.
    """
    # Log available columns
    logging.debug(f"Available columns in EIF DataFrame: {df.columns.tolist()}")
    logging.debug(f"Expected feature columns: {feature_columns}")
    
    # Extract features
    X = df[feature_columns].values
    
    # Initialize EIF with corrected parameter names and pass X as the first argument
    try:
        model = iForest(
            X,  # Pass X as the first positional argument
            ntrees=eif_params['ntrees'],
            sample_size=eif_params['sample_size'],
            ExtensionLevel=eif_params['extension_level']  # Corrected parameter name
        )
    except TypeError as te:
        logging.error(f"TypeError during EIF model initialization: {te}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during EIF model initialization: {e}")
        raise
    
    # Compute anomaly scores (lower scores are more anomalous)
    try:
        scores = model.compute_paths(X_in=X)
    except Exception as e:
        logging.error(f"Error computing anomaly scores: {e}")
        raise
    
    # Determine threshold based on contamination
    try:
        threshold = np.percentile(scores, eif_params['contamination'] * 100)
    except Exception as e:
        logging.error(f"Error calculating threshold: {e}")
        raise
    
    # Label anomalies
    labels = scores < threshold  # True if anomaly
    
    # Add to DataFrame
    df['eif_anomaly_label'] = labels
    df['eif_anomaly_score'] = scores
    
    logging.info(f"EIF: Detected {labels.sum()} anomalies out of {len(labels)} trades.")
    
    return df