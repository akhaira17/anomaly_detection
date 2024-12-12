# main.py
import logging
import pandas as pd
from config import CONFIG  # Updated import
from data_generation import generate_synthetic_data
from data_preprocessing import preprocess_data
from eif_model import run_eif
from deepif_model import run_deepif
from combine_results import combine_results
from store_results import store_results

def main():
    # Setup Logging
    logging.basicConfig(
        level=getattr(logging, CONFIG['LOG_LEVEL']),
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Generate Synthetic Data (Replace this with actual data loading if needed)
    logging.info("Generating synthetic data...")
    data = generate_synthetic_data(size=CONFIG['SAMPLE_DATA_SIZE'])
    
    # Display first few rows of the synthetic data
    print(data.head())
    
    # Preprocess Data
    logging.info("Preprocessing data...")
    X_processed_df, preprocessor = preprocess_data(data, CONFIG['FEATURE_COLUMNS'])
    
    # Run Extended Isolation Forest (EIF) on preprocessed features
    logging.info("Running Extended Isolation Forest (EIF)...")
    processed_eif_df = run_eif(
        df=X_processed_df.copy(),
        feature_columns=X_processed_df.columns.tolist(),
        eif_params=CONFIG['EIF_PARAMS']
    )
    
    # Run Deep Isolation Forest (DeepIF) on preprocessed features
    logging.info("Running Deep Isolation Forest (DeepIF)...")
    processed_deepif_df, deepif_model = run_deepif(
        df=X_processed_df.copy(),
        feature_columns=X_processed_df.columns.tolist(),
        deepif_params=CONFIG['DEEPIF_PARAMS'],
        preprocessor=preprocessor
    )
    
    # Combine Results
    logging.info("Combining anomaly detection results...")
    combined_df = combine_results(
        df_eif=processed_eif_df,
        df_deepif=processed_deepif_df,
        original_df=data
    )
    
    # Optional: Explain anomalies (Top N)
    # Uncomment the following lines if you want to generate SHAP explanations
    from interpretability import explain_anomalies
    explain_anomalies(deepif_model, combined_df, preprocessor, top_n=10)
    
    # Store Results
    logging.info("Storing results...")
    store_results(combined_df, CONFIG)
    
    logging.info("Anomaly detection pipeline completed successfully.")

if __name__ == "__main__":
    main()