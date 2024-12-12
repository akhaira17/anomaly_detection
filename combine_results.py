# combine_results.py
import pandas as pd
import logging

def combine_results(df_eif, df_deepif, original_df):
    """
    Combine EIF and DeepIF results with the original DataFrame.
    Strategies:
    - Logical AND: Flag as anomaly only if both models agree.
    - Average Scores: Combine anomaly scores for ranking.
    """
    # Ensure the DataFrames have the same index
    if not (df_eif.index.equals(df_deepif.index) and df_eif.index.equals(original_df.index)):
        logging.error("DataFrames do not have matching indices. Cannot combine results.")
        raise ValueError("DataFrames indices mismatch.")
    
    # Merge anomaly labels and scores
    combined_df = original_df.copy()
    combined_df['eif_anomaly_label'] = df_eif['eif_anomaly_label']
    combined_df['eif_anomaly_score'] = df_eif['eif_anomaly_score']
    combined_df['deepif_anomaly_label'] = df_deepif['deepif_anomaly_label']
    combined_df['deepif_anomaly_score'] = df_deepif['deepif_anomaly_score']
    
    # Logical AND
    combined_df['combined_anomaly_label'] = combined_df['eif_anomaly_label'] & combined_df['deepif_anomaly_label']
    
    # Average Scores (assuming lower scores indicate anomalies)
    combined_df['combined_anomaly_score'] = (combined_df['eif_anomaly_score'] + combined_df['deepif_anomaly_score']) / 2.0
    
    logging.info(f"Combined Results: {combined_df['combined_anomaly_label'].sum()} anomalies detected.")
    
    return combined_df