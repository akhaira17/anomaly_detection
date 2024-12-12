# anomaly_config.py
import os

CONFIG = {
    # "DB_CONNECTION_STRING": os.getenv("DB_CONNECTION_STRING", "sqlite:///trades.db"),  # Removed or commented out
    "BATCH_DATE": os.getenv("BATCH_DATE", "2024-12-11"),
    "EIF_PARAMS": {
        "ntrees": 100,
        "sample_size": 256,
        "extension_level": 1,  # Corrected to match 'ExtensionLevel' in eif
        "contamination": 0.01  # 1% anomalies
    },
    "DEEPIF_PARAMS": {
        "latent_dim": 2,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "if_n_estimators": 100,
        "if_contamination": 0.01
    },
    "FEATURE_COLUMNS": ["our_cents", "cp_cents", "notional", "impact_dollars", "product_type", "counterparty"],
    "LOG_LEVEL": "INFO",
    "RESULT_TABLE": "trade_anomaly_detection_results",  # Can be removed if not using
    "SAMPLE_DATA_SIZE": 1000  # Number of samples for synthetic data
}