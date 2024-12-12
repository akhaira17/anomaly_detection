# interpretability.py
import shap
import matplotlib.pyplot as plt

def explain_anomalies(model, df, feature_columns, preprocessor, top_n=10):
    """
    Use SHAP to explain the top N anomalies.
    """
    # Select top N anomalies based on combined score
    anomalies = df[df['combined_anomaly_label']].sort_values('combined_anomaly_score').head(top_n)
    if anomalies.empty:
        print("No anomalies to explain.")
        return
    
    # Preprocess the data as done before
    X = anomalies[feature_columns].values
    X_processed = preprocessor.transform(anomalies[feature_columns])
    
    # Create a DataFrame with processed features
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Initialize SHAP
    explainer = shap.Explainer(model.encoder, X_processed_df)
    shap_values = explainer(X_processed_df)
    
    # Plot SHAP summary
    shap.summary_plot(shap_values, X_processed_df, feature_names=feature_names)