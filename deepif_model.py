# deepif_model.py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import shap

# Define the Autoencoder in PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def train_autoencoder(model, dataloader, epochs, lr, device):
    """
    Train the autoencoder model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"DeepIF: Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return model

def run_deepif(df, feature_columns, deepif_params, preprocessor):
    """
    Apply Deep Isolation Forest (DeepIF) for anomaly detection.
    """
    # Extract features
    X = df[feature_columns].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to torch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=deepif_params['batch_size'], shuffle=True)
    
    # Define autoencoder
    input_dim = X_scaled.shape[1]
    latent_dim = deepif_params['latent_dim']
    autoencoder = Autoencoder(input_dim, latent_dim)
    
    # Train autoencoder
    autoencoder = train_autoencoder(
        model=autoencoder,
        dataloader=dataloader,
        epochs=deepif_params['epochs'],
        lr=deepif_params['learning_rate'],
        device='cpu'  # Change to 'cuda' if GPU is available
    )
    
    # Extract latent embeddings
    autoencoder.eval()
    with torch.no_grad():
        latent_embeddings = autoencoder.encoder(X_tensor)  # Corrected line
    latent_embeddings = latent_embeddings.numpy()
    
    # Train Isolation Forest on latent embeddings
    isolation_forest = IsolationForest(
        n_estimators=deepif_params['if_n_estimators'],
        contamination=deepif_params['if_contamination'],
        random_state=42
    )
    isolation_forest.fit(latent_embeddings)
    
    # Predict anomalies
    anomaly_scores = isolation_forest.decision_function(latent_embeddings)
    anomaly_labels = isolation_forest.predict(latent_embeddings)  # -1 = anomaly, 1 = normal
    
    # Add results to DataFrame
    df['deepif_anomaly_score'] = anomaly_scores
    df['deepif_anomaly_label'] = (anomaly_labels == -1)
    
    logging.info(f"DeepIF: Detected {df['deepif_anomaly_label'].sum()} anomalies out of {len(df)} trades.")
    
    # Optional: Interpretability with SHAP
    # Note: SHAP explanations for autoencoder can be complex. Here's a basic example.
    try:
        explainer = shap.Explainer(autoencoder.encoder, X_scaled)
        shap_values = explainer(X_scaled[:100])  # Explain first 100 instances for speed
        shap.summary_plot(shap_values, X_scaled[:100], feature_names=preprocessor.get_feature_names_out())
    except Exception as e:
        logging.error(f"SHAP explainability failed: {e}")
    
    return df, isolation_forest