# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, feature_columns):
    # Separate features
    X = df[feature_columns]
    
    # Define categorical and numerical columns
    categorical_cols = ['product_type', 'counterparty']
    numerical_cols = ['our_cents', 'cp_cents', 'notional', 'impact_dollars']
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after encoding
    ohe = preprocessor.named_transformers_['cat']['onehot']
    cat_features = ohe.get_feature_names_out(categorical_cols)
    all_features = numerical_cols + list(cat_features)
    
    X_processed_df = pd.DataFrame(X_processed, columns=all_features)
    
    return X_processed_df, preprocessor