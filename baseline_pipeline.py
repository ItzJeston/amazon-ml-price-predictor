import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# 1. Custom SMAPE Evaluation Metric
def calculate_smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

# 2. Text Normalization Function
def clean_text(text):
    """Lowercases text and removes special characters."""
    text = str(text).lower()
    return re.sub(r'[^\w\s]', '', text)

def run_pipeline():
    # --- PHASE 1: Mock Data Generation (Replace with pd.read_csv later) ---
    print("Generating mock e-commerce dataset...")
    train_data = pd.DataFrame({
        'sample_id': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        'catalog_content': [
            "Apple iPhone 13 128GB Blue", 
            "Samsung Galaxy S21 5G", 
            np.nan, # Simulating missing data
            "Sony WH-1000XM4 Wireless Headphones", 
            "Sony WH-1000XM4 Wireless Headphones", # Simulating duplicate
            "Dell XPS 13 Laptop 16GB RAM"
        ],
        'price': [799.00, 699.00, 450.00, 348.00, 348.00, 1200.00]
    })
    
    test_data = pd.DataFrame({
        'sample_id': ['V1', 'V2'],
        'catalog_content': ["Apple iPhone 14 Pro", "Bose QuietComfort 45"]
    })

    # --- PHASE 2: Data Preprocessing ---
    print("\n--- Starting Data Preprocessing ---")
    # Drop duplicates based on unique identifier
    train_data = train_data.drop_duplicates(subset=['sample_id'])
    
    # Handle missing values
    train_data = train_data.dropna(subset=['price'])
    train_data['catalog_content'] = train_data['catalog_content'].fillna("unknown_item")
    test_data['catalog_content'] = test_data['catalog_content'].fillna("unknown_item")
    
    # Normalize text
    train_data['clean_content'] = train_data['catalog_content'].apply(clean_text)
    test_data['clean_content'] = test_data['catalog_content'].apply(clean_text)
    print(f"Cleaned Training Data Shape: {train_data.shape}")

    # --- PHASE 3: Feature Extraction (TF-IDF) ---
    print("\n--- Extracting Features ---")
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit on training text and transform both train and test text
    X_train_full = vectorizer.fit_transform(train_data['clean_content'])
    y_train_full = train_data['price'].values
    X_test_final = vectorizer.transform(test_data['clean_content'])

    # --- PHASE 4: Model Training & Validation ---
    print("\n--- Training Model ---")
    # Split training data to create a local validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    # Initialize and train Ridge Regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predict and evaluate on validation set
    val_predictions = model.predict(X_val)
    # Ensure no negative price predictions
    val_predictions = np.maximum(val_predictions, 0) 
    
    smape_score = calculate_smape(y_val, val_predictions)
    print(f"Local Validation SMAPE Score: {smape_score:.2f}%")

    # --- PHASE 5: Formatting the Submission ---
    print("\n--- Generating Test Predictions ---")
    # Retrain on the entire training dataset for maximum accuracy
    model.fit(X_train_full, y_train_full)
    
    # Predict on the unlabelled test set
    test_predictions = model.predict(X_test_final)
    test_predictions = np.maximum(test_predictions, 0)
    
    # Format output directly to CSV requirements
    submission_df = pd.DataFrame({
        'sample_id': test_data['sample_id'],
        'price': test_predictions
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print("Saved 'submission.csv' successfully. Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()