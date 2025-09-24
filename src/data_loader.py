import os
import pandas as pd

def load_data(filename='raw.csv'):
    # Corrected the path join logic
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', filename)
    full_path = os.path.abspath(data_path)
    
    print(f"[INFO] Loading file from: {full_path}")  # Print before return
    return pd.read_csv(full_path)
