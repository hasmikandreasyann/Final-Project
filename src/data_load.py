# src/data_load.py
import pandas as pd
from pathlib import Path

def load_dataset(file_name):
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'  
    file_path = data_dir / 'raw' / file_name

    # Load the dataset
    dataset = pd.read_csv(file_path)
    return dataset

if __name__ == "__main__":

    train_data = load_dataset('train.csv')
    test_data = load_dataset('test.csv')