import os
import pandas as pd
import numpy as np

# set the path of the processed data

def data_loader():
    processed_data_path = os.path.join(os.path.pardir, 'src', 'data','processed')
    train_file_path = os.path.join(processed_data_path, 'train.csv')
    test_file_path = os.path.join(processed_data_path, 'test2.csv')
    
    X = pd.read_csv(train_file_path)
    y = pd.read_csv(test_file_path)
    
    return X, y