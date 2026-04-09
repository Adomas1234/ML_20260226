import numpy as np
import pandas as pd

data = pd.read_csv('./data/entities.csv')

X = data[['x1', 'X2']]
Y = data['y']

def train_test_split(X, Y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X.iloc[train_indices]
    Y_train = Y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    Y_test = Y.iloc[test_indices]
    
    return X_train, X_test, Y_train, Y_test