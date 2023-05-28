""" preprocessing methods"""
import numpy as np

class MinMaxScaler:
    """
    equivalent implementation from scratch of the MinMaxScaler from sklearn
    methods:
    - fit()
    - transform()
    - fit_transform()
    """
    def __init__(self, feature_range=(0, 1)):
        self.range_ = feature_range
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
    
    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted")
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return X_scaled * (self.range_[1] - self.range_[0]) + self.range_[0]  # scale to feature_range
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class StandardScaler:
    """
    equivalent implementation from scratch of the StandardScaler from sklearn
    methods:
    - fit()
    - transform()
    - fit_transform()
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)