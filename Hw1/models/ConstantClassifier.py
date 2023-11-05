import numpy as np

class ConstantClassifier:

    def __init__(self) -> None:
        self.feature_dim = 0
        self.constant = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if np.mean(y) < 0.5:
            self.constant = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.constant)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0],2), float(self.constant))

