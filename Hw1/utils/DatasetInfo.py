import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

class DatasetInfo:
    '''
    Class encapsulates:
    - loading of datasets
    - X, y, X_predict
    - train, test split
    - X_train, X_test 
    - scaling of features
    - computing svd
    - plotting singular values
    - representation with redifined __str__ method

    '''

    def _load_csv(self, num: int) -> tuple:

        path_x = f'data/task1_{num}_learn_X.csv'
        path_y = f'data/task1_{num}_learn_y.csv'
        path_x_test = f'data/task1_{num}_test_X.csv'

        X = np.loadtxt(path_x, delimiter=' ')
        y = np.loadtxt(path_y)
        X_test = np.loadtxt(path_x_test, delimiter=' ')

        return (X, y, X_test)
    
    def train_test_split(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, stratify=self.y)

    def __init__(self, num_of_dataset: int) -> None:
        '''

        '''

        self.num_of_dataset = num_of_dataset
        (self.X_initial, self.y, self.X_predict_initial) = self._load_csv(num_of_dataset)
        self.num_of_samples = self.X_initial.shape[0]
        self.num_of_features = self.X_initial.shape[1]

        self.experiments_info = {}
        self.test_size = 0.2

        self.class_balance = np.mean(self.y)

        self.dataset_info = (
            f"Dataset num: {self.num_of_dataset}\n"
            f"Total number of samples = {self.num_of_samples}, features = {self.num_of_features}\n"
            f'Train/test split ratio {1 - self.test_size}/{self.test_size}\n'
            f'Positive class ratio: {self.class_balance}\n'
        )
        print(self)

        self.pca = None

        self.preprocess_dataset()
        self.train_test_split()
        self.compute_PCA()


    def __str__(self) -> str:
        'I decided to store info in a string to do further concatenation to other strings'
        return self.dataset_info
    
    def preprocess_dataset(self):
        
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_initial)
        self.X = self.scaler.transform(self.X_initial)
        self.X_predict = self.scaler.transform(self.X_predict_initial)

    def compute_PCA(self):
        self.pca = PCA()
        self.pca.fit(self.X_train)

    def plot_singular_values(self, figsize=(10,8)):
        if self.pca is None:
            self.compute_PCA()
            
        singular_values = self.pca.singular_values_

        plt.figure(figsize=(10, 6))
        plt.plot(singular_values, 'o-')
        plt.title('Singular Values vs. Number of Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Singular Value')
        plt.grid(True)
        plt.show()

        