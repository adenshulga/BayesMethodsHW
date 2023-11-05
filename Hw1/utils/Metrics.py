import sklearn
import numpy as np

from typing import Literal

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import utils.DatasetInfo

from importlib import reload

reload(utils.DatasetInfo)
from utils.DatasetInfo import DatasetInfo


class Metrics:
    '''
    Class handles: 
    - metric evaluation(AUC, NUM, ASY) for given threshold
    - plotting ROC
    - plotting metric thresholding dependence
    - redifined __str__ method to obtain all metrics
    - >, < operators, if one model is worse by all metrics than other, returns True
    '''

    P = np.array([[0,1], [1, 0]])
    P1 = np.array([[-9, 9], [1, 0]])
    P2 = np.array([[-1,3], [2, -1]])

    def AUC(self) -> float:
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_gt, self.y_pred_prob[:, 1])
        # Filter the values where thresholds are between 0 and 1
        valid_indices = np.where((self.thresholds >= 0) & (self.thresholds <= 1))
        # valid_indices = valid_indices[~np.isnan(valid_indices)]
        self.fpr = self.fpr[valid_indices]
        self.tpr = self.tpr[valid_indices]
        self.thresholds = self.thresholds[valid_indices]
        self.auc = auc(self.fpr, self.tpr)
    
    def NUM(self, threshold: float=None) -> float:
        if threshold is None:
            return np.mean(~(self.y_pred == self.y_gt))
        else:
            y_pred = np.where(self.y_pred_prob[:, 1] > threshold, 1, 0)
            # return np.mean(~(y_pred == self.y_gt))
            return np.sum(~(y_pred == self.y_gt))

    def ASY(self, P, threshold: float=None) -> float:
        if threshold is None:
            return np.sum(P*confusion_matrix(self.y_gt, self.y_pred))/self.y_gt.shape[0]
        else:
            y_pred = np.where(self.y_pred_prob[:, 1] > threshold, 1, 0)
            # return np.sum(P * confusion_matrix(self.y_gt, y_pred)) / self.y_gt.shape[0]
            return np.sum(P * confusion_matrix(self.y_gt, y_pred))

    
    def plot_ROC(self, figsize=(10, 8)) -> None:
        plt.figure(figsize=figsize)
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show() 
    
    def plot_thresholding_dependence(self, figsize=(10, 8)) -> None:
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        
        num_vals = [self.NUM(t) for t in self.thresholds[:-1]]
        asy1_vals = [self.ASY(self.P1, t) for t in self.thresholds[:-1]]
        asy2_vals = [self.ASY(self.P2, t) for t in self.thresholds[:-1]]
    
        axs[0].plot(self.thresholds[:-1], num_vals, color='blue', lw=2, label='NUM')
        axs[0].set_title('NUM vs. Threshold')
        axs[0].set_xlabel('Threshold')
        axs[0].set_ylabel('NUM')
        axs[0].legend(loc="best")
        axs[0].grid(True)
    
        axs[1].plot(self.thresholds[:-1], asy1_vals, color='red', lw=2, label='ASY1')
        axs[1].set_title('ASY1 vs. Threshold')
        axs[1].set_xlabel('Threshold')
        axs[1].set_ylabel('ASY1')
        axs[1].legend(loc="best")
        axs[1].grid(True)
    
        axs[2].plot(self.thresholds[:-1], asy2_vals, color='green', lw=2, label='ASY2')
        axs[2].set_title('ASY2 vs. Threshold')
        axs[2].set_xlabel('Threshold')
        axs[2].set_ylabel('ASY2')
        axs[2].legend(loc="best")
        axs[2].grid(True)

    
        plt.tight_layout()
        plt.show()

    

    def save_y(self, y_gt: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray) -> None:
            self.y_gt = y_gt
            self.y_pred = y_pred
            self.y_pred_prob = y_pred_prob

    def __init__(self, dataset: DatasetInfo, model: sklearn, evaluation_mode: Literal['train', 'test'], model_description: str = None) -> None:
        '''model has to be already fitted'''
        
        self.dataset = dataset

        if model_description is None:
            model_description = model.__class__.__name__

        self.evaluation_mode = evaluation_mode
        
        if evaluation_mode=='train':
            self.save_y(dataset.y_train,
                        model.predict(self.dataset.X_train),
                        model.predict_proba(self.dataset.X_train))
        elif evaluation_mode=='test':
            self.save_y(dataset.y_test,
                        model.predict(self.dataset.X_test),
                        model.predict_proba(self.dataset.X_test))
            
        self.model_description = model_description

        self.AUC()
        # self.num = self.NUM()
        # self.asy1 = self.ASY(Metrics.P1)
        # self.asy2 = self.ASY(Metrics.P2)
        self.get_best_thresholds()

    # TODO: write this in better way
    def get_best_thresholds(self):

        num_scores = []
        asy1_scores = []
        asy2_scores = []
        for thr in self.thresholds:
            num_scores.append(self.NUM(thr))
            asy1_scores.append(self.ASY(Metrics.P1, thr))
            asy2_scores.append(self.ASY(Metrics.P2, thr))

        num_scores = np.array(num_scores)
        asy1_scores = np.array(asy1_scores)
        asy2_scores = np.array(asy2_scores)

        self.num = num_scores.min()
        self.num_thr = self.thresholds[num_scores.argmin()]

        self.asy1 = asy1_scores.min()
        self.asy1_thr = self.thresholds[asy1_scores.argmin()]

        self.asy2 = asy2_scores.min()
        self.asy2_thr = self.thresholds[asy2_scores.argmin()]


    def __str__(self) -> str:
        metrics_info = (
            f'-------------------------------------------------\n'
            f'{self.model_description}\n'
            f'Evaluation mode: {self.evaluation_mode}\n'
            f'AUC: {self.auc} \n'
            f'NUM: {self.num} \n'
            f'ASY1: {self.asy1}\n'
            f'ASY2: {self.asy2}\n'
            f'-------------------------------------------------\n'
        ) 
        return metrics_info

    def __gt__(self, other) -> bool:
        return (self.auc > other.auc) and (self.num < other.num) and (self.asy1 < other.asy1) and (self.asy2 < other.asy2)
    
    def __lt__(self, other) -> bool:
        return other > self

