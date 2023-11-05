import pandas as pd
import numpy as np
from copy import deepcopy

from importlib import reload
import utils.Metrics
import utils.DatasetInfo

reload(utils.Metrics)
reload(utils.DatasetInfo)

from utils.Metrics import Metrics
from utils.DatasetInfo import DatasetInfo


class ExperimentInfo:
    '''
    Class represents info about experiment(evaluating multiple models on dataset)
    Here's supported actions:
    - fitting multiple models
    - evaluating multiple models
    - drop "bad" models method, which drops model if exists better by all metrics model
    - redifined __str__ method, to print experiment info
    - best_model method, which saves best model for each metric
    - save best model method, which saves experiment info in format required by  
    '''

    def __init__(self, dataset: DatasetInfo, models: dict) -> None:
        '''
        Expected list of models looks like:
        {'model1' : dict_with_model}
        dict_with_model = {
            'model' : model_object,
            'description' : model_description
        }

        model must support .fit(), .predict(), .predict_proba_methods()
        '''
        self.dataset = dataset
        self.models = models
        self.best_models = None
        self.bad_models = None

    def fit(self, hide_progress: bool):
        for model, model_data in self.models.items():
            if not hide_progress:
                print(~hide_progress)
                print(f'Fitting {model}')
            model_data['model'].fit(self.dataset.X_train, self.dataset.y_train)
    
    def evaluate(self, hide_progress: bool):
        for model_name, model_data in self.models.items():
            if not hide_progress:
                print(f'Evaluating {model_name}')
            model_data['train'] = Metrics(self.dataset, model_data['model'], 'train', model_data['description'])
            model_data['test'] = Metrics(self.dataset, model_data['model'], 'test', model_data['description'])

    def drop_bad_models(self):
        self.bad_models = {}
        models_to_remove = []

        for model_name, model_data in self.models.items():
            # is_bad = True
            is_bad = False


            # Check if the current model is less than every other model
            for other_name, other_data in self.models.items():
                if model_name != other_name: # Do not compare with itself
                    if (model_data['test'] < other_data['test']):
                        is_bad = True
                        break

            # If it is less than all other models, add to bad models and mark for removal
            if is_bad:
                self.bad_models[model_name] = model_data
                models_to_remove.append(model_name)

        # Remove bad models from the models dictionary
        # for model_name in models_to_remove:
        #     self.bad_models[model_name] = self.models[model_name]
        #     # del self.models[model_name]
            
        #     self.models.pop(model_name)

    def __str__(self):
        experiment_info_str = str(self.dataset)
        if self.best_models is None:
            for model_name, model_data in self.models.items():
                experiment_info_str += str(model_data['test'])
                experiment_info_str += str(model_data['train'])

        else:
            # for metric_name, model in self.best_models.items():
            experiment_info_str += (
                f'{"AUC"} = {self.models[self.best_models["AUC"]]["test"].auc}, model = {self.best_models["AUC"]}\n'
                f'{"NUM"} = {self.models[self.best_models["NUM"]]["test"].num}, model = {self.best_models["NUM"]}\n'
                f'{"ASY1"} = {self.models[self.best_models["ASY1"]]["test"].asy1}, model = {self.best_models["ASY1"]}\n'
                f'{"ASY2"} = {self.models[self.best_models["ASY2"]]["test"].asy2}, model = {self.best_models["ASY2"]}\n'
            )
        if self.bad_models is not None:
            experiment_info_str += (
                f'Definetly worse models: \n'
                f'{list(self.bad_models.keys())}'
            )        
        return experiment_info_str
                

    def best_model(self):
        best_auc_model = None
        best_auc_value = -np.inf
        
        best_num_model = None
        best_num_value = np.inf
        
        best_asy1_model = None
        best_asy1_value = np.inf
        
        best_asy2_model = None
        best_asy2_value = np.inf

        for model_name, model_data in self.models.items():
            # Check AUC
            if model_data['test'].auc > best_auc_value:
                best_auc_model = model_name
                best_auc_value = model_data['test'].auc
            
            # Check NUM
            if model_data['test'].num < best_num_value:
                best_num_model = model_name
                best_num_value = model_data['test'].num
                
            # Check ASY1
            if model_data['test'].asy1 < best_asy1_value:
                best_asy1_model = model_name
                best_asy1_value = model_data['test'].asy1
                
            # Check ASY2
            if model_data['test'].asy2 < best_asy2_value:
                best_asy2_model = model_name
                best_asy2_value = model_data['test'].asy2

        self.best_models = {
            'AUC': best_auc_model,
            'NUM': best_num_model,
            'ASY1': best_asy1_model,
            'ASY2': best_asy2_model
        }      

    def auto_experiment(self, hide_progress: bool=True):
        self.fit(hide_progress=hide_progress)
        self.evaluate(hide_progress=hide_progress)
        self.drop_bad_models()
        self.best_model()
        print(self)

    def save_best_model(self):
        tmp_dict = {}
        # for metric in ['NUM', 'ASY1', 'ASY2']:
        #     tmp_dict[metric] = self.models[self.best_models[metric]]['model'].predict_proba(self.dataset.X_predict)
        tmp_dict['NUM'] = self.models[self.best_models['NUM']]['model'].predict_proba(self.dataset.X_predict)[:, 1] >\
        self.models[self.best_models['NUM']]['train'].num_thr

        tmp_dict['ASY1'] = self.models[self.best_models['ASY1']]['model'].predict_proba(self.dataset.X_predict)[:, 1] >\
        self.models[self.best_models['ASY1']]['train'].asy1_thr

        tmp_dict['ASY2'] = self.models[self.best_models['ASY2']]['model'].predict_proba(self.dataset.X_predict)[:, 1] >\
        self.models[self.best_models['ASY2']]['train'].asy2_thr

        tmp_dict['AUC'] = self.models[self.best_models['AUC']]['model'].predict_proba(self.dataset.X_predict)[:, 1]
        df = pd.DataFrame(tmp_dict)
        df.to_csv(f'results/task1_{self.dataset.num_of_dataset}_ans.csv', index=False)