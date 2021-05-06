"""
The controller will be responsible to get inputs from the user and run the experiment. 
Information such as what are the preprocessing algorithms, models, how many ML process
to run or the time to take into the experiment. After the experiment runs, the controller will
responsible to save and communicate the results.

Glossary:
Experiment -> Test different ML process to find the best for a given dataset and problem.
ML Process -> Combination of Preprocessing Algorithms and Model to get to a prediction.
"""
import time
import random
from MLProcess import MLProcess
import matplotlib

from visual import show_results

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.model_selection import train_test_split

def parameters_select(estimator_list):
    estimators = []
    for i in estimator_list:
        p = {}
        for j in i['parameters']:
            p[i['parameters'][j]['name']] = select_parameter_value(i['parameters'][j]['values'])
        estimators.append(i['model'].set_params(**p))
    return estimators

def select_parameter_value(values):
    if values['type'] == 'boolean':
        value = random.choice([True, False])
    elif values['type'] == 'int':
        value = random.randint(values['min'], values['max'])
    elif values['type'] == 'float':
        value = random.uniform(values['min'], values['max'])
    elif values['type'] == 'list':
        value = random.choice(values['list'])
    return value

class Controller:
    def __init__(self, features, target, test_size, score_func, pipeline_constructor_json, pipelines_to_test):
        self.features = features  # The features of your dataset
        self.target = target  # y part of your dataset
        self.test_size = test_size  # y part of your dataset
        self.score_func = score_func  # Function that you pass y and y_hat and return a score (example MAPE)
        self.pipeline_constructor_json = pipeline_constructor_json # Json with information of estimators, preprocessors and parameters that will be tested
        self.pipelines_to_test = pipelines_to_test  # Number of pipelines that will be tested

        self.estimators = [] # List of estimators that will be tested, it starts as blank and it will be populated by the read_construct_json function.
        self.pre_estimators = [] # List of preprocessing that will be tested, it starts as blank and it will be populated by the read_construct_json function.
        self.read_constructor_json()
        self.run_experiment()

    def run_experiment(self):
        """
        Create pipelines at random score it and save the best pipeline in the object
        """
        # TODO: Split train and test
        # X_train = self.features
        # y_train = self.target
        # X_test = self.features
        # y_test = self.target
        X_train, X_test, y_train, y_test = train_test_split(self.features,self.target, test_size=self.test_size)

        max_score = -float('inf')
        self.results = list()
        for i in range(self.pipelines_to_test):
            start = time.time()
            n_pre_estimators = random.randint(1, len(self.pre_estimators))
            pre_estimators = random.sample(self.pre_estimators, n_pre_estimators)
            estimator = random.sample(self.estimators, 1)

            pre_estimators = parameters_select(pre_estimators)
            estimator = parameters_select(estimator)
            print(estimator)
            print(pre_estimators)
            pipeline = MLProcess(estimator[0],
                                 pre_estimators,
                                 self.score_func)
            pipeline.fit(X=X_train, y=y_train)
            score_train = pipeline.score(X=X_train, y=y_train)
            score_test = pipeline.score(X=X_test, y=y_test)
            y_pred = pipeline.predict(X_test)
            show_results(y_test, y_pred, estimator)
            end = time.time()
            print(score_train)
            print(score_test)
            if score_test > max_score:
                self.pipeline = pipeline

            self.results.append({'pre_estimators':pre_estimators,
                            'estimator':estimator,
                            'score_train':score_train,
                            'score_test':score_test,
                            'execution_time':end-start
                            })

    def read_constructor_json(self):
        for i in self.pipeline_constructor_json['estimators']:
            model_not_found = False

            if self.pipeline_constructor_json['estimators'][i]['model'] == 'RandomForestRegressor':
                model = RandomForestRegressor(n_jobs=-1)
            elif self.pipeline_constructor_json['estimators'][i]['model'] == 'Lasso':
                model = Lasso()
            else:
                model_not_found = True

            if model_not_found:
                print('Unidentfied estimator: ' + self.pipeline_constructor_json['estimators'][i]['model'])
            else: 
                self.estimators.append({
                    'model': model, 
                    'parameters': self.pipeline_constructor_json['estimators'][i]['parameters']
                })

        for i in self.pipeline_constructor_json['pre-estimators']:
            model_not_found = False

            if self.pipeline_constructor_json['pre-estimators'][i]['model'] == 'VarianceThreshold':
                model = VarianceThreshold()
            elif self.pipeline_constructor_json['pre-estimators'][i]['model'] == 'SelectKBest':
                model = SelectKBest()
            else:
                model_not_found = True

            if model_not_found:
                print('Unidentfied pre-estimator: ' + self.pipeline_constructor_json['estimators'][i]['model'])
            else: 
                self.pre_estimators.append({
                    'model': model, 
                    'parameters': self.pipeline_constructor_json['pre-estimators'][i]['parameters']
                })      

