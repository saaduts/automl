######To install this package with conda run one of the following:###
## conda install -c conda-forge pysimplegui
## conda install -c conda-forge/label/cf202003 pysimplegui

import json
import PySimpleGUI as sg

from sample_process import main
from visual import show_image

sg.theme('SystemDefaultForReal')

layout = [

    ###################################
    ##Frame Layout for Pre-Estimators##
    ###################################
    [sg.Frame(layout=[
        [sg.Checkbox('Standard Scaler', size=(12, 1), key='standard_scaler')],

        # Train test split
        [sg.Checkbox('Train test split', size=(14, 1)),
            sg.Frame(layout=[[
                         sg.Text('Test Size'),
                         sg.Slider(
                             range=(1, 100),
                             orientation='h',
                             size=(14, 20),
                             default_value=20,
                             key='test_size'),
                         ]]
                         ,title='Set test size (percentage)', title_color='red', relief=sg.RELIEF_SUNKEN)
         ],

        # Pipeline selection
        [sg.Checkbox('Pipeline selection', size=(14, 1)),
            sg.Frame(layout=[[
                         sg.Text('Pipeline'),
                         sg.Slider(
                             range=(1, 100),
                             orientation='h',
                             size=(14, 20),
                             default_value=5,
                             key='no_of_pipeline'),
                         ]]
                         ,title='Select number of pipeline', title_color='red', relief=sg.RELIEF_SUNKEN)
         ],

        [sg.Checkbox('Normalizer', size=(14, 1), key='normalaizer'),
         sg.Frame(layout=[[sg.Checkbox('l1', size=(3, 1), key='l1'),
                           sg.Checkbox('l2', default=True, key='l2'),
                           sg.Checkbox('max', key='normalaizer_max')]]
                  , title='norm', title_color='red', relief=sg.RELIEF_SUNKEN)],

        [sg.Checkbox('SimpleImputer', size=(14, 1), key='simple_imputer'),
         sg.Frame(layout=[[sg.Checkbox('mean', key='mean'),
                           sg.Checkbox('median', default=True, key='median'),
                           sg.Checkbox('most_frequent', key='most_frequent'),
                           sg.Checkbox('constant', key='constant')]]
                  , title='strategy', title_color='red', relief=sg.RELIEF_SUNKEN)],

        [sg.Checkbox('MinMaxScaler', size=(14, 1), key='min_max_scaler'),
         sg.Frame(layout=[[
             sg.Text('min', key='min_max_scaler_min'),
             sg.Slider(range=(1, 100), orientation='h', size=(14, 20), default_value=1, key='min_max_scaler_min_range'),
             sg.Text('min_max_caler_max'), sg.Slider(range=(1, 100), orientation='h', size=(14, 20), default_value=100,
                                                     key='min_max_scaler_max_range')]]
             , title='MinMax Value', title_color='red', relief=sg.RELIEF_SUNKEN)],

        [sg.Checkbox('Binarization', size=(14, 1), key='binarization'),
         sg.Frame(layout=[[
             sg.Text('min', key='binarization_min'),
             sg.Slider(range=(1, 100), orientation='h', size=(14, 20), default_value=1, key='binarization_min_range'),
             sg.Text('max', key='binarization_max'),
             sg.Slider(range=(1, 100), orientation='h', size=(14, 20), key='binarization_max_range',
                       default_value=100)]]
             , title='threshold', title_color='red', relief=sg.RELIEF_SUNKEN)],

        [sg.Checkbox('VarianceThreshold', size=(14, 1), key='variance_threshold'),
         sg.Frame(layout=[[
             sg.Text('min', key='variance_threshold_min'),
             sg.Slider(range=(0, 10), orientation='h', size=(14, 20), default_value=0,
                       key='variance_threshold_min_range'),
             sg.Text('max', key='variance_threshold_max'),
             sg.Slider(range=(0, 10), orientation='h', size=(14, 20), default_value=10,
                       key='variance_threshold_max_range')]]
             , title='threshold', title_color='red', relief=sg.RELIEF_SUNKEN)],

        [sg.Checkbox('SelectKBest', size=(14, 1), key='select_k_best'),
         sg.Frame(layout=[[
             sg.Text('min', key='select_k_best_min'),
             sg.Slider(range=(1, 8), orientation='h', size=(14, 20), default_value=1, key='select_k_best_min_range'),
             sg.Text('max', key='select_k_best_max'),
             sg.Slider(range=(1, 8), orientation='h', size=(14, 20), default_value=8,
                       key='select_k_best_max_range')]]
             , title='k feat', title_color='red', relief=sg.RELIEF_SUNKEN)]
    ], title='Pre-Estimators', title_color='Blue', relief=sg.RELIEF_SUNKEN,
        tooltip='Use these to choose pre-estimators', key='pre_estimator'),

        ###############################
        ##Frame Layout for Estimators##
        ###############################
        sg.Frame(layout=[
            [sg.Checkbox('RandomForestRegressor', size=(19, 1), key='random_forest_regressor')],
            [sg.Frame(layout=[[sg.Text('n_estimators:       ', key='n_estimators'),
                               sg.Text('min', key='n_estimators_min'),
                               sg.Slider(range=(1, 1000), orientation='h', size=(14, 20), default_value=1,
                                         key='n_estimators_min_range'),
                               sg.Text('max', key='n_estimators_max'),
                               sg.Slider(range=(1, 1000), orientation='h', size=(14, 20), default_value=1000,
                                         key='n_estimators_max_range')
                               ],

                              [sg.Text('criterion:              ', key='criterion'),
                               sg.Checkbox('mse', key='mse'),
                               sg.Checkbox('mae', default=True, key='mae')
                               ],

                              [sg.Text('max_depth:          ', key='max_depth'),
                               sg.Text('min', key='max_depth_min'),
                               sg.Slider(range=(2, 13), orientation='h', size=(14, 20), default_value=2,
                                         key='max_depth_min_range'),
                               sg.Text('max', key='max_depth_max'),
                               sg.Slider(range=(2, 13), orientation='h', size=(14, 20), default_value=13,
                                         key='max_depth_max_range')
                               ],

                              [sg.Text('min_samples_leaf:', key='min_samples_leaf'),
                               sg.Text('min', key='min_samples_leaf_min'),
                               sg.Slider(range=(1, 250), orientation='h', size=(14, 20), default_value=1,
                                         key='min_samples_leaf_min_range'),
                               sg.Text('max', key='min_samples_leaf_max'),
                               sg.Slider(range=(1, 250), orientation='h', size=(14, 20), default_value=250,
                                         key='min_samples_leaf_max_range')
                               ],

                              [sg.Text('max_features:       ', key='max_features'),
                               sg.Checkbox('auto', default=True, key='auto'),
                               sg.Checkbox('sqrt', key='sqrt'),
                               sg.Checkbox('log2', key='log2')
                               ]
                              ]

                      , title='RandomForestRegressor Parameters', title_color='red', relief=sg.RELIEF_SUNKEN,
                      key='random_forest_regressor_params')],

            [sg.Checkbox('Lasso', size=(14, 1), key='lasso')],
            [sg.Frame(layout=[[sg.Text('Alpha :         ', key='alpha'),
                               sg.Text('min', key='alpha_min'),
                               sg.Slider(range=(1, 100), orientation='h', size=(14, 20), default_value=1,
                                         key='alpha_min_range'),
                               sg.Text('max', key='alpha_max'),
                               sg.Slider(range=(1, 100), orientation='h', size=(14, 20), default_value=100,
                                         key='alpha_max_range')],

                              [sg.Text('fit intercept : ', key='fit_intercept'),
                               sg.Checkbox('True', default=True, key='fit_intercept_true'),
                               sg.Checkbox('False', key='fit_intercept_false')],

                              [sg.Text('normalize :   ', key='normalize'),
                               sg.Checkbox('True', default=True, key='normalize_true'),
                               sg.Checkbox('False', key='normalize_false')]]
                      , title='Parameter', title_color='red', relief=sg.RELIEF_SUNKEN, key='parameter')],

        ], title='Estimators', title_color='Blue', relief=sg.RELIEF_SUNKEN,
            tooltip='Use these to choose pre-estimators',
            key='estimators')
    ],

    [sg.Submit(tooltip='Click to submit this form', key='submit'), sg.Cancel(key='cancel')]]

window = sg.Window('Machine Learning Process', layout, default_element_size=(40, 1), grab_anywhere=False)
event, values = window.read()
window.close()
print(json.dumps(values, indent=4))
pop = sg.Popup('Title',
               'The results of the window.',
               'The button clicked was "{}"'.format(event),
               'The values are', values)


criterion = []
if values['mse']:
    criterion.append('mse')
if values['mae']:
    criterion.append('mae')

max_features = []
if values['auto']:
    max_features.append('auto')
if values['sqrt']:
    max_features.append('sqrt')
if values['log2']:
    max_features.append('log2')

selections = {
    'test_size' : int(values['test_size']),
    'no_of_pipeline' : int(values['no_of_pipeline']),
  "estimators": {
    "0": {
      "model": "RandomForestRegressor",
      "parameters": {
        "0": {
          "name": "n_estimators",
          "values": {
            "type": "int",
            "min": values['n_estimators_min_range'],
            "max": values['n_estimators_max_range']
          }
        },
        "1": {
          "name": "criterion",
          "values": {
            "type": "list",
            "list": criterion
          }
        },
        "2": {
          "name": "max_depth",
          "values": {
            "type": "int",
            "min": values['max_depth_min_range'],
            "max": values['max_depth_max_range']
          }
        },
        "3": {
          "name": "min_samples_leaf",
          "values": {
            "type": "int",
            "min": values['min_samples_leaf_min_range'],
            "max": values['min_samples_leaf_max_range']
          }
        },
        "4": {
          "name": "max_features",
          "values": {
            "type": "list",
            "list": max_features
          }
        }
      }
    },
    "1": {
      "model": "Lasso",
      "parameters": {
        "0": {
          "name": "alpha",
          "values": {
            "type": "float",
            "min": values['alpha_min_range'],
            "max": values['alpha_max_range']
          }
        },
        "1": {
          "name": "fit_intercept",
          "values": {
            "type": "boolean"
          }
        },
        "2": {
          "name": "normalize",
          "values": {
            "type": "boolean"
          }
        }
      }
    }
  },
  "pre-estimators": {
    "0": {
      "model": "VarianceThreshold",
      "parameters": {
        "0": {
          "name": "threshold",
          "values": {
            "type": "float",
            "min": values['variance_threshold_min_range'],
            "max": values['variance_threshold_min_range']
          }
        }
      }
    },
    "1": {
      "model": "SelectKBest",
      "parameters": {
        "0": {
          "name": "k",
          "values": {
            "type": "int",
            "min": values['select_k_best_min_range'],
            "max": values['select_k_best_max_range']
          }
        }
      }
    }
  }
}
if pop == 'OK':
    path = main(selections)
    show_image(path)
