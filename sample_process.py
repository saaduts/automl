from Controller import Controller
from datetime import datetime
import json
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from result_dir import make_dir_if_not_exists
from settings import DIR_RESULT, DIR_REPORT


def main(selections):
    # file_path = 'Boston.csv'
    # dataset = pd.read_csv(file_path, index_col=False)
    # show_data(dataset)
    # features = dataset.drop(labels=['medv'], axis=1)
    # features.replace({'?':0}, inplace=True)
    # target = dataset['medv']

    test_size = selections['test_size']/100

    features, target = load_boston(return_X_y=True)

    score_func = r2_score
    with open('constructor.json', 'r') as fp:
        constructor_json = json.load(fp)
    print(json.dumps(selections, indent=4))
    c = Controller(features=features,
                   target=target,
                    test_size = test_size,
                   score_func=score_func,
                   pipeline_constructor_json=selections,
                   pipelines_to_test=selections['no_of_pipeline'])

    results_df = pd.DataFrame(c.results)
    report_path = f'{DIR_RESULT}/{DIR_REPORT}/'
    make_dir_if_not_exists(report_path)
    results_df.to_csv(f"{report_path}result.csv")

    return DIR_RESULT