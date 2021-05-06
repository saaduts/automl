import time
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

features, target = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2)

def create_and_evaluate_model(model):
    start = time.time()
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    score_train = r2_score(y_train, y_pred_train)
    print(f'Train Accuracy : {score_train}')
    y_pred_test = model.predict(x_test)
    score_test = r2_score(y_test, y_pred_test)
    print(f'Test Accuracy : {score_test}')
    end = time.time()
    print(f'Execution time: {end-start}')

# print('Random forest')
# random_forest = RandomForestRegressor(criterion='mae', max_depth=10, min_samples_leaf=32,
#                       n_estimators=188, n_jobs=-1)
# create_and_evaluate_model(random_forest)

print('Lesso')
random_lasso = Lasso(alpha=5.741929810894153, fit_intercept=False, normalize=True)
create_and_evaluate_model(random_lasso)
