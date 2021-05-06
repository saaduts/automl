"""
This class will be responsible to receive the preprocessing algorithms and estimator from the controller,
create a dynamic pipeline, train, predict and score the pipeline, communicate back to the controller the results.
The ML Process saves the trained preprocessing algorithms and estimator to be used to re-estimate the model in the test period.

features_train =>
"""

class MLProcess:
  def __init__(self,
               estimator,
               pre_estimators,
               score_func):
      self.estimator = estimator
      self.pre_estimators = pre_estimators
      self.score_func = score_func

  def fit(self, X, y):
      for pre in self.pre_estimators:
          pre.fit(X=X, y=y)
          X = pre.transform(X=X)

      self.estimator.fit(X=X, y=y)

  def predict(self, X):
    for pre in self.pre_estimators:
      X = pre.transform(X=X)

    return self.estimator.predict(X)

  def score(self, X, y):
      for pre in self.pre_estimators:
          X = pre.transform(X=X)
      y_hat = self.estimator.predict(X=X)
      return self.score_func(y_pred=y_hat, y_true=y)