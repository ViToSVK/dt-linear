# Decision Tree built using scikit learn

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class DT_sklearn:
    def __init__(self):
      self.dt = DecisionTreeClassifier(
      criterion='entropy', random_state=42, min_impurity_decrease=0.)
      self.ready = False


    def fit(self, X, Y):
      self.dt.fit(X, Y)
      self.ready = True


    def fit_ds(self, dataset):
      self.fit(dataset.X, dataset.Y)


    def predict(self, X):
      if not self.ready:
        raise Exception('Tree is not built yet, can not predict')
      return self.dt.predict(X)


    def predict_ds(self, dataset):
      return self.predict(dataset.X)


    def score(self, X, Y, normalize=True):
      if not self.ready:
        raise Exception('Tree is not built yet, can not score')
      return accuracy_score(Y, self.predict(X), normalize)
      return self.dt.score(X, Y)


    def score_ds(self, dataset, normalize=True):
      return self.score(dataset.X, dataset.Y, normalize)


    def is_correct(self, X, Y):
      if not self.ready:
        raise Exception('Tree is not built yet, can not check if correct')
      return (self.score(X, Y, normalize=False) == Y.size)


    def is_correct_ds(self, dataset):
      return self.is_correct(dataset.X, dataset.Y)


    def inner_nodes(self):
      if not self.ready:
        return -1
      nc = self.dt.tree_.node_count
      if nc == 0:
        return 0
      assert(nc % 2 == 1) # leaves = 2 * inner - 1
      return (nc - 1) // 2
