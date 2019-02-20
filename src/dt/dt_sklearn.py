# Decision Tree built using scikit learn

import graphviz
import os
import subprocess
from sklearn.metrics import accuracy_score
from sklearn import tree


class DT_sklearn:
  def __init__(self):
    self.dt = tree.DecisionTreeClassifier(
    criterion='entropy', random_state=42, min_impurity_decrease=0.)
    self.ready = False
    self.Xnames = None
    self.Ynames = None


  def fit(self, X, Y):
    self.dt.fit(X, Y)
    self.ready = True


  def fit_ds(self, dataset):
    self.Xnames = dataset.Xnames
    self.Ynames = []
    i = -1
    for name in sorted(dataset.Ynames):
      i += 1
      assert(dataset.Ynames[name] == i)
      self.Ynames.append(name)
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


  def score_ds(self, dataset, normalize=True):
    return self.score(dataset.X, dataset.Y, normalize)


  def is_correct(self, X, Y):
    if not self.ready:
      raise Exception('Tree is not built yet, can not check if correct')
    return (self.score(X, Y, normalize=False) == Y.size)


  def is_correct_ds(self, dataset):
    return self.is_correct(dataset.X, dataset.Y)


  def graph(self, filename, png=False):
    if not self.ready:
      raise Exception('Tree is not built yet, can not create graph')
    if not os.path.exists('results/dot'):
      os.makedirs('results/dot')
    dot_data = tree.export_graphviz(
    self.dt, out_file='results/dot/%s.dot' % filename,
    feature_names=self.Xnames, class_names=self.Ynames, rounded=True)
    if png:
      if not os.path.exists('results/png'):
        os.makedirs('results/png')
      subprocess.check_call(['dot', '-Tpng', '-o',
                             'results/png/%s.png' % filename,
                             'results/dot/%s.dot' % filename])


  def inner_nodes(self):
    if not self.ready:
      return -1
    nc = self.dt.tree_.node_count
    if nc == 0:
      return 0
    assert(nc % 2 == 1) # nc = inner + leaves = 2 * inner + 1
    return (nc - 1) // 2
