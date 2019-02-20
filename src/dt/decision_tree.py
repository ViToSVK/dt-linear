# Decision Tree base class

import os
import subprocess


class Decision_tree:
  def __init__(self):
    self.root = None
    self.nodes = -1
    self.Xnames = None
    self.Ynames = None


  def ready(self):
    return (root is not None)


  def fit(self, dataset):
    pass


  def predict(self, X):
    if not self.ready():
      raise Exception('Tree is not built yet, can not predict')
    y = []
    # convert to one-row matrix if single sample
    XX = X[np.newaxis,:] if (len(X.shape) == 1) else X
    for sample in XX:
      c = self.root
      while not c.is_leaf():
        if c.evaluate(sample):
          c = c.childSAT
        else:
          c = c.childUNSAT
      y.append(c.evaluate(sample) if c.is_line() else c.answer)
    return y[0] if len(y) == 1 else y


  def predict_ds(self, dataset):
    return self.predict(dataset.X)


  def is_correct(self, X, Y):
    if not self.ready():
      raise Exception('Tree is not built yet, can not check if correct')
    return (self.predict(X) == Y)


  def is_correct_ds(self, dataset):
    return self.is_correct(dataset.X, dataset.Y)


  def graphhelp(node):
    result = []
    if node.is_leaf():
      result.append('N%s [label=\"%s\" shape=box style=filled]\n' %
                    (node.id, node.name()))
    else:
      result.append('N%s [label=\"%s\"]\n' %
                    (node.id, node.name()))
      result.append('N%s -> N%s [style=solid]\n' %
                    (node.parent.id, node.childSAT.id))
      result.append(DecisionTree.graphhelp(node.childSAT))
      result.append('N%s -> N%s [style=dashed]\n' %
                    (node.parent.id, node.childUNSAT.id))
      result.append(DecisionTree.graphhelp(node.childUNSAT))
    ret = ''.join(result)
    return ret


  def graph(self, filename, png=False):
    if not self.ready:
      raise Exception('Tree is not built yet, can not create graph')
    if not os.path.exists('results/dot'):
      os.makedirs('results/dot')
    f = open('results/dot/%s.dot' % filename,'w')
    f.write('digraph DecisionTree {\n')
    f.write(DecisionTree.graphhelp(self.root))
    f.write('}\n')
    f.close()
    if png:
      if not os.path.exists('results/png'):
        os.makedirs('results/png')
      subprocess.check_call(['dot', '-Tpng', '-o',
                             'results/png/%s.png' % filename,
                             'results/dot/%s.dot' % filename])


  def inner_nodes(self):
    if not self.ready():
      return -1
    nc = self.nodes
    if nc == 0:
      return 0
    assert(nc % 2 == 1) # nc = inner + leaves = 2 * inner + 1
    return (nc - 1) // 2
