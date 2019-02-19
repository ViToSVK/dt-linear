# Decision Tree base class

import os


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


  def classify(self, X):
    y = []
    # convert to one-row matrix if single sample
    XX = X[np.newaxis,:] if (len(X.shape) == 1) else X
    for sample in XX:
      c = self.root
      while not c.is_leaf():
        if c.evaluate(sample):
          c = c.childR
        else:
          c = c.childL
      y.append(c.evaluate(sample) if c.is_line() else c.answer)
    return y[0] if len(y) == 1 else y


  def graphhelp(node):
    result = []
    if node.is_leaf():
      result.append('N%s [label=\"%s\" shape=box style=filled]\n' %
                    (node.id, node.name()))
    else:
      result.append('N%s [label=\"%s\"]\n' %
                    (node.id, node.name()))
      result.append('N%s -> N%s [style=dashed]\n' %
                    (node.parent.id, node.childL.id))
      result.append(DecisionTree.graphhelp(node.childL))
      result.append('N%s -> N%s [style=solid]\n' %
                    (node.parent.id, node.childR.id))
      result.append(DecisionTree.graphhelp(node.childR))
    ret = ''.join(result)
    return ret


  def graph(self, filename):
    if not os.path.exists('results/dot'):
      os.makedirs('results/dot')
    f = open('results/dot/%s.dot' % filename,'w')
    f.write('digraph DecisionTree {\n')
    f.write(DecisionTree.graphhelp(self.root))
    f.write('}\n')
    f.close()

