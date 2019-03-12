# Decision Tree base class

import numpy as np
import os
import subprocess


class Decision_tree:
  def __init__(self):
    self.root = None
    self.nodes = -1
    self.lc_nodes = -1
    self.Xnames = None
    self.Ynames = None


  def ready(self):
    return (self.root is not None)


  def fit(self, dataset):
    pass


  def fit_ds(self, dataset):
    self.fit(dataset)


  def predict(self, X):
    if not self.ready():
      raise Exception('Tree is not built yet, can not predict')
    y = []
    # convert to one-row matrix if single sample
    XX = X[np.newaxis,:] if (len(X.shape) == 1) else X
    for sample in XX:
      c = self.root
      while not c.is_leaf():
        assert(c.is_predicate())
        if c.predicate.evaluate_sample(sample):
          c = c.childSAT
        else:
          c = c.childUNSAT
      if c.is_line():
        y.append(c.line.evaluate(sample))
      else:
        assert(c.is_answer())
        y.append(c.answer.evaluate())
    return y[0] if len(y) == 1 else np.array(y)


  def predict_ds(self, dataset):
    return self.predict(dataset.X)


  def is_correct(self, X, Y):
    if not self.ready():
      raise Exception('Tree is not built yet, can not check if correct')
    return (np.array_equal(self.predict(X), Y))


  def is_correct_ds(self, dataset):
    return self.is_correct(dataset.X, dataset.Y)


  def graphhelp(node, sat=True):
    result = []
    if not node.is_root():
      result.append('N%s -> N%s [style=%s]\n' %
                    (node.parent.id, node.id, 'solid' if sat else 'dashed'))
    if node.is_leaf():
      result.append('N%s [label=\"%s\" shape=box style=filled]\n' %
                    (node.id, node.name()))
    else:
      result.append('N%s [label=\"%s\"]\n' %
                    (node.id, node.name()))
      result.append(Decision_tree.graphhelp(node.childSAT, sat=True))
      result.append(Decision_tree.graphhelp(node.childUNSAT, sat=False))
    ret = ''.join(result)
    return ret


  def graph(self, filename, png=False):
    if not self.ready:
      raise Exception('Tree is not built yet, can not create graph')
    if not os.path.exists('results/dot'):
      os.makedirs('results/dot')
    fname = filename.replace('.arff', '').replace('.prism', '')
    f = open('results/dot/%s.dot' % fname,'w')
    f.write('digraph DecisionTree {\n')
    f.write(Decision_tree.graphhelp(self.root))
    f.write('}\n')
    f.close()
    if png:
      if not os.path.exists('results/png'):
        os.makedirs('results/png')
      subprocess.check_call(['dot', '-Tpng', '-o',
                             'results/png/%s.png' % fname,
                             'results/dot/%s.dot' % fname])


  def inner_nodes(self):
    if not self.ready():
      return -1
    nc = self.nodes
    if nc == 0:
      return 0
    assert(nc % 2 == 1) # nc = inner + leaves = 2 * inner + 1
    return (nc - 1) // 2


  def inner_and_lc_nodes(self):
    return self.inner_nodes() + self.lc_nodes


  def depth_help(self, node):
    if (node.is_line()):
      return node.level * node.line.sample_no, node.line.sample_no
    elif (node.is_answer()):
      return node.level * node.answer.sample_no, node.answer.sample_no
    else:
      assert(not node.is_leaf())
      c1, n1 = self.depth_help(node.childSAT)
      c2, n2 = self.depth_help(node.childUNSAT)
      return c1+c2, n1+n2


  def weighted_avg_depth(self):
    if not self.ready:
      raise Exception('Tree is not built yet, can not calculate depth')
    [cumulative_depth, number_of_samples] = self.depth_help(self.root)
    return float(cumulative_depth) / float(number_of_samples)

