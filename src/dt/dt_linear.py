# Decision Tree with linear classifiers

from asyncio import Queue
from decision_tree import Decision_tree
from node import Node
import node_types
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


class DT_linear(Decision_tree):
  'Subclass that uses linear classifiers at the leaves'
  def __init__(self, split_criterion):
    super(DT_linear, self).__init__()
    self.split_criterion = split_criterion


  def fit(self, dataset):
    nodes_at_start = Node.totalcount
    self.Xnames = dataset.Xnames
    self.Ynames = dataset.Ynames
    self.root = Node(dataset, parent=None)

    que = Queue()
    que.put_nowait(self.root)

    while not que.empty():
      c = que.get_nowait()

      if len(c.data.Ynames) == 1:
        # One class remaining, create an answer node
        for answername, answer in c.data.Ynames.items():
          assert(c.answer is None)
          c.answer = node_types.Answer(answer, answername, c.data.Y.size)
        continue

      if len(c.data.Ynames) == 2:
        # Possible to crate a linear classifier here
        lc = svm.LinearSVC(penalty='l1', tol=0.000001, C=10000.0,
                           dual=(c.data.X.shape[0] < c.data.X.shape[1]),
                           fit_intercept=True, random_state=42)
        # Create a column filter to only include features
        # that do not contain the same value in all samples
        feature_filter = []
        for i, rnge in enumerate(c.data.Xranges):
          assert(rnge[0] <= rnge[1])
          if rnge[0] < rnge[1]:
            feature_filter.add(i)
        lc.fit(c.data.X[:,feature_filter], c.data.Y)
        sc = accuracy_score(normalize=False, y_true=c.data.Y
                            y_pred=lc.predict(c.data.X[:,feature_filter]))
        if sc == c.data.Y.size:
          # Linear classifier is correct, make a leaf node with it
          c.answer = node_types.Line(lc, feature_filter,
                                     list(c.data.Ynames.keys()), c.data.Y.size)
          continue

      # We need to split here
      c.predicate = self.split_criterion(c.data)
      mask, sat_Xranges, unsat_Xranges = c.predicate.evaluate_ranges(c.data.X)
      Ynames_back = {}
      for name, idx in c.data.Ynames.items():
        assert(idx not in Ynames_back)
        Ynames_back[idx] = name
      for
      # SAT
      s_X = c.data.X[mask]
      s_Y = c.data.Y[mask]
      s_Xnames = c.data.Xnames
      s_Ynames = {}
      s_Yids = set()
      for y in s_Y:
        if y not in s_Yids:
          s_Yids.add(y)
          assert(y in Ynames_back and Ynames_back[y] not in s_Ynames)
          s_Ynames[Ynames_back[y]] = y
      c.childSAT = Node(Dataset(s_X, s_Y, s_Xnames, s_Xranges, s_Ynames), c)
      que.put_nowait(c.childSAT)
      # UNSAT
      u_X = c.data.X[~mask]
      u_Y = c.data.Y[~mask]
      u_Xnames = c.data.Xnames
      u_Ynames = {}
      u_Yids = set()
      for y in u_Y:
        if y not in u_Yids:
          u_Yids.add(y)
          assert(y in Ynames_back and Ynames_back[y] not in u_Ynames)
          u_Ynames[Ynames_back[y]] = y
      c.childUNSAT = Node(Dataset(u_X, u_Y, u_Xnames, u_Xranges, u_Ynames), c)
      que.put_nowait(c.childUNSAT)
      # Clean up previous dataset
      c.data = None
      assert(dataset is not None) # c.data was pointer copy, not reference

    # Finished
    self.nodes = Node.totalcount - nodes_at_start



