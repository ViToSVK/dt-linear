# Decision Tree with linear classifiers

from asyncio import Queue
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sys
sys.path.insert(0, 'src/parsing')

from dataset import Dataset
from decision_tree import Decision_tree
from node import Node
import node_types


class DT_linear(Decision_tree):
  'Subclass that uses linear classifiers at the leaves'
  def __init__(self, split_criterion, use_lc=True):
    super(DT_linear, self).__init__()
    self.split_criterion = split_criterion
    self.use_lc = use_lc


  def fit(self, dataset):
    nodes_at_start = Node.totalcount
    self.Xnames = dataset.Xnames
    self.Ynames = dataset.Ynames
    self.root = Node(dataset, parent=None)
    self.lc_nodes = 0

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

      if len(c.data.Ynames) == 2 and self.use_lc:
        # Possible to crate a linear classifier here
        # Conjunction (penalty='l1' loss='squared_hinge' dual=True) not supported
        lc = LinearSVC(penalty='l1', tol=0.000001, C=10000.0,
                       dual=False, fit_intercept=True, random_state=42)
        # Create a mask to only include features that
        # do not contain the same value in all samples
        feature_mask = []
        for i, rnge in enumerate(c.data.Xranges):
          assert(rnge[0] <= rnge[1])
          feature_mask.append(rnge[0] < rnge[1])
        # Create a scaler to transform data before fitting
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(c.data.X[:,feature_mask])
        lc.fit(X_transformed, c.data.Y)
        sc = accuracy_score(normalize=False, y_true=c.data.Y,
                            y_pred=lc.predict(X_transformed))
        if sc == c.data.Y.size:
          # Linear classifier is correct, make a leaf node with it
          c.line = node_types.Line(lc, feature_mask, scaler,
                                   list(c.data.Ynames.keys()), c.data.Y.size)
          self.lc_nodes += 1
          continue

      # We need to split here
      c.predicate = self.split_criterion.best(c.data)
      mask, s_Xranges, u_Xranges = c.predicate.evaluate_ranges(c.data.X)
      Ynames_back = {}
      for name, idx in c.data.Ynames.items():
        assert(idx not in Ynames_back)
        Ynames_back[idx] = name
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
      c.childSAT = Node(Dataset(s_X, s_Y, s_Xnames, s_Xranges, s_Ynames,
                                c.data.Xineqforbidden.copy(), c.data.ActionIDtoName.copy()),
                        c)
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
      c.childUNSAT = Node(Dataset(u_X, u_Y, u_Xnames, u_Xranges, u_Ynames,
                                  c.data.Xineqforbidden.copy(), c.data.ActionIDtoName.copy()),
                          c)
      que.put_nowait(c.childUNSAT)
      # Clean up previous dataset
      c.data = None
      assert(dataset is not None) # c.data was pointer copy, not reference

    # Finished
    self.nodes = Node.totalcount - nodes_at_start

