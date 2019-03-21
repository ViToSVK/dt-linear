# Area under the curve splitting criterion

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sys
sys.path.insert(0, '../dt')

from node_types import Predicate


class Split_auc:
  def __init__(self, use_svm=False):
    self.b_score = -1.
    self.b_pos = -1
    self.b_val = -1
    self.b_eq = None
    self.feature_mask = []
    self.use_svm = use_svm
    self.EPSILON = 0.00000001
    self.pos_uses = []


  def best(self, data, node):
    self.b_score = 0.
    self.b_pos = -1
    self.b_val = -1
    self.b_eq = None

    # Create a mask to only include features that
    # do not contain the same value in all samples
    self.feature_mask = []
    self.pos_uses = []
    for i, dom in enumerate(data.Xdomains):
      assert(len(dom) >= 1)
      self.feature_mask.append(len(dom) > 1)
      self.pos_uses.append(0)

    cur = node.parent
    while (cur is not None):
      assert(cur.is_predicate())
      assert(cur.predicate.fpos >= 0 and cur.predicate.fpos < len(self.pos_uses))
      self.pos_uses[cur.predicate.fpos] += 1
      cur = cur.parent

    # Compute for each predicate
    for i, dom in enumerate(data.Xdomains):
      assert(len(dom) > 0)
      if (len(dom) == 2):
        # {0,1} --> =1
        # {0,8} --> =8
        self.split_score(data, i, max(dom), True, dom)
      elif (len(dom) > 2):
        # {0,1,2} --> =0 =1 =2  (<1 IS =0; <2 IS =!2)
        # {0,3,4,6} --> =0 =3 =4 <4 (<3 IS =0; <6 IS =!6)
        for idx, val in enumerate(sorted(dom)):
          assert(idx == 0 or val != min(dom))
          also_ineq = (i not in data.Xineqforbidden and
                       idx >= 2 and val != max(dom))
          if (also_ineq):
            self.split_score(data, i, val, False, dom)
          self.split_score(data, i, val, True, dom)

    # Done; return the best predicate
    assert(self.b_eq is not None)
    assert(self.b_pos >= 0)
    assert(self.b_pos < data.Xnames.size)

    numname = None
    if (data.Xnames[self.b_pos] == 'module'):
      assert(self.b_eq)
      assert(self.b_val in data.ModuleIDtoName)
      numname = data.ModuleIDtoName[self.b_val]
    elif (data.Xnames[self.b_pos] == 'action'):
      assert(self.b_eq)
      assert(self.b_val in data.ActionIDtoName)
      numname = data.ActionIDtoName[self.b_val]

    return Predicate(fname=data.Xnames[self.b_pos], fpos=self.b_pos,
                     equality=self.b_eq, number=self.b_val, numberName=numname)


  def split_score(self, data, pos, value, equality, domain):
    pred = Predicate(fname='', fpos=pos, equality=equality, number=value)
    mask = (pred.evaluate_matrix(data.X))
    sat_Y = data.Y[mask]
    if (sat_Y.size == 0 or sat_Y.size == data.Y.size):
      return
    sat_X = data.X[mask]
    sat_X = sat_X[:,self.feature_mask]  # Apply feature mask
    unsat_X = data.X[~mask]
    unsat_X = unsat_X[:,self.feature_mask]  # Apply feature mask
    unsat_Y = data.Y[~mask]

    reg = LinearRegression()
    clf = None if not self.use_svm else \
          LinearSVC(penalty='l1', tol=0.000001, C=10000.0,
                    dual=False, fit_intercept=True, random_state=42)
    area = {'sat': 0., 'unsat': 0.}

    sides_done = set()
    sides_done_clean = 0
    for (X, Y, name) in [(sat_X, sat_Y, 'sat'), (unsat_X, unsat_Y, 'unsat')]:
      assert(X.shape[0] > 0)
      Ys_present = set()
      for y in Y:
        Ys_present.add(y)
      assert(len(Ys_present) > 0)
      if (len(Ys_present) == 1):  # only one class present
        area[name] = 1.
        sides_done.add(name)
        sides_done_clean += 1
      elif (len(Ys_present) == 2):
        X_tr = StandardScaler().fit_transform(X)
        if (self.use_svm):
          # SVM classifier
          clf.fit(X_tr, Y)
          ac = accuracy_score(normalize=False, y_true=Y,
                              y_pred=clf.predict(X_tr))
          if (ac == Y.size):
            area[name] = 1.
            sides_done.add(name)
          else:
            reg.fit(X_tr, Y)
            area[name] = roc_auc_score(y_true=Y, y_score=reg.predict(X_tr))
        else:
          # Linear regressor
          reg.fit(X_tr, Y)
          area[name] = roc_auc_score(y_true=Y, y_score=reg.predict(X_tr))
      assert(area[name] < 1. + self.EPSILON)

    assert(len(sides_done) <= 2)
    if (len(sides_done) == 2):
      # Making sure solving split wins
      assert(sides_done_clean <= 2)
      if (sides_done_clean == 2):
        # Two clean partitions - best
        area['sat'], area['unsat'] = 4., 4.
      elif (sides_done_clean == 1):
        # One clean one LC partition - second best
        area['sat'], area['unsat'] = 3., 3.
      else:
        # Two LC partitions - third best
        area['sat'], area['unsat'] = 2., 2.
    else:
      # Punishment of obviously unfavourable split
      if (pos not in data.Xineqforbidden):
        assert(pos >= 0 and pos < len(self.pos_uses))
        area['sat'] /= float(1 + self.pos_uses[pos])
        area['unsat'] /= float(1 + self.pos_uses[pos])
      #
      if (equality and (pos not in data.Xineqforbidden) and
          value != min(domain) and value != max(domain)):
        area['sat'] /= 2
        area['unsat'] /= 2
      #
      if (len(domain) > 2):
        sat_dom = 0
        for domv in domain:
          if (equality):
            if (domv == value):
              sat_dom += 1
          else:
            if (domv < value):
              sat_dom += 1
        unsat_dom = len(domain) - sat_dom
        assert(sat_dom >= 1 and unsat_dom >= 1)
        assert((not equality) or sat_dom == 1)
        for name, sz, dm in [('sat', sat_Y.size, sat_dom),
                             ('unsat', unsat_Y.size, unsat_dom)]:
          if (dm > 1 and (name not in sides_done) and
              sz / float(data.Y.size) > 0.99):
            area['sat'] /= float(dm)
            area['unsat'] /= float(dm)

    if (area['sat'] + area['unsat'] > self.b_score + self.EPSILON):
      self.b_score = area['sat'] + area['unsat']
      self.b_pos = pos
      self.b_val = value
      self.b_eq = equality

