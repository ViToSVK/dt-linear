# Area under the curve splitting criterion

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, '../dt')

from node_types import Predicate


class Split_auc:
  def __init__(self, clean_boost=False):
    self.b_score = 0.
    self.b_pos = -1
    self.b_val = -1
    self.b_eq = None
    self.feature_mask = []
    self.clean_boost = clean_boost
    self.EPSILON = 0.00000001


  def best(self, data):
    self.b_score = 0.
    self.b_pos = -1
    self.b_val = -1
    self.b_eq = None

    # Create a mask to only include features that
    # do not contain the same value in all samples
    self.feature_mask = []
    for i, dom in enumerate(data.Xdomains):
      assert(len(dom) >= 1)
      self.feature_mask.append(len(dom) > 1)

    # Compute for each predicate
    for i, dom in enumerate(data.Xdomains):
      assert(len(dom) > 0)
      if (len(dom) == 2):
        # {0,1} --> =1
        # {0,8} --> =8
        self.split_score(data, i, max(dom), True)
      elif (len(dom) > 2):
        # {0,1,2} --> =0 =1 =2 <1 (NOT <2 - IT'S FLIPPED =2)
        # {0,3,4} --> =0 =3 =4 <3 (NOT <4 - IT'S FLIPPED =4)
        for val in sorted(dom):
          also_ineq = (i not in data.Xineqforbidden and
                       val != min(dom) and val != max(dom))
          self.split_score(data, i, val, True)
          if (also_ineq):
            self.split_score(data, i, val, False)

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


  def split_score(self, data, pos, value, equality):
    pred = Predicate(fname='', fpos=pos, equality=equality, number=value)
    mask = (pred.evaluate_matrix(data.X))
    sat_X = data.X[mask]
    sat_X = sat_X[:,self.feature_mask]  # Apply feature mask
    sat_Y = data.Y[mask]
    unsat_X = data.X[~mask]
    unsat_X = unsat_X[:,self.feature_mask]  # Apply feature mask
    unsat_Y = data.Y[~mask]
    clf = LinearRegression()

    areaSAT = 0.
    if (sat_X.shape[0] > 0):
      Ys_present = set()
      for y in sat_Y:
        Ys_present.add(y)
      assert(len(Ys_present) > 0)
      if (len(Ys_present) == 1):  # only one class present
        areaSAT = 1. + (0.01 if self.clean_boost else 0.)
      elif (len(Ys_present) == 2):
        clf.fit(sat_X, sat_Y)
        areaSAT = roc_auc_score(y_true=sat_Y, y_score=clf.predict(sat_X))

    areaUNSAT = 0.
    if (unsat_X.shape[0] > 0):
      Ys_present = set()
      for y in unsat_Y:
        Ys_present.add(y)
      assert(len(Ys_present) > 0)
      if (len(Ys_present) == 1):  # only one class present
        areaUNSAT = 1. + (0.01 if self.clean_boost else 0.)
      elif (len(Ys_present) == 2):
        clf.fit(unsat_X, unsat_Y)
        areaUNSAT = roc_auc_score(y_true=unsat_Y, y_score=clf.predict(unsat_X))

    if (areaSAT + areaUNSAT > self.b_score + self.EPSILON):
      self.b_score = areaSAT + areaUNSAT
      self.b_pos = pos
      self.b_val = value
      self.b_eq = equality

