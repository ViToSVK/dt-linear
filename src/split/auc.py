# Area under the curve splitting criterion

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, '../dt')

from node_types import Predicate


class Split_auc:
  def __init__(self):
    self.b_score = 0.
    self.b_pos = -1
    self.b_val = -1
    self.b_eq = None
    self.EPSILON = 0.00000001


  def best(self, data):
    self.b_score = 0.
    self.b_pos = -1
    self.b_val = -1
    self.b_eq = None

    for i, ran in enumerate(data.Xranges):
      assert(ran[0] <= ran[1])
      if (ran[1] - ran[0] == 1):
        self.split_score(data, i, ran[1], True)  # [0,1] --> =1
      elif (ran[1] - ran[0] > 1):
        # [0,2] --> =0 =1 =2 <1 <2
        self.split_score(data, i, ran[1], True)
        for val in range(ran[0]+1, ran[1]+1):
          self.split_score(data, i, val, True)
          if (i not in data.Xineqforbidden):
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
    sat_Y = data.Y[mask]
    unsat_X = data.X[~mask]
    unsat_Y = data.Y[~mask]
    clf = LinearRegression()

    areaSAT = 0.
    if (sat_X.size > 0):
      Ys_present = set()
      for y in sat_Y:
        Ys_present.add(y)
      assert(len(Ys_present) > 0)
      if (len(Ys_present) == 1):  # only one class present
        areaSAT = 1.
      elif (len(Ys_present) == 2):
        clf.fit(sat_X, sat_Y)
        areaSAT = roc_auc_score(y_true=sat_Y, y_score=clf.predict(sat_X))

    areaUNSAT = 0.
    if (unsat_X.size > 0):
      Ys_present = set()
      for y in unsat_Y:
        Ys_present.add(y)
      assert(len(Ys_present) > 0)
      if (len(Ys_present) == 1):  # only one class present
        areaUNSAT = 1.
      elif (len(Ys_present) == 2):
        clf.fit(unsat_X, unsat_Y)
        areaUNSAT = roc_auc_score(y_true=unsat_Y, y_score=clf.predict(unsat_X))

    if (areaSAT + areaUNSAT > self.b_score + self.EPSILON):
      self.b_score = areaSAT + areaUNSAT
      self.b_pos = pos
      self.b_val = value
      self.b_eq = equality

