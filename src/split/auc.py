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


  def best(self, data):
    self.b_score = 0.
    self.b_pos = -1
    self.b_val = -1
    self.b_eq = None

    for i, ran in enumerate(data.Xranges):
      assert(ran[0] <= ran[1])
      if (ran[1] - ran[0] == 1):
        self.split_score(data, i, ran[1], True)  # [0,1] --> =1
      """
      elif (ran[1] - ran[0] > 1):
        # [0,2] --> =0 =1 =2 <1 <2
        self.split_score(data, i, ran[1], True)
        for val in range(ran[0]+1, ran[1]+1):
          self.split_score(data, i, val, True)
          self.split_score(data, i, val, False)
      """
    assert(self.b_eq is not None)
    return Predicate(fname=data.Xnames[self.b_pos], fpos=self.b_pos,
                     equality=self.b_eq, number=self.b_val)


  def split_score(self, data, pos, value, equality):
    mask = (data.X[:, pos] == value)
    data0X = data.X[mask]
    data0Y = data.Y[mask]
    data1X = data.X[~mask]
    data1Y = data.Y[~mask]
    clf = LinearRegression()

    area0 = 0
    if (data0X.size > 0):
      ysum = np.sum(data0Y)
      if (ysum == 0 or ysum == data0Y.shape[0]): # only one class present
        area0 = 1
      else:
        clf.fit(data0X, data0Y)
        area0 = roc_auc_score(y_true=data0Y, y_score=clf.predict(data0X))

    area1 = 0
    if (data1X.size > 0):
      ysum = np.sum(data1Y)
      if (ysum == 0 or ysum == data1Y.shape[0]): # only one class present
        area1 = 1
      else:
        clf.fit(data1X, data1Y)
        area1 = roc_auc_score(y_true=data1Y, y_score=clf.predict(data1X))

    if (area0 + area1 > self.b_score):
      self.b_score = area0 + area1
      self.b_pos = pos
      self.b_val = value
      self.b_eq = equality

