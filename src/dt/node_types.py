# Predicate, Line, Answer

import numpy as np


class Predicate:
  'Equality or inequality predicate'
  def __init__(self, fname, fpos, equality, number):
    assert(fpos >= 0)
    self.fpos = fpos
    self.equality = equality
    self.number = number
    self.name = '%s %s %d' % (fname, ('=' if equality else '<'), number)


  def evaluate(self, sample):
    assert(self.fpos < len(sample))
    if self.equality:
      return (sample[self.fpos] == self.number)
    else:
      return (sample[self.fpos] < self.number)


  def evaluate_ranges(self, X):
    assert(len(X.shape) == 2)
    sat_min, sat_max, unsat_min, unsat_max = None, None, None, None
    mask = []
    for sample_float in X:
      res = self.evaluate(sample_float)
      mask.append(res)
      sample = sample_float.astype(int)
      if res:
        if sat_min is None:
          sat_min = sample
          sat_max = sample
        else:
          sat_min = np.minimum(sat_min, sample)
          sat_max = np.maximum(sat_max, sample)
      else:
        if unsat_min is None:
          unsat_min = sample
          unsat_max = sample
        else:
          unsat_min = np.minimum(unsat_min, sample)
          unsat_max = np.maximum(unsat_max, sample)
    sat_Xranges, unsat_Xranges = [], []
    for i in range(sat_min.size):
      sat_Xranges.append([sat_min[i], sat_max[i]])
      unsat_Xranges.append([unsat_min[i], unsat_max[i]])
    return np.array(mask), np.array(sat_Xranges), np.array(unsat_Xranges)


class Line:
  'Linear classifier'
  def __init__(self, lc, feature_mask, cnames, sample_no):
    assert(len(cnames) == 2)
    self.lc = lc
    self.feature_mask = feature_mask
    self.name = '[%s / %s] LC (%d)' % (cnames[0], cnames[1], sample_no)


  def evaluate(self, X):
    # convert to one-row matrix if single sample
    XX = X[np.newaxis,:] if (len(X.shape) == 1) else X
    # apply feature mask
    XX = XX[:,self.feature_mask]
    ret = self.lc.predict(XX)
    return np.squeeze(ret) if ret.size == 1 else ret


class Answer:
  'One-class answer'
  def __init__(self, answer, answername, sample_no):
    self.answer = answer
    self.name = '%s (%d)' % (answername, sample_no)

  def evaluate(self):
    return int(self.answer)

