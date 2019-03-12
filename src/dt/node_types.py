# Predicate, Line, Answer

import numpy as np


class Predicate:
  'Equality or inequality predicate'
  def __init__(self, fname, fpos, equality, number, numberName = None):
    assert(fpos >= 0)
    self.fpos = fpos
    self.equality = equality
    self.number = number
    self.name = '%s %s %s' % (fname, ('=' if equality else '<'),
                              str(number) if numberName is None else numberName)


  def evaluate_sample(self, sample):
    assert(self.fpos < len(sample))
    if self.equality:
      return (sample[self.fpos] == float(self.number))
    else:
      return (sample[self.fpos] < float(self.number))


  def evaluate_matrix(self, X):
    return self.evaluate_sample(np.transpose(X))


  def evaluate_domains(self, X):
    assert(len(X.shape) == 2)
    sat_Xdomains = [set() for _ in range(X.shape[1])]
    unsat_Xdomains = [set() for _ in range(X.shape[1])]
    mask = []
    for sample_float in X:
      res = self.evaluate_sample(sample_float)
      mask.append(res)
      sample = sample_float.astype(int)
      if res:
        for idx, val in enumerate(sample):
          assert(idx >= 0 and idx < X.shape[1])
          sat_Xdomains[idx].add(val)
      else:
        for idx, val in enumerate(sample):
          assert(idx >= 0 and idx < X.shape[1])
          unsat_Xdomains[idx].add(val)
    return np.array(mask), sat_Xdomains, unsat_Xdomains


class Line:
  'Linear classifier'
  def __init__(self, lc, feature_mask, scaler, cnames, sample_no):
    assert(len(cnames) == 2)
    self.lc = lc
    self.feature_mask = feature_mask
    self.scaler = scaler
    self.name = '[%s / %s] LC (%d)' % (cnames[0], cnames[1], sample_no)
    self.sample_no = sample_no


  def evaluate(self, X):
    # convert to one-row matrix if single sample
    XX = X[np.newaxis,:] if (len(X.shape) == 1) else X
    # apply feature mask
    XX = XX[:,self.feature_mask]
    XX_trans = self.scaler.transform(XX)
    ret = self.lc.predict(XX_trans)
    return np.squeeze(ret) if ret.size == 1 else ret


class Answer:
  'One-class answer'
  def __init__(self, answer, answername, sample_no):
    self.answer = answer
    self.name = '%s (%d)' % (answername, sample_no)
    self.sample_no = sample_no


  def evaluate(self):
    return int(self.answer)

