# Predicate, Line, Answer

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


class Line:
  'Linear classifier'
  def __init__(self, lc, cnames):
    assert(len(cnames) == 2)
    self.lc = lc
    self.name = '[%s / %s] LC' % (cnames[0], cnames[1])


  def evaluate(self, X):
    # convert to one-row matrix if single sample
    XX = X[np.newaxis,:] if (len(X.shape) == 1) else X
    ret = self.lc.predict(XX)
    return np.squeeze(ret) if ret.size == 1 else ret


class Answer:
  'One-class answer'
  def __init__(self, answer, answername):
    self.answer = answer
    self.name = '%s' % answername

