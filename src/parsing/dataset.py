# Dataset class

class Dataset:
  def __init__(self, X, Y, Xnames, Xranges, Ynames, Xineqforbidden, ActionIDtoName):
    self.X = X  # numpy2D(float)
    self.Y = Y  # numpy1D(int)
    self.Xnames = Xnames  # list[string]
    self.Xranges = Xranges  # list[int-lowerbound,int-upperbound]
    self.Ynames = Ynames  # map[string->int]
    self.Xineqforbidden = Xineqforbidden  # set[int]
    self.ActionIDtoName = ActionIDtoName  # map[int->string]


  def dump(self, short=False):
    if (short):
      print('%d samples, %d features' % (self.X.shape[0], self.X.shape[1]))
      return
    assert(self.Y.size == self.X.shape[0])
    print('%d samples that have %d features. First 3 samples:'
          % (self.X.shape[0], self.X.shape[1]))
    print(self.X[:3])
    print(self.Y[:3])
    print('Names of features:')
    print(self.Xnames)
    print('Value ranges for features:')
    for i, ran in enumerate(self.Xranges):
      if i > 0 and not i % 10:
        print()
      print(ran, end=' ' if i % 10 != 9 else '')
    print()
    print('Names of classes occuring:')
    print(self.Ynames)
    if (len(self.Xineqforbidden)):
      print('Features that have inequality predicates forbidden:')
      for i in self.Xineqforbidden:
        print('%s (id %d)' % (self.Xnames[i], i))

