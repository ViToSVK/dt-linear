# Dataset class

class Dataset:
  def __init__(self, X, Y, Xnames, Xranges, Ynames):
    self.X = X
    self.Y = Y
    self.Xnames = Xnames
    self.Xranges = Xranges
    self.Ynames = Ynames

  def dump(self):
    assert(self.Y.size == self.X.shape[0])
    print("%d samples that have %d features. First 3 samples:"
          % (self.X.shape[0], self.X.shape[1]))
    print(self.X[:3])
    print(self.Y[:3])
    print("Names of features:")
    print(self.Xnames)
    print("Value ranges for features:")
    for i,ran in enumerate(self.Xranges):
      if i > 0 and not i % 10:
        print()
      print(ran,end=' ' if i % 10 != 9 else '')
    print()
    print("Names of classes occuring:")
    print(self.Ynames)
