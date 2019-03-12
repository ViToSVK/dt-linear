# Dataset class

class Dataset:
  def __init__(self, X, Y, Xnames, Xdomains, Ynames,
               Xineqforbidden=set(), ActionIDtoName={}, ModuleIDtoName={}):
    self.X = X  # numpy2D(float)
    self.Y = Y  # numpy1D(int)
    self.Xnames = Xnames  # list[string]
    self.Xdomains = Xdomains  # set[int]
    self.Ynames = Ynames  # map[string->int]
    self.Xineqforbidden = Xineqforbidden  # set[int]
    self.ActionIDtoName = ActionIDtoName  # map[int->string]
    self.ModuleIDtoName = ModuleIDtoName  # map[int->string]


  def dump(self, short=False):
    if (short):
      print('%d samples, %d features' % (self.X.shape[0], self.X.shape[1]))
      return
    assert(self.Y.size == self.X.shape[0])
    print('%d samples that have %d features. First 6 samples:'
          % (self.X.shape[0], self.X.shape[1]))
    print(self.X[:6])
    print(self.Y[:6])
    print('Names of features:')
    print(self.Xnames)
    print('Value domains for features:')
    for i, dom in enumerate(self.Xdomains):
      if i > 0 and not i % 10:
        print()
      print('[%d – %d]' % (min(dom), max(dom)), end=' ' if i % 10 != 9 else '')
    print()
    print('Names of classes occuring:')
    print(self.Ynames)
    if (len(self.Xineqforbidden)):
      print('Features that have inequality predicates forbidden:')
      for i in self.Xineqforbidden:
        print('%s (id %d)' % (self.Xnames[i], i))
    if (len(self.ModuleIDtoName)):
      print('Module ID to name:')
      print(self.ModuleIDtoName)
    if (len(self.ActionIDtoName)):
      print('Action ID to name:')
      print(self.ActionIDtoName)

