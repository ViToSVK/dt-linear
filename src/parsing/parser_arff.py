# Parser for games - arff

import numpy as np

from dataset import Dataset


def parse_arff(folder, filename):
  assert('games' in folder)
  assert('.arff' in filename)
  data = False
  X = []
  Y = []
  Xnames = []
  Xranges = []
  Ynames = {}
  for line in open('%s/%s' % (folder, filename), 'r'):
    if data:
      sample = [float(e) if e.isdigit() else e for e in line.split(',')]
      cl = sample[-1].strip()
      del sample[-1]
      if cl not in Ynames:
        pos = len(Ynames)
        Ynames[cl] = pos
      Y.append(Ynames[cl])
      X.append(sample)
    else:
      if line.startswith('@DATA'):
        data = True
      elif line.startswith('@ATTRIBUTE'):
        label = line.split('\"')[1]
        if label != 'class':
          Xnames.append(label)
          Xranges.append([0, 1])
  return Dataset(np.array(X), np.array(Y), np.array(Xnames),
                 np.array(Xranges), Ynames)

