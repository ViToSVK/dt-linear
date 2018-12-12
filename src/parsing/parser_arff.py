# Parser for games - arff

import numpy as np

from dataset import Dataset


def parse_arff(folder, filename):
  assert('games' in folder)
  assert('.arff' in filename)
  data = False
  X = []
  y = []
  labels = []
  for line in open('%s/%s' % (folder, filename), 'r'):
    if data:
      sample = [float(e) if e.isdigit() else e for e in line.split(',')]
      y.append(-1. if sample[-1].startswith("no") else 1.)
      del sample[-1]
      X.append(sample)
    else:
      if line.startswith('@DATA'):
        data = True
      elif line.startswith('@ATTRIBUTE'):
        labels.append(line.split('\"')[1])
  return Dataset(np.array(X), np.array(y), np.array(labels))
