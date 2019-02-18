# Parser for MDPs - prism

import numpy as np

from dataset import Dataset


def parse_prism(folder, filename):
  assert('mdps' in folder)
  assert('.prism' in filename)
  data = False
  X = []
  Y = []
  Xnames = []
  Xranges = []
  Ynames = {}
  for line in open('%s/%s' % (folder, filename), 'r'):
    if line.startswith('#'):
      continue # first line
    if line.startswith('\n'):
      break # we have finished parsing
    if line.startswith('('):
      # sample
      assert(len(Xnames) and len(Xranges))
      [smp, cl] = line.split(':')
      assert(smp[0] == '(' and smp[-1] == ')')
      smp = smp[1:-1]
      sample = [float(e) for e in smp.split(',')]
      cl = cl.strip()
      if cl not in Ynames:
        pos = len(Ynames)
        Ynames[cl] = pos
      Y.append(Ynames[cl])
      X.append(sample)
    else:
      assert(not len(Xnames) and not len(Xranges))
      namesRanges = line.split(']')[:-1]
      assert(len(namesRanges) > 1)
      for nr in namesRanges:
        [name, rnge] = nr.split('[')
        if ';' in name:
          name = name[name.index(';')+1:]
        Xnames.append(name[:-1])
        Xranges.append([int(e) for e in rnge.split(',')])
  return Dataset(np.array(X), np.array(Y), np.array(Xnames),
                 np.array(Xranges), Ynames)
