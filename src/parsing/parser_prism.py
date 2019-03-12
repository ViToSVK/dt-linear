# Parser for MDPs - prism

import numpy as np

from dataset import Dataset


def parse_prism(folder, filename):
  assert('mdps' in folder)
  assert('.prism' in filename)

  def nondigits_to_float(token):
    if (token == 'false'):
      return 0.0
    if (token == 'true'):
      return 1.0
    assert(token[0] == '-')
    return -(float(token[1:]))

  data = False
  X = []
  Y = []
  Xnames = []
  Xranges = []
  Ynames = {'no': 0, 'yes': 1}
  Actions = {}  # map string(action name)->int(action id)
  ActionIDtoName = {}  # map int(action id)->string(action name)

  # First scan to collect actions played, names and ranges
  for line in open('%s/%s' % (folder, filename), 'r'):
    if line.startswith('#'):
      continue # first line
    if line.startswith('\n'):
      break # we have finished parsing
    if line.startswith('('):
      # sample - get action played
      assert(len(Xnames) and len(Xranges))
      [_, act] = line.split(':')
      act = act.strip()
      if act not in Actions:
        pos = len(Actions)
        Actions[act] = pos
        ActionIDtoName[pos] = act
    else:
      assert(not len(Xnames) and not len(Xranges))
      namesRanges = line.split(']')[:-1]
      assert(len(namesRanges) > 1)
      for nr in namesRanges:
        [name, rnge] = nr.split('[')
        if ';' in name:
          name = name[name.index(';')+1:]
        name = name[:-1]
        if ('_' in name and len(name.split('_')) == 2 and
            len(name.split('_')[1]) > 1):
          name = name.split('_')[1]
        Xnames.append(name)
        Xranges.append([int(e) for e in rnge.split(',')])
  # Add action feature
  Xnames.append('action')
  Xranges.append([0, len(Actions) - 1])
  Xineqforbidden = set({len(Xnames) - 1})

  # Asserts
  allnames = set()
  for name in Xnames:
    assert(name not in allnames)
    allnames.add(name)
  assert(len(Actions) == len(ActionIDtoName))
  for action in Actions:
    assert(ActionIDtoName[Actions[action]] == action)
  assert(len(Actions) >= 2)

  # Second scan to collect samples
  for line in open('%s/%s' % (folder, filename), 'r'):
    if line.startswith('#'):
      continue # first line
    if line.startswith('\n'):
      break # we have finished parsing
    if line.startswith('('):
      # sample
      assert(len(Xnames) and len(Xranges))
      [smp, act] = line.split(':')
      assert(smp[0] == '(' and smp[-1] == ')')
      smp = smp[1:-1]
      sample = [float(e) if e.isdigit() else nondigits_to_float(e)
                for e in smp.split(',')]
      act = act.strip()
      assert(act in Actions)
      sample.append(float(Actions[act]))
      Y.append(Ynames['yes'])
      X.append(sample)
      # copies for other actions played
      for badact in Actions:
        if (badact != act):
          badsam = sample.copy()
          badsam[-1] = float(Actions[badact])
          X.append(badsam)
          Y.append(Ynames['no'])

  return Dataset(np.array(X), np.array(Y), np.array(Xnames),
                 np.array(Xranges), Ynames, Xineqforbidden, ActionIDtoName)

