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
  Xdomains = []
  Ynames = {'no': 0, 'yes': 1}

  Actions = {}  # map string(action name)->int(action id)
  ActionIDtoName = {}  # map int(action id)->string(action name)
  Modules = {'synchronous': 0}  # map string->int
  ModuleIDtoName = {0: 'synchronous'}  # map int->string
  ModActPlayed = set()

  # First scan to collect actions played, names and ranges
  for line in open('%s/%s' % (folder, filename), 'r'):
    if line.startswith('#'):
      continue # first line
    if line.startswith('\n'):
      break # we have finished parsing
    if line.startswith('('):
      # sample - get action played
      assert(len(Xnames) and len(Xdomains))
      [_, act] = line.split(':')
      act = act.strip()
      mod = 'synchronous'
      if ('async_' in act):
        assert(len(act.split('_')) > 2)  # async actionname modulename
        mod = act.split('_')[-1]
        act = '_'.join(act.split('_')[:-1])
      if act not in Actions:
        pos = len(Actions)
        Actions[act] = pos
        ActionIDtoName[pos] = act
      if mod not in Modules:
        pos = len(Modules)
        Modules[mod] = pos
        ModuleIDtoName[pos] = mod
      if (mod, act) not in ModActPlayed:
        ModActPlayed.add((mod, act))
    else:
      assert(not len(Xnames) and not len(Xdomains))
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
        [mi, ma] = rnge.split(',')
        mi, ma = int(mi), int(ma)
        assert(mi < ma)
        Xdomains.append(set({e for e in range(mi, ma+1)}))
  # Add module and action features
  Xnames.append('module')
  Xdomains.append(set({e for e in range(len(Modules))}))
  Xnames.append('action')
  Xdomains.append(set({e for e in range(len(Actions))}))
  Xineqforbidden = set({len(Xnames) - 2, len(Xnames) - 1})

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
      assert(len(Xnames) and len(Xdomains))
      [smp, act] = line.split(':')
      assert(smp[0] == '(' and smp[-1] == ')')
      smp = smp[1:-1]
      sample = [float(e) if e.isdigit() else nondigits_to_float(e)
                for e in smp.split(',')]
      act = act.strip()
      mod = 'synchronous'
      if ('async_' in act):
        assert(len(act.split('_')) > 2)  # async actionname modulename
        mod = act.split('_')[-1]
        act = '_'.join(act.split('_')[:-1])
      assert(act in Actions)
      assert(mod in Modules)
      assert((mod, act) in ModActPlayed)
      sample.append(float(Modules[mod]))
      sample.append(float(Actions[act]))
      Y.append(Ynames['yes'])
      X.append(sample)
      # copies for other (module,action)s played
      for (modX, actX) in sorted(ModActPlayed):
        if (modX != mod or actX != act):
          badsam = sample.copy()
          badsam[-2] = float(Modules[modX])
          badsam[-1] = float(Actions[actX])
          X.append(badsam)
          Y.append(Ynames['no'])

  return Dataset(np.array(X), np.array(Y), np.array(Xnames),
                 Xdomains, Ynames,
                 Xineqforbidden, ActionIDtoName, ModuleIDtoName)

