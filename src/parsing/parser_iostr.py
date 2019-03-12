# Parser for MDPs - iostr

import numpy as np

from dataset import Dataset


def parse_iostr(folder, filename):
  assert('mdps' in folder)
  assert('.iostr' in filename)
  data = False
  X = []
  Y = []
  Xnames = np.array([])
  Svarno = -1
  Avarno = -1
  Xdomains = []
  Ynames = {'no': 0, 'yes': 1}

  def decimal_to_binary(decimal, length):
    res = []
    for i in range(length):
      res.append(decimal % 2)
      decimal = decimal // 2  # int division
    return list(reversed(res))

  def binary_to_decimal(binary):
    res, add = 0, 1
    for i in reversed(binary):
      if (i == 1):
        res += add
      add *= 2
    return res

  assert(47 == binary_to_decimal(decimal_to_binary(47, 9)))

  lineno = 0
  for line in open('%s/%s' % (folder, filename), 'r'):
    lineno += 1
    if (lineno == 1):
      assert(filename.replace('.iostr','') in line)
      continue
    if (lineno == 2):
      assert('State Boolean variable names' in line)
      Svarno = int(line.split()[-1].replace('(','').replace(')',''))
      continue
    if (lineno == 3):
      Xnames = np.array(line.split())
      assert(Xnames.size == Svarno)
      continue
    if (lineno == 4):
      assert('Action Boolean variable names' in line)
      Avarno = int(line.split()[-1].replace('(','').replace(')',''))
      continue
    if (lineno == 5):
      Xnames = np.append(Xnames, np.array(line.split()))
      assert(Xnames.size == Svarno + Avarno)
      for _ in range(Xnames.size):
        Xdomains.append(set({0, 1}))
      continue
    if (lineno == 6):
      assert('State-action pairs' in line)
      continue
    if (lineno == 7):
      assert(line == '# statenumber:(s,t,a,t,e,v,a,l,u,e,s):successornumber:(p,l,a,y,t,h,i,s,a,c,t,i,o,n)\n')
      continue

    # State-action pairs
    spl = line.split(':')
    assert(len(spl) == 4)
    stval = [float(e) for e in spl[1].replace('(','').replace(')','').split(',')]
    assert(len(stval) == Svarno)
    acval = [int(e) for e in spl[3].replace('(','').replace(')','').split(',')]
    assert(len(acval) == Avarno)

    X.append(stval + [float(e) for e in acval])  # copies stval
    Y.append(1)
    acdec = binary_to_decimal(acval)
    for i in range(2 ** Avarno):
      if (i != acdec):
        X.append(stval + [float(e) for e in decimal_to_binary(i, Avarno)])
        Y.append(0)

  return Dataset(np.array(X).astype(float), np.array(Y).astype(int),
                 Xnames, Xdomains, Ynames)

