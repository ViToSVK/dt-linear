# Entropy splitting criterion

from math import log2
import sys
sys.path.insert(0, '../dt')

from node_types import Predicate


class Split_entropy:
  def __init__(self):
    self.EPSILON = 0.00000001


  def best(self, data):
    Ypossibilities = []
    min_y = 31337
    for name, idx in data.Ynames.items():
      Ypossibilities.append(idx)
      if idx < min_y:
        min_y = idx

    # For each equality predicate:
    # #sat; #sat&y==k; #unsat&y==k
    eq_s, eq_s_y, eq_u_y = {}, {}, {}

    # For each inequality predicate:
    # #sat; #sat&y==k; #unsat&y==k
    in_s, in_s_y, in_u_y = {}, {}, {}

    # Collect predicates that will be considered
    for i, ran in enumerate(data.Xranges):
      assert(ran[0] <= ran[1])
      if (ran[1] - ran[0] == 1):
        # [0,1] --> =1
        eq_s[(i, ran[1])] = 0
        eq_s_y[(i, ran[1])] = {}
        eq_u_y[(i, ran[1])] = {}
        for y in Ypossibilities:
          if y != min_y:
            eq_s_y[(i, ran[1])][y] = 0
            eq_u_y[(i, ran[1])][y] = 0

      elif (ran[1] - ran[0] > 1):
        # [0,2] --> =0 =1 =2 <1 <2
        eq_s[(i, ran[0])] = 0
        eq_s_y[(i, ran[0])] = {}
        eq_u_y[(i, ran[0])] = {}
        for y in Ypossibilities:
          if y != min_y:
            eq_s_y[(i, ran[0])][y] = 0
            eq_u_y[(i, ran[0])][y] = 0
        for val in range(ran[0]+1, ran[1]+1):  # ran[0]+1, ..., ran[1]
          eq_s[(i, val)] = 0
          eq_s_y[(i, val)] = {}
          eq_u_y[(i, val)] = {}
          in_s[(i, val)] = 0
          in_s_y[(i, val)] = {}
          in_u_y[(i, val)] = {}
          for y in Ypossibilities:
            if y != min_y:
              eq_s_y[(i, val)][y] = 0
              eq_u_y[(i, val)][y] = 0
              in_s_y[(i, val)][y] = 0
              in_u_y[(i, val)][y] = 0

    # Predicates collected
    # Go through the dataset and collect the statistics
    assert(data.X.shape[0] == data.Y.size)
    assert(data.X.shape[0] > 0)
    total = {}
    for i, x in enumerate(data.X):
      y = data.Y[i]
      if y not in total:
        total[y] = 1
      else:
        total[y] += 1
      for (pos, val) in eq_s:
        if x[pos] == val:
          eq_s[(pos, val)] += 1
          assert(y in eq_s_y[(pos, val)] or y == min_y)
          if y in eq_s_y[(pos, val)]:
            eq_s_y[(pos, val)][y] += 1
        else:
          assert(y in eq_u_y[(pos, val)] or y == min_y)
          if y in eq_u_y[(pos, val)]:
            eq_u_y[(pos, val)][y] += 1
      for (pos, val) in in_s:
        if x[pos] < val:
          in_s[(pos, val)] += 1
          assert(y in in_s_y[(pos, val)] or y == min_y)
          if y in in_s_y[(pos, val)]:
            in_s_y[(pos, val)][y] += 1
        else:
          assert(y in in_u_y[(pos, val)] or y == min_y)
          if y in in_u_y[(pos, val)]:
            in_u_y[(pos, val)][y] += 1

    # Statistics collected
    # Compute current entropy (times number of samples in X)
    cur_ent = 0.
    log2_Xshape0 = log2(data.X.shape[0])
    for y in total:
      cur_ent -= (total[y] * (log2(total[y]) - log2_Xshape0))
    assert(cur_ent > 0.)

    # Find the best predicate (entropies are positive, we minimize)
    b_ent = cur_ent
    b_pos = -1
    b_val = -1
    b_eq = None

    def get_ent(pos, val, equality):
      # Compute entropy (times number of samples in X) after this split
      sat = eq_s if equality else in_s
      sat_y = eq_s_y if equality else in_s_y
      uns_y = eq_u_y if equality else in_u_y

      sat_ALL = sat[(pos, val)]
      if sat_ALL == 0 or sat_ALL == data.X.shape[0]:
        return None  # Useless predicate
      cand_ent = 0.
      # SAT part of split
      log2_sat_ALL = log2(sat_ALL)
      sMINY = sat_ALL
      for _, sY in sat_y[(pos, val)].items():
        assert(sY >= 0)
        if sY > 0:
          sMINY -= sY
          cand_ent -= (sY * (log2(sY) - log2_sat_ALL))
      assert(sMINY >= 0)
      if sMINY > 0:
        cand_ent -= (sMINY * (log2(sMINY) - log2_sat_ALL))
      # UNSAT part of split
      uns_ALL = data.X.shape[0] - sat_ALL
      log2_uns_ALL = log2(uns_ALL)
      uMINY = uns_ALL
      for _, uY in uns_y[(pos, val)].items():
        assert(uY >= 0)
        if uY > 0:
          uMINY -= uY
          cand_ent -= (uY * (log2(uY) - log2_uns_ALL))
      assert(uMINY >= 0)
      if uMINY > 0:
        cand_ent -= (uMINY * (log2(uMINY) - log2_uns_ALL))
      # Done; return computed entropy
      return cand_ent

    for (pos, val) in eq_s:
      ent = get_ent(pos, val, equality=True)
      if ent is not None and (ent < cur_ent + self.EPSILON):
        assert(ent >= 0.)
        if b_eq is None or (b_ent - ent > self.EPSILON):
          # New best
          b_ent = ent
          b_pos = pos
          b_val = val
          b_eq = True

    for (pos, val) in in_s:
      ent = get_ent(pos, val, equality=False)
      if ent is not None and (ent < cur_ent + self.EPSILON):
        assert(ent >= 0.)
        if b_eq is None or (b_ent - ent > self.EPSILON):
          # New best
          b_ent = ent
          b_pos = pos
          b_val = val
          b_eq = False

    # Done; return the best predicate
    assert(b_pos >= 0)
    assert(b_pos < data.Xnames.size)
    return Predicate(fname=data.Xnames[b_pos], fpos=b_pos,
                     equality=b_eq, number=b_val)

