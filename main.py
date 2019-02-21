# Main file

import cProfile
import io
import os
import pstats
import sys
sys.path.insert(0, 'src/dt')
sys.path.insert(0, 'src/parsing')
sys.path.insert(0, 'src/split')

from parser import parse_timeprof as parse
from dt_sklearn import DT_sklearn
from dt_linear import DT_linear
from entropy import Split_entropy

def main_timeprof():
  def sklearn_timeprof(tree, dataset):
    tree.fit_ds(dataset)
  def lc_timeprof(tree, dataset):
    tree.fit_ds(dataset)

  """
  for mdpdir in ['csma', 'consensus']:
    for filename in os.listdir('datasets/mdps_%s' % mdpdir):
      ds = parse('datasets/mdps_%s' % mdpdir, filename)
      ds.dump()
      print()
  """

  folder = 'games_wash'
  filename = '2_s_C_wash_2_3_1_2_t'

  ds = parse('datasets/%s' % folder, '%s.%s'
             % (filename, 'arff' if 'games' in folder else 'prism'))
  ds.dump()
  print()


  sk = DT_sklearn()
  sklearn_timeprof(sk, ds)
  print(sk.inner_nodes())
  print(sk.score_ds(ds))
  print(sk.is_correct_ds(ds))
  sk.graph('%s_SK' % filename, png=True)
  print()

  lc = DT_linear(Split_entropy())
  lc_timeprof(lc, ds)
  print(lc.inner_nodes())
  print(lc.is_correct_ds(ds))
  lc.graph('%s_LC' % filename, png=True)
  print()


  assert(False and 'Disable assertions (python -O) for timeprofiling')


def main_with_time_profiling():
  pr = cProfile.Profile()
  pr.enable()

  main_timeprof()

  pr.disable()
  s = io.StringIO()
  ps = pstats.Stats(pr, stream=s)
  ps.sort_stats('cumulative').print_stats('_timeprof')
  print(s.getvalue())
  s.close()


main_with_time_profiling()
