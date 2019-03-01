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


def main_timeprof(folder, filename):
  def sklearn_timeprof(tree, dataset):
    tree.fit_ds(dataset)
  def dtwithlc_timeprof(tree, dataset):
    tree.fit_ds(dataset)

  ds = parse('datasets/%s' % folder, filename)
  #ds.dump()

  sk = DT_sklearn()
  sklearn_timeprof(sk, ds)
  print('sklearn_nodes: %d' % sk.inner_nodes())
  print('sklearn_correct: %s' % sk.is_correct_ds(ds))
  #sk.graph('%s_SK' % filename, png=True)

  lc = DT_linear(Split_entropy())
  dtwithlc_timeprof(lc, ds)
  print('dtwithlc_nodes: %d' % lc.inner_and_lc_nodes())
  corr = lc.is_correct_ds(ds)
  print('dtwithlc_correct: %s' % corr)
  #lc.graph('%s_LC' % filename, png=True)

  #assert(False and 'Disable assertions (python -O) for timeprofiling')


def main_with_time_profiling():
  if len(sys.argv) != 3:
    print('NEED 2 ARGUMENTS - FOLDER NAME, DATASET NAME')
    return
  if 'games' not in sys.argv[1] and 'mdps' not in sys.argv[1]:
    print('FOLDER NAME MUST CONTAIN "games" OR "mdps", PROVIDED: %s' % sys.argv[1])
    return
  if not os.path.isdir('datasets/%s' % sys.argv[1]):
    print('PROVIDED FOLDER IS NOT IN DATASETS: %s' % sys.argv[1])
    return
  fullpath = 'datasets/%s/%s' % (sys.argv[1], sys.argv[2])
  if not os.path.isfile(fullpath):
    print('PROVIDED DATASET DOES NOT EXIST: %s' % fullpath)
    return

  pr = cProfile.Profile()
  pr.enable()

  main_timeprof(sys.argv[1], sys.argv[2])

  pr.disable()
  s = io.StringIO()
  ps = pstats.Stats(pr, stream=s)
  ps.sort_stats('cumulative').print_stats('_timeprof')
  print(s.getvalue())
  s.close()


main_with_time_profiling()

