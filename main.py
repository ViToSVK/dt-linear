# Main file

import cProfile
import io
import os
import pstats
import sys
sys.path.insert(0, 'src/parsing')
sys.path.insert(0, 'src/dt')

from parser import parse_timeprof as parse
from dt_sklearn import DT_sklearn


def main_timeprof():
  """
  for mdpdir in ['csma', 'consensus']:
    for filename in os.listdir('datasets/mdps_%s' % mdpdir):
      ds = parse('datasets/mdps_%s' % mdpdir, filename)
      ds.dump()
      print()
  """

  # ds = parse('datasets/games_wash', '2_s_C_wash_2_3_1_2_t.arff')
  # ds = parse('datasets/games_wash', '2_s_C_wash_3_4_4_3_t.arff')
  ds = parse('datasets/games_wash', '2_s_C_wash_3_5_3_3_f.arff')
  ds.dump()
  print()

  x = DT_sklearn()
  x.fit_ds(ds)
  print(x.inner_nodes())
  print(x.score_ds(ds))
  print(x.is_correct_ds(ds))
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
