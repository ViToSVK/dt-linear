# Main file

import cProfile
import io
import pstats
import sys
sys.path.insert(0, 'src/parsing')

from parser import parse_timeprof as parse


def main_timeprof():
  ds = parse('datasets/games_wash','2_s_C_wash_4_2_1_3_f.arff')
  print(ds.X.shape)
  print(ds.X[:4])
  print(ds.y.shape)
  print(ds.y[:4])
  print(ds.labels)
  print()


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
