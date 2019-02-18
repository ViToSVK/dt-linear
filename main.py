# Main file

import cProfile
import io
import os
import pstats
import sys
sys.path.insert(0, 'src/parsing')

from parser import parse_timeprof as parse


def main_timeprof():
  for mdpdir in ['csma', 'consensus']:
    for filename in os.listdir('datasets/mdps_%s' % mdpdir):
      ds = parse('datasets/mdps_%s' % mdpdir, filename)
      ds.dump()
      print()

  ds = parse('datasets/games_wash', '2_s_C_wash_2_3_1_2_t.arff')
  ds.dump()
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
