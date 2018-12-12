# Main file

import cProfile
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


cProfile.run('main_timeprof()', 'cprofile_stats.txt')
p = pstats.Stats('cprofile_stats.txt')
p.strip_dirs().sort_stats('cumulative').print_stats('_timeprof')
