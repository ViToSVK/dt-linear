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
from auc import Split_auc


def main_timeprof(folder, filename):

  ds = parse('datasets/%s' % folder, filename)
  ds.dump(short=True)

  #return

  sk = DT_sklearn()
  def sklearn_timeprof():
    sk.fit_ds(ds)
  sklearn_timeprof()
  print('sklearn_nodes: %d' % sk.inner_nodes())
  print('sklearn_correct: %s' % sk.is_correct_ds(ds))
  #sk.graph('%s_SK' % filename, png=True)

  bl = DT_linear(Split_entropy(), use_lc=False)
  def baseline_timeprof():
    bl.fit_ds(ds)
  baseline_timeprof()
  print('baseline_nodes: %d' % bl.inner_and_lc_nodes())
  print('baseline_wavg_depth: %.4f' % bl.weighted_avg_depth())
  print('baseline_correct: %s' % bl.is_correct_ds(ds))
  #bl.graph('%s_BL' % filename, png=True)

  lc_ent = DT_linear(Split_entropy(), use_lc=True)
  def lc_ent_timeprof():
    lc_ent.fit_ds(ds)
  lc_ent_timeprof()
  print('lc_ent_nodes: %d' % lc_ent.inner_and_lc_nodes())
  print('lc_ent_wavg_depth: %.4f' % lc_ent.weighted_avg_depth())
  print('lc_ent_correct: %s' % lc_ent.is_correct_ds(ds))
  #lc_ent.graph('%s_LC_ENT' % filename, png=True)

  lc_auc_reg = DT_linear(Split_auc(), use_lc=True)
  def lc_auc_reg_timeprof():
    lc_auc_reg.fit_ds(ds)
  lc_auc_reg_timeprof()
  print('lc_auc_reg_nodes: %d' % lc_auc_reg.inner_and_lc_nodes())
  print('lc_auc_reg_wavg_depth: %.4f' % lc_auc_reg.weighted_avg_depth())
  print('lc_auc_reg_correct: %s' % lc_auc_reg.is_correct_ds(ds))
  #lc_auc_reg.graph('%s_LC_AUC_REG' % filename, png=True)

  lc_auc_clf = DT_linear(Split_auc(use_svm=True), use_lc=True)
  def lc_auc_clf_timeprof():
    lc_auc_clf.fit_ds(ds)
  lc_auc_clf_timeprof()
  print('lc_auc_clf_nodes: %d' % lc_auc_clf.inner_and_lc_nodes())
  print('lc_auc_clf_wavg_depth: %.4f' % lc_auc_clf.weighted_avg_depth())
  print('lc_auc_clf_correct: %s' % lc_auc_clf.is_correct_ds(ds))
  #lc_auc_clf.graph('%s_LC_AUC_CLF' % filename, png=True)

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

