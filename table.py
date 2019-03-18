# Generate table from prism results

import os

from plots import collect_stats, join_stats


# Algorithms
ALGOS = {'baseline': 'NoLC',
         'lc_ent': 'LC-ent',
         'lc_auc_clf': 'LC-auc',
        }
ALGOS_SORTED = ['baseline', 'lc_ent', 'lc_auc_clf']

SPECS = {'coi': {}, 'csm': {}, 'lea': {}, 'mer': {}}
SPECS['coi']['p1'] = 'F[finished]'
SPECS['coi']['p2'] = 'F[finished\&agree]'
for k in range(9):
  SPECS['csm']['p1_k%d' % k] = '[!max\_bo]U[all\_del]'
  SPECS['csm']['p2_k%d' % k] = 'F[succ\_min\_bo$\leq$%d]' % k
  SPECS['csm']['p3_k%d' % k] = 'F[max\_col$\geq$%d]' % k
SPECS['lea']['p1'] = 'G[leaders$\leq$1]'
SPECS['lea']['p2'] = 'F[elected]'
SPECS['mer']['p2'] = 'G[!err\_G]'


def pretty_time(s):

  if s < 0:
    return '-'

  m = int(s/60)
  h = int(m/60)

  p = ''
  if h>0: p = p + str(h) + 'h'
  if m>0:
    p = p + str(int(m%60)) + 'm'
    p = p + str(int(s%60)) + 's'
  else:
    p = p + str("%.2f" % s) + 's'

  return p


TYPE = ''
def table_item(stats, i):
  global TYPE
  parts = []
  model = stats['names'][i]
  spec = model[model.rfind('__'):]
  model = model.replace(spec, '').replace('_','\_')
  model += ' \\hspace{2mm}'
  spec = spec.replace('__','').replace('.prism','')

  if (model[:3] != TYPE):
    res = '\\hline\n'
  else:
    res = ''
  TYPE = model[:3]

  assert(spec in SPECS[TYPE])
  spec = SPECS[TYPE][spec] + ' \\hspace{2mm}'
  res += '%s & %s' % (model, spec)

  parts.append(stats['samples'][i])
  parts.append(stats['features'][i])

  mi = (0, 31337)
  idx = 1
  for algo in ALGOS_SORTED:
    idx += 1
    x = stats[algo][i]
    if (x < mi[1]):
      mi = (idx, x)
    parts.append(x)

  for i, p in enumerate(parts):
    if (mi[0] == i):
      res += ' & \\textbf{%d}' % p
    else:
      res += ' & %d' % p
  res += ' \\\\'
  return res


def create_table(stats):
  print('\\small')  # scriptsize
  print('\\begin{longtable}{|l | l | l | l || r | r | r |}')
  print('\\hline')
  print('\\textbf{Model} & \\textbf{Spec} & \\textbf{Samples} & \\textbf{Features} & $\\textbf{%s}$ & $\\textbf{%s}$ & $\\textbf{%s}$ \\\\'
        % (ALGOS['baseline'], ALGOS['lc_ent'], ALGOS['lc_auc_clf']))
  print('\\hline')

  for stat in stats:
    assert('max' in stat or 'min' in stat or
           len(stats[stat]) == len(stats['names']))
  for i in range(len(stats['names'])):
    print(table_item(stats, i))

  print('\\hline')
  print('\\end{longtable}')


def main():
  if not os.path.isdir('results/reports'):
    return

  stats_prism = []
  for filename in sorted(os.listdir('results/reports')):
    if ('report_' in filename and 'prism' in filename and '.txt' in filename):
      #print(filename)
      stats_prism.append(collect_stats(filename))
  if (len(stats_prism) > 0):
    stats = join_stats(stats_prism)
    create_table(stats)


if (__name__ == "__main__"):
  main()

