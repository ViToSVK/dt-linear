# Generate plots from results

from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


# Algorithms
ALGOS = {'sklearn': 'Scikit-learn',
         'baseline': 'Baseline',
         'lc_ent': 'LC_entropy',
         'lc_auc': 'LC_auc',
        }


def collect_stats(filename):
  stats = {'names': []}
  for algo in ALGOS:
    stats[algo] = []
    stats['%s_max' % algo] = 1
    stats['%s_min' % algo] = 31337

  namefollows = False
  disregard = True
  c = {'name': ''}
  for algo in ALGOS:
    c[algo] = -1
  for line in open('results/reports/%s' % filename, 'r'):
    if '===' in line:
      if not disregard:
        stats['names'].append(c['name'])
        for algo in ALGOS:
          stats[algo].append(c[algo])
          if c[algo] > stats['%s_max' % algo]:
            stats['%s_max' % algo] = c[algo]
          if c[algo] < stats['%s_min' % algo]:
            stats['%s_min' % algo] = c[algo]
      namefollows = True
      continue
    if namefollows:
      namefollows = False
      c['name'] = line.strip()
      for algo in ALGOS:
        c[algo] = -1
      disregard = False
      continue
    if 'nodes' in line and 'correct' in line and 'time' in line:
      for algo in ALGOS:
        if algo in line:
          assert(c[algo] == -1)
          l = line.split()
          c[algo] = int(l[1])
          if l[3] != 'T':
            print('%s TREE FOR %s IS NOT CORRECT!' % (algo, c['name']))
            disregard = True
          break

  stats['names'] = np.array(stats['names'])
  for algo in ALGOS:
    stats[algo] = np.array(stats[algo])
  return stats


def join_stats(stats_list):
  res = {'names': np.array([])}
  for algo in ALGOS:
    res[algo] = np.array([])
    res['%s_max' % algo] = 1
    res['%s_min' % algo] = 1
  for stats in stats_list:
    res['names'] = np.concatenate((res['names'], stats['names']))
    for algo in ALGOS:
      res[algo] = np.concatenate((res[algo], stats[algo]))
      res['%s_max' % algo] = max(res['%s_max' % algo], stats['%s_max' % algo])
  return res


def search(stats):
  for i,x in enumerate(stats['lc_auc']):
    if x > stats['baseline'][i]:
      print(stats['names'][i])


def all_plot(stats, plotname, x_algo, y_algo, z_algo=None, shift=True):
  SHIFT = 4 if shift else 0
  l = stats[x_algo].size
  assert(l == stats[y_algo].size)
  assert(z_algo is None or l == stats[z_algo].size)

  fig = plt.figure(num=None, figsize=(15, 7), dpi=150, facecolor='w', edgecolor='k')
  ind = np.lexsort((stats[x_algo], stats[y_algo])) # first by y_algo
  plt.scatter(range(l), [stats[x_algo][i] + SHIFT for i in ind],
              color='b', edgecolor='black', linewidth=0.4, alpha=1.)
  plt.scatter(range(l), [stats[y_algo][i] + SHIFT for i in ind],
              color='r', edgecolor='black', linewidth=0.4, alpha=1.)
  if z_algo is not None:
    plt.scatter(range(l), [stats[z_algo][i] + SHIFT for i in ind],
                color='g', edgecolor='black', linewidth=0.4, alpha=0.6)

  ma = max(stats['%s_max' % x_algo], stats['%s_max' % y_algo])
  if z_algo is not None:
    ma = max(ma, stats['%s_max' % z_algo])
  mi = min(stats['%s_min' % x_algo], stats['%s_min' % y_algo])
  if z_algo is not None:
    mi = min(mi, stats['%s_min' % z_algo])

  plt.ylim([(SHIFT+mi) * 0.9, ma * 1.1])
  plt.xlim([-(l-1) * 0.007, (l-1) * 1.007])

  ax = plt.gca()
  ax.set_yscale('log', basey=np.e)
  y_ticks = SHIFT + np.ceil(np.logspace(np.log(mi), np.log(ma), num=10, base=np.e))
  ax.set_yticks(y_ticks)
  ax.set_yticklabels((y_ticks - SHIFT).astype(int))
  plt.xlabel('Benchmark number', fontsize=18)
  plt.ylabel('Decision tree size', fontsize=18)

  patches = []
  patches.append(mpatches.Patch(color='blue', label=ALGOS[x_algo]))
  patches.append(mpatches.Patch(color='red', label=ALGOS[y_algo]))
  if z_algo is not None:
    patches.append(mpatches.Patch(color='green', label=ALGOS[z_algo]))
  plt.legend(handles=patches, loc=4)

  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_all_plot_%s_%s.png' %
              (plotname, x_algo, y_algo), bbox_inches='tight')
  plt.clf()
  plt.close(fig)


def versus_plot(stats, plotname, x_algo, y_algo):
  fig = plt.figure(num=None, figsize=(12, 6), dpi=150, facecolor='w', edgecolor='k')
  plt.scatter(stats[x_algo], stats[y_algo],
              color='b', edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.plot([0, 100000], [0, 100000], color='r', linewidth=2.0, alpha=0.4)
  plt.plot([0, 100000], [0, 10000], color='r', linewidth=2.0, alpha=0.2)
  plt.xlim([-1, stats['%s_max' % x_algo] * 1.01])
  plt.ylim([-1, stats['%s_max' % y_algo] * 1.015])
  plt.xlabel('%s size' % ALGOS[x_algo], fontsize=18)
  plt.ylabel('%s size' % ALGOS[y_algo], fontsize=18)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_vs_plot_%s_%s.png' %
              (plotname, x_algo, y_algo), bbox_inches='tight')
  plt.clf()
  plt.close(fig)


def ratio_plot(stats, plotname, x_algo, y_algo):
  l = stats[x_algo].size
  assert(l == stats[y_algo].size)
  fig = plt.figure(num=None, figsize=(12, 6), dpi=150, facecolor='w', edgecolor='k')
  plt.clf()
  plt.scatter(range(l), stats[y_algo] / stats[x_algo],
              color='b', edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.plot([-10, 100000], [1, 1], color='r', linewidth=4.0, alpha=0.6)
  plt.ylim([min(0.05, min(stats[y_algo] / stats[x_algo])),
            max(1.15, max(stats[y_algo] / stats[x_algo]) + 0.02)])
  plt.xlim([-(l-1) * 0.007, (l-1) * 1.007])
  ax = plt.gca()
  #ax.set_yscale('log', basey=np.e)
  ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1]
  ax.set_yticks(ticks)
  ax.set_yticklabels(ticks)
  plt.xlabel('Benchmark number', fontsize=18)
  plt.ylabel('%s/%s size ratio' % (ALGOS[y_algo], ALGOS[x_algo]), fontsize=18)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_ratio_plot_%s_%s.png' %
              (plotname, x_algo, y_algo), bbox_inches='tight')
  plt.clf()
  plt.close(fig)


def create_plots(stats, plotname):
  all_plot(stats, plotname, 'baseline', 'lc_auc', 'lc_ent')
  versus_plot(stats, plotname, 'baseline', 'lc_auc')  # lc_ent
  ratio_plot(stats, plotname, 'baseline', 'lc_auc')  # lc_ent
  #all_plot(stats, plotname, 'sklearn', 'lc_auc', 'lc_ent')
  #versus_plot(stats, plotname, 'sklearn', 'lc_auc')
  #ratio_plot(stats, plotname, 'sklearn', 'lc_auc')


def main():
  if not os.path.isdir('results/reports'):
    return

  stats_prism = []
  for filename in sorted(os.listdir('results/reports')):
    if 'report_' in filename and '.txt' in filename:
      print(filename)
      if ('prism' in filename):
        stats_prism.append(collect_stats(filename))
      else:
        stats = collect_stats(filename)
        plotname = filename.replace('report_', '').replace('.txt', '')
        search(stats)
        create_plots(stats, plotname)
  if (len(stats_prism) > 0):
    stats = join_stats(stats_prism)
    search(stats)
    create_plots(stats, 'mdps_prism')


main()

