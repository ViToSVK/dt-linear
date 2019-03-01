# Generate plots from results

from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


# Algorithms
ALGOS = ['sklearn', 'baseline', 'dtwithlc']


def collect_stats(filename):
  stats = {'names': []}
  for algo in ALGOS:
    stats[algo] = []
    stats['%s_max' % algo] = 1

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


def all_plot(stats, x_algo, y_algo, plotname):
  l = stats[x_algo].size
  assert(l == stats[y_algo].size)
  m = max(stats['%s_max' % x_algo], stats['%s_max' % y_algo])
  plt.figure(num=None, figsize=(15, 6), dpi=150, facecolor='w', edgecolor='k')
  ind = np.lexsort((stats[x_algo], stats[y_algo])) # first by y_algo
  plt.scatter(range(l), [stats[x_algo][i] for i in ind],
              color='b', edgecolor='black', linewidth=0.4, alpha=1.)
  plt.scatter(range(l), [stats[y_algo][i] for i in ind],
              color='r', edgecolor='black', linewidth=0.4, alpha=1.)
  plt.ylim([0.9, m * 1.1])
  plt.xlim([-(l-1) * 0.007, (l-1) * 1.007])
  ax = plt.gca()
  ax.set_yscale('log', basey=np.e)
  y_ticks = np.logspace(np.log(1.), np.log(m), num=10, base=np.e)
  ax.set_yticks(y_ticks)
  ax.set_yticklabels(np.ceil(y_ticks).astype(int))
  plt.xlabel('Benchmark number', fontsize=18)
  plt.ylabel('Decision tree size', fontsize=18)
  blue_patch = mpatches.Patch(color='blue', label=x_algo)
  red_patch = mpatches.Patch(color='red', label=y_algo)
  plt.legend(handles=[blue_patch, red_patch], loc=4)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_all_plot_%s_%s.png' %
              (plotname, x_algo, y_algo), bbox_inches='tight')


def versus_plot(stats, x_algo, y_algo, plotname):
  plt.figure(num=None, figsize=(12, 6), dpi=150, facecolor='w', edgecolor='k')
  plt.scatter(stats[x_algo], stats[y_algo],
              color='b', edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.plot([0, 100000], [0, 100000], color='r', linewidth=2.0, alpha=0.4)
  plt.plot([0, 100000], [0, 10000], color='r', linewidth=2.0, alpha=0.2)
  plt.xlim([-1, stats['%s_max' % x_algo] * 1.01])
  plt.ylim([-1, stats['%s_max' % y_algo] * 1.015])
  plt.xlabel('%s size' % x_algo, fontsize=18)
  plt.ylabel('%s size' % y_algo, fontsize=18)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_vs_plot_%s_%s.png' %
              (plotname, x_algo, y_algo), bbox_inches='tight')


def ratio_plot(stats, x_algo, y_algo, plotname):
  l = stats[x_algo].size
  assert(l == stats[y_algo].size)
  plt.figure(num=None, figsize=(12, 6), dpi=150, facecolor='w', edgecolor='k')
  plt.clf()
  plt.scatter(range(l), stats[y_algo] / stats[x_algo],
              color='b', edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.plot([-10, 100000], [1, 1], color='r', linewidth=4.0, alpha=0.6)
  plt.ylim([0.001, 3.])
  plt.xlim([-(l-1) * 0.007, (l-1) * 1.007])
  ax = plt.gca()
  ax.set_yscale('log', basey=np.e)
  ticks = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 2., 3.]
  ax.set_yticks(ticks)
  ax.set_yticklabels(ticks)
  plt.xlabel('Benchmark number', fontsize=18)
  plt.ylabel('%s/%s size ratio' % (y_algo, x_algo), fontsize=18)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_ratio_plot_%s_%s.png' %
              (plotname, x_algo, y_algo), bbox_inches='tight')


def main():
  if not os.path.isdir('results/reports'):
    return
  for filename in sorted(os.listdir('results/reports')):
    if 'report_' in filename and '.txt' in filename:
      print(filename)
      stats = collect_stats(filename)
      plotname = filename.replace('report_', '').replace('.txt', '')
      all_plot(stats, 'baseline', 'dtwithlc', plotname)
      versus_plot(stats, 'baseline', 'dtwithlc', plotname)
      ratio_plot(stats, 'baseline', 'dtwithlc', plotname)
      all_plot(stats, 'sklearn', 'dtwithlc', plotname)
      versus_plot(stats, 'sklearn', 'dtwithlc', plotname)
      ratio_plot(stats, 'sklearn', 'dtwithlc', plotname)

main()





