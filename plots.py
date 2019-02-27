# Generate plots from results

from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


def collect_stats(filename):
  stats = {'names': [], 'sklearn': [], 'dtwithlc': [],
           'max_sk': 1, 'max_lc': 1}

  namefollows = False
  disregard = True
  c_name, c_sk, c_dtlc = '', -1, -1
  for line in open('results/reports/%s' % filename, 'r'):
    if '===' in line:
      if not disregard:
        stats['names'].append(c_name)
        stats['sklearn'].append(c_sk)
        stats['dtwithlc'].append(c_dtlc)
        if c_sk > stats['max_sk']:
          stats['max_sk'] = c_sk
        if c_dtlc > stats['max_lc']:
          stats['max_lc'] = c_dtlc
        if c_sk < c_dtlc:
          print(c_name)
      namefollows = True
      continue
    if namefollows:
      namefollows = False
      c_name = line.strip()
      c_sk, c_dtlc = -1, -1
      disregard = False
      continue
    if 'nodes' in line and 'correct' in line and 'time' in line:
      if 'sklearn' in line:
        assert(c_sk == -1)
        l = line.split()
        c_sk = int(l[1])
        if l[3] != 'T':
          print('SKLEARN TREE FOR %s IS NOT CORRECT!' % c_name)
          disregard = True
      elif 'dtwithlc' in line:
        assert(c_dtlc == -1)
        l = line.split()
        c_dtlc = int(l[1])
        if l[3] != 'T':
          print('DTWITHLC TREE FOR %s IS NOT CORRECT!' % c_name)
          disregard = True

  for stat in ['names', 'sklearn', 'dtwithlc']:
    stats[stat] = np.array(stats[stat])
  return stats


def all_plot(stats, plotname):
  SHIFT = 1.1
  l = stats['sklearn'].size
  m = max(stats['max_sk'], stats['max_lc'])
  assert(l == stats['dtwithlc'].size)
  plt.figure(num=None, figsize=(15, 6), dpi=150, facecolor='w', edgecolor='k')
  plt.scatter(range(l), stats['sklearn']+SHIFT,
              color='b', edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.scatter(range(l), stats['dtwithlc']+SHIFT,
              color='r', edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.ylim([1, m * 1.1])
  plt.xlim([-(l-1) * 0.007, (l-1) * 1.007])
  ax = plt.gca()
  ax.set_yscale('log', basey=np.e)
  y_ticks = np.logspace(np.log(SHIFT), np.log(m+SHIFT), num=10, base=np.e)
  ax.set_yticks(y_ticks)
  ax.set_yticklabels(np.ceil(y_ticks-SHIFT).astype(int))
  plt.xlabel('Benchmark number', fontsize=18)
  plt.ylabel('Decision tree size', fontsize=18)
  blue_patch = mpatches.Patch(color='blue', label='Scikit-learn')
  red_patch = mpatches.Patch(color='red', label='DTwithLC')
  plt.legend(handles=[blue_patch, red_patch], loc=4)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_all_plot.png' % plotname, bbox_inches='tight')


def versus_plot(stats, plotname):
  plt.figure(num=None, figsize=(12, 6), dpi=150, facecolor='w', edgecolor='k')
  plt.scatter(stats['sklearn'], stats['dtwithlc'],
              color='b', edgecolor='black', linewidth=0.6, alpha=0.8)
  plt.plot([0, 100000], [0, 100000], color='r', linewidth=2.0, alpha=0.4)
  plt.plot([0, 100000], [0, 10000], color='r', linewidth=2.0, alpha=0.2)
  plt.xlim([-1, stats['max_sk'] * 1.01])
  plt.ylim([-1, stats['max_lc'] * 1.015])
  plt.xlabel('Scikit-learn size', fontsize=18)
  plt.ylabel('DTwithLC size', fontsize=18)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_vs_plot.png' % plotname, bbox_inches='tight')


def ratio_plot(stats, plotname):
  l = stats['sklearn'].size
  assert(l == stats['dtwithlc'].size)
  plt.figure(num=None, figsize=(12, 6), dpi=150, facecolor='w', edgecolor='k')
  plt.clf()
  plt.scatter(range(l), stats['dtwithlc'] / stats['sklearn'],
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
  plt.ylabel('DTwitLC/Scikit-learn size ratio', fontsize=18)
  if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
  plt.savefig('results/plots/%s_ratio_plot.png' % plotname, bbox_inches='tight')


def main():
  if not os.path.isdir('results/reports'):
    return
  for filename in sorted(os.listdir('results/reports')):
    if 'report_' in filename and '.txt' in filename:
      print(filename)
      stats = collect_stats(filename)
      plotname = filename.replace('report_', '').replace('.txt', '')
      all_plot(stats, plotname)
      versus_plot(stats, plotname)
      ratio_plot(stats, plotname)


main()





