# Run experiments and collect the output

import numpy as np
import os, sys, time, operator
import subprocess, shlex, signal
from threading import Timer


# Run with assertions?
ASSERTIONS = True
# Timeout in seconds (240 hours)
TIMEOUT = 240*60*60
# A flag that tracks whether a timeout occurred
TIMED_OUT = False
# Path to the report file - set up from sys.argv[1]
REPORT_PATH = None
# Algorithms
ALGOS = ['sklearn', 'baseline', 'lc_ent', 'lc_auc_reg', 'lc_auc_clf']


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


# Kills the process p if a timeout occurred - sets the flag TIMED_OUT
def kill_process(p):
  global TIMED_OUT
  TIMED_OUT = True
  os.killpg(os.getpgid(p.pid), signal.SIGTERM)
  #print('killed')


# Runs a command and sets a timer for TIMEOUT
def run_command(cmd):
  global TIMED_OUT
  TIMED_OUT = False
  p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
  global TIMEOUT
  timer = Timer(TIMEOUT, kill_process, [p])
  stdout, stderr = '', ''
  try:
    timer.start()
    stdout, stderr = p.communicate()
  finally:
    timer.cancel()
  return stdout, stderr


# Decode and parse output from stdout after running a command
def parse_output(output):
  result = {'info': ''}
  global ALGOS
  for algo in ALGOS:
    result[algo] = {'time': -1., 'nodes': -1, 'depth': -1., 'correct': 'F'}
  try:
    parts = output.split(b'\n')
    for pp in parts:
      p = pp.decode('utf-8')
      if 'samples,' in p and 'features' in p:
        result['info'] = p.strip()
      else:
        for algo in ALGOS:
          if algo in p:
            if '_timeprof' in p:
              # Tree building time
              result[algo]['time'] = float(p.split()[3])
            elif '_nodes' in p:
              # Number of inner nodes
              result[algo]['nodes'] = int(p.split(':')[1].strip())
            elif '_wavg_depth' in p:
              result[algo]['depth'] = float(p.split(':')[1].strip())
            elif '_correct' in p:
              cor = p.split(':')[1].strip()
              if cor == 'True':
                result[algo]['correct'] = 'T'
            break
  except:
    pass

  return result


# Decode 'bytes' output from stderr after running a command
def decode_output(output):
  decoded = []
  try:
    parts = output.split(b'\n')
    for pp in parts:
      dec = pp.decode('utf-8')
      if 'ConvergenceWarning' not in dec:
        decoded.append(dec)
  except:
    decoded.append('Failed to parse the output')
  global TIMED_OUT
  if TIMED_OUT:
    global TIMEOUT
    decoded.append('Timed out after: %s' % pretty_time(TIMEOUT))
  return '\n'.join(decoded)


def write_output(out, err):
  global REPORT_PATH
  f = open(REPORT_PATH,'a')
  f.write('='*105 + '\n')
  f.write(out['filename'] + '\n')
  f.write('-'*105 + '\n')
  f.write(out['info'] + '\n')
  global ALGOS
  for algo in ALGOS:
    f.write(algo)
    d = 40 - len(algo)
    f.write(('{:>%d}' % d).format('%s nodes' % out[algo]['nodes']))
    f.write('{:>20}'.format('%s correct' % out[algo]['correct']))
    f.write('{:>22}'.format('%s depth' % out[algo]['depth']))
    f.write('{:>22}'.format('%s time\n' % out[algo]['time']))
  f.write('%s\n' % err)
  f.close()


def run_one(folder, filename):
  command_parts = ['python' if ASSERTIONS else 'python -O',
                   'main.py', folder, filename, 'do_sklearn', 'do_auc_reg']
  command = ' '.join(command_parts)
  stdout, stderr = run_command(command)

  out = parse_output(stdout)
  out['filename'] = filename
  err = decode_output(stderr)
  print('STDOUT')
  print(out)
  print('STDERR')
  print(err)

  write_output(out, err)


def info():
  print('---')
  print('experiments.py runs main.py on all datasets in a given folder,')
  print('collects the statistics and saves them in "results/reports".')
  print('First two arguments to experiments.py are required, third optional.')
  print('1st argument: folder name that contains "games" OR "mdps"')
  print('2nd argument: "debug" OR "release" (whether to check code assertions)')
  print('3rd optional argument: "replace" (whether to replace old report file)')


def run_all():
  if len(sys.argv) < 3 or len(sys.argv) > 4:
    print('ARGUMENTS - FOLDER NAME, "debug" OR "release", (optional) "replace"')
    info()
    return

  if 'games' not in sys.argv[1] and 'mdps' not in sys.argv[1]:
    print('FOLDER NAME MUST CONTAIN "games" OR "mdps", PROVIDED: %s' % sys.argv[1])
    info()
    return
  if not os.path.isdir(sys.argv[1]):
    print('PROVIDED FOLDER DOES NOT EXIST: %s' % sys.argv[1])
    info()
    return

  global ASSERTIONS
  if sys.argv[2] == 'debug':
    ASSERTIONS = True
  elif sys.argv[2] == 'release':
    ASSERTIONS = False
  else:
    print('SECOND ARGUMENT SHOULD BE "debug" OR "release", GIVEN: %s' % sys.argv[2])
    info()
    return

  replace = False
  if len(sys.argv) == 4:
    if sys.argv[3] != 'replace':
      print('OPTIONAL THIRD ARGUMENT SHOULD BE "replace", GIVEN: %s' % sys.argv[3])
      info()
      return
    replace = True

  if not os.path.exists('results/reports'):
    os.makedirs('results/reports')
  global REPORT_PATH
  subfolder = sys.argv[1]
  if subfolder.endswith('/'):
    subfolder = subfolder[:-1]
  subfolder = subfolder.split('/')[-1]
  REPORT_PATH = 'results/reports/report_%s.txt' % subfolder
  if replace:
    try:
      os.remove(REPORT_PATH)
    except OSError: pass

  for filename in sorted(os.listdir(sys.argv[1])):
    if ('.arff' in filename and 'games' in sys.argv[1]) or (
        '.prism' in filename and 'mdps' in sys.argv[1]) or (
        '.iostr' in filename and 'mdps' in sys.argv[1]):
      print(filename)
      run_one(sys.argv[1], filename)


run_all()

