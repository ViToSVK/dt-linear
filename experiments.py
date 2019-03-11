# Run experiments and collect the output

import numpy as np
import os, sys, time, operator
import subprocess, shlex, signal
from threading import Timer


# Run with assertions?
ASSERTIONS = True
# Timeout in seconds (4 hours)
TIMEOUT = 4*60*60
# A flag that tracks whether a timeout occurred
TIMED_OUT = False
# Path to the report file - set up from sys.argv[1]
REPORT_PATH = None
# Algorithms
ALGOS = ['sklearn', 'baseline', 'dtwithlc', 'dtauclc']


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
  result = {}
  global ALGOS
  for algo in ALGOS:
    result[algo] = {'time': -1., 'nodes': -1, 'correct': 'F'}
  try:
    parts = output.split(b'\n')
    for pp in parts:
      p = pp.decode('utf-8')
      for algo in ALGOS:
        if algo in p:
          if '_timeprof' in p:
            # Tree building time
            result[algo]['time'] = float(p.split()[3])
          elif '_nodes' in p:
            # Number of inner nodes
            result[algo]['nodes'] = int(p.split(':')[1].strip())
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


def run_one(folder, filename):
  command_parts = ['python' if ASSERTIONS else 'python -O',
                   'main.py', folder, filename]
  command = ' '.join(command_parts)
  stdout, stderr = run_command(command)

  out = parse_output(stdout)
  err = decode_output(stderr)
  print('STDOUT')
  print(out)
  print('STDERR')
  print(err)

  global REPORT_PATH
  f = open(REPORT_PATH,'a')
  f.write('='*100 + '\n')
  f.write(filename + '\n')
  f.write('-'*100 + '\n')
  global ALGOS
  for algo in ALGOS:
    f.write(algo)
    d = 40 - len(algo)
    f.write(('{:>%d}' % d).format('%s nodes' % out[algo]['nodes']))
    f.write('{:>20}'.format('%s correct' % out[algo]['correct']))
    f.write('{:>30}'.format('%s time\n' % out[algo]['time']))
  f.write('%s\n' % err)
  f.close()


def run_all():
  if len(sys.argv) < 3 or len(sys.argv) > 4:
    print('ARGUMENTS - FOLDER NAME, "debug" OR "release", (optional) "replace"')
    return

  if 'games' not in sys.argv[1] and 'mdps' not in sys.argv[1]:
    print('FOLDER NAME MUST CONTAIN "games" OR "mdps", PROVIDED: %s' % sys.argv[1])
    return
  if not os.path.isdir('datasets/%s' % sys.argv[1]):
    print('PROVIDED FOLDER IS NOT IN DATASETS: %s' % sys.argv[1])
    return

  global ASSERTIONS
  if sys.argv[2] == 'debug':
    ASSERTIONS = True
  elif sys.argv[2] == 'release':
    ASSERTIONS = False
  else:
    print('SECOND ARGUMENT SHOULD BE "debug" OR "release", GIVEN: %s' % sys.argv[2])
    return

  replace = False
  if len(sys.argv) == 4:
    if sys.argv[3] != 'replace':
      print('OPTIONAL THIRD ARGUMENT SHOULD BE "replace", GIVEN: %s' % sys.argv[3])
      return
    replace = True

  if not os.path.exists('results/reports'):
    os.makedirs('results/reports')
  global REPORT_PATH
  REPORT_PATH = 'results/reports/report_%s.txt' % sys.argv[1]
  if replace:
    try:
      os.remove(REPORT_PATH)
    except OSError: pass

  for filename in sorted(os.listdir('datasets/%s' % sys.argv[1])):
    if ('.arff' in filename and 'games' in sys.argv[1]) or (
        '.prism' in filename and 'mdps' in sys.argv[1]) or (
        '.iostr' in filename and 'mdps' in sys.argv[1]):
      print(filename)
      run_one(sys.argv[1], filename)


run_all()

