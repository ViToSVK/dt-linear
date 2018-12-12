# General parser

from parser_arff import parse_arff


def parse_timeprof(folder, filename):
  if '.arff' in filename:
    return parse_arff(folder, filename)
  assert(False and 'Unsupported file type for parsing')
