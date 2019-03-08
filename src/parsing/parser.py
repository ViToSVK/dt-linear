# General parser

from parser_arff import parse_arff
from parser_prism import parse_prism
from parser_iostr import parse_iostr


def parse_timeprof(folder, filename):
  if '.arff' in filename:
    return parse_arff(folder, filename)
  if '.prism' in filename:
    return parse_prism(folder, filename)
  if '.iostr' in filename:
    return parse_iostr(folder, filename)
  assert(False and 'Unsupported file type for parsing')

