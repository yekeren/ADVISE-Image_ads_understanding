
import os
import sys
import json
import string
import argparse

import nltk
from readers.utils import load_action_reason_annots
from readers.utils import load_symbol_cluster
from readers.utils import load_symbol_raw_annots


def main(args):
  """Main."""
  # Load symbol annotations.
  symbol_annots = load_symbol_raw_annots(args.symbol_raw_annot_path)
  print >> sys.stderr, 'Load symbol annotations for %i images.' % (len(symbol_annots))

  word_to_id, id_to_symbol = load_symbol_cluster(args.symbol_cluster_path)
  print >> sys.stderr, 'Load %i pairs of mapping.' % (len(word_to_id))
  print >> sys.stderr, 'Symbol list: \n%s' % (json.dumps(id_to_symbol, indent=2))

  results = {}
  for image_id, annots in symbol_annots.iteritems():
    symbol_set = set()
    for annot in annots:
      symbols = [s.strip() for s in annot[4].lower().split('/') if len(s.strip()) > 0]
      symbols = [word_to_id[s] for s in symbols if s in word_to_id]
      symbol_set.update(symbols)
    if len(symbol_set):
      results[image_id] = sorted(symbol_set)
    
  with open(args.output_json_path, 'w') as fp:
    fp.write(json.dumps(results))
  print >> sys.stderr, 'Export %i symbols' % (len(results))
  print >> sys.stderr, 'Done'

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--symbol_raw_annot_path', type=str,
      default='data/train/Symbols_train.json', 
      help='Path to the raw symbol annotation file.')
  parser.add_argument(
      '--symbol_cluster_path', type=str,
      default='data/additional/clustered_symbol_list.json', 
      help='Path to the symbol clustering json file.')
  parser.add_argument(
      '--output_json_path', type=str,
      default='output/symbol_train.json', 
      help='Path to the output symbol json file.')

  args = parser.parse_args()
  assert os.path.isfile(args.symbol_raw_annot_path)
  assert os.path.isfile(args.symbol_cluster_path)

  print >> sys.stderr, 'parsed input parameters:'
  print >> sys.stderr, json.dumps(vars(args), indent=2)

  main(args)

  exit(0)
