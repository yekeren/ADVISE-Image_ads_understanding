
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
  word_to_id, id_to_symbol = load_symbol_cluster(args.symbol_cluster_path)
  print >> sys.stderr, 'Load %i pairs of mapping.' % (len(word_to_id))
  print >> sys.stderr, 'Symbol list: \n%s' % (json.dumps(id_to_symbol, indent=2))

  id_to_symbol = sorted(id_to_symbol.iteritems(), lambda x, y: cmp(x[0], y[0]))
  with open(args.output_vocab_path, 'w') as fp:
    for symbol_id, symbol in id_to_symbol:
      if symbol_id != 0:
        fp.write('%s\t%i\n' % (symbol, 999))

  print >> sys.stderr, 'Done'

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--symbol_cluster_path', type=str,
      default='data/additional/clustered_symbol_list.json', 
      help='Path to the symbol clustering json file.')
  parser.add_argument(
      '--output_vocab_path', type=str,
      default='output/symbol_vocab.txt', 
      help='Path to the output vocab file.')

  args = parser.parse_args()
  assert os.path.isfile(args.symbol_cluster_path)

  print >> sys.stderr, 'parsed input parameters:'
  print >> sys.stderr, json.dumps(vars(args), indent=2)

  main(args)

  exit(0)
