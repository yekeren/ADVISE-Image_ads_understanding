
import json
import string

import nltk

_printable = set(string.printable)
_convert_to_printable = lambda caption: filter(lambda x: x in _printable, caption)

tokenize = lambda x: nltk.word_tokenize(x.lower().replace('<unk>', 'TEXT'))


def load_vocab(filename):
  """Loads vocabulary and builds word to id mapping.

  Args:
    filename: path to the vocab file.

  Returns:
    vocab: a dict mapping from word to id, 0 is reserved for UNK.
  """
  with open(filename, 'r') as fp:
    words = [x.strip('\n').split('\t')[0] for x in fp.readlines()]
  vocab = dict(((word, i + 1) for i, word in enumerate(words)))
  vocab['UNK'] = 0
  return vocab


def load_action_reason_annots(filename):
  """Loads action reason annotations.

  Args:
    filename: path to the action-reason annotation file.

  Returns:
    annots: a dict mapping from image_id to a list of related statements.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  annots = {}
  for image_id, examples in data.iteritems():
    if len(examples) == 2:
      pos_examples, all_examples = examples
    elif len(examples) == 15:
      pos_examples, all_examples = [], examples
    else:
      raise ValueError('Invalid format.')

    assert len(all_examples) == 15
    annots[image_id] = {
      'pos_examples': [_convert_to_printable(x) for x in pos_examples],
      'all_examples': [_convert_to_printable(x) for x in all_examples],
    }
  return annots


def load_densecap_annots(filename, max_densecaps_per_image=10):
  """Loads densecap annotations.

  Args:
    filename: path to the densecap annotation file.

  Returns:
    annots: a dict mapping from image_id to a list of densecap strings.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  annots = {}
  for image_id, annot in data.iteritems():
    annots[image_id] = [region['name'] for region \
                       in annot['regions'][:max_densecaps_per_image]]
  return annots

def load_symbol_cluster(filename):
  """Loads the symbol word mapping.

  Args:
    filename: path to the symbol mapping file.

  Returns:
    word_to_id: a dict mapping from arbitrary word to symbol_id.
    id_to_symbol: a dict mapping from symbol_id to symbol name.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  word_to_id = {}
  id_to_symbol = {0: 'unclear'}

  for cluster in data['data']:
    id_to_symbol[cluster['cluster_id']] = cluster['cluster_name']
    for symbol in cluster['symbols']:
      word_to_id[symbol] = cluster['cluster_id']
  return word_to_id, id_to_symbol


def load_raw_annots(filename):
  """Loads raw annotations.

  Args:
    filename: path to the raw annotation file.

  Returns:
    data: a python dict.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())
  return data


def load_densecap_raw_annots(filename):
  """Loads densecap raw annotations.

  Args:
    filename: path to the densecap raw annotation file.

  Returns:
    annots: a dict mapping from image_id to densecap annotations.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())
  return data


def load_symbol_raw_annots(filename):
  """Loads symbol raw annotations.

  Args:
    filename: path to the symbol raw annotation file.

  Returns:
    annots: a dict mapping from image_id to symbol annotations.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())
  return data
