
import os
import sys
import json
import argparse
import numpy as np


def _load_vocab(filename):
  """Loads vocabulary from file.

  Args:
    filename: path to the vocabulary file.

  Returns:
    vocab: a list of words
  """
  vocab = []
  with open(filename, 'r') as fp:
    for line in fp.readlines():
      word, freq = line.strip('\n').split('\t')
      vocab.append((word, int(freq)))
  print >> sys.stderr, 'Load %i words from %s.' % (len(vocab), filename)
  return vocab


def _load_data(filename):
  ''' Load a pre-trained word2vec file.

  Args:
    filename: path to the word2vec file.

  Returns:
    a numpy matrix.
  '''
  with open(filename, 'r') as fp:
    lines = fp.readlines()

  # Get the number of words and embedding size.
  num_words = len(lines)
  embedding_size = len(lines[0].strip('\n').split()) - 1

  word2vec = {}
  for line_index, line in enumerate(lines):
    items = line.strip('\n').split()
    word, vec = items[0], map(float, items[1:])
    assert len(vec) == embedding_size

    word2vec[word] = np.array(vec)
    if line_index % 10000== 0:
      print >> sys.stderr, 'On load %s/%s' % (line_index, len(lines))
  return word2vec


def _export_data(word2vec, vocab, filename_emb, filename_vocab, min_tf=10):
  """Export word embeddings to npz file.

  Args:
    word2vec: a mapping from word to vector.
    vocab: a list of (word, freq) tuples.
    filename_emb: the name of output embedding file.
    filename_vocab: the name of output vocab file.
  """
  dims = word2vec['the'].shape[0]

  # 0 - UNK.
  vecs, words = [], []
  vecs.append(args.init_width * (np.random.rand(dims) * 2 - 1))

  count = 0
  for word, freq in vocab:
    if word in word2vec:
      vec = word2vec[word]
    elif freq >= min_tf:
      count += 1
      vec = args.init_width * (np.random.rand(dims) * 2 - 1)
      print >> sys.stderr, 'Unknown word: %s, freq=%i.' % (word, freq)
    else:
      continue

    vecs.append(vec)
    words.append(word)
  vecs = np.stack(vecs, axis=0)

  with open(filename_emb, 'wb') as fp:
    np.save(fp, vecs)

  with open(filename_vocab, 'w') as fp:
    fp.write('\n'.join(words))

  print >> sys.stderr, 'Shape of word2vec:', vecs.shape
  print >> sys.stderr, 'Unknown words: %i/%i(%.2lf%%).' % (
      count, len(words), count * 100.0 / len(words))


def main(args):
  vocab = _load_vocab(args.vocab_path)
  word2vec = _load_data(args.data_path)

  _export_data(word2vec, vocab, 
      args.output_emb_path, args.output_vocab_path, args.min_tf)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--vocab_path', type=str,
      default='output/action_reason_vocab.txt', 
      help='Path to the action-reason vocab file.')
  parser.add_argument(
      '--data_path', type=str,
      default='zoo/glove.6B.200d.txt',
      help='Path to the GLOVE data file.')
  parser.add_argument(
      '--output_emb_path', type=str,
      default='output/action_reason_vocab_200d.npy', 
      help='Path to the output embedding file.')
  parser.add_argument(
      '--output_vocab_path', type=str,
      default='output/action_reason_vocab_200d.txt', 
      help='Path to the output vocab file.')
  parser.add_argument(
      '--init_width', type=float,
      default=0.03, 
      help='Initial random width for unknown word.')
  parser.add_argument(
      '--min_tf', type=int,
      default=10, 
      help='The minimum word frequency for the word that are not in GLOVE.')

  args = parser.parse_args()
  assert os.path.isfile(args.vocab_path)
  assert os.path.isfile(args.data_path)

  print >> sys.stderr, 'parsed input parameters:'
  print >> sys.stderr, json.dumps(vars(args), indent=2)

  main(args)

  print >> sys.stderr, 'Done'
  exit(0)
