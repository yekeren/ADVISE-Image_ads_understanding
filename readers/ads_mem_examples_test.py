
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow import logging
from google.protobuf import text_format

import ads_mem_examples
from protos import ads_mem_examples_pb2
from readers.utils import load_symbol_cluster

slim = tf.contrib.slim

def _load_vocab(filename):
  """Loads vocabulary.

  Args:
    filename: path to the vocabulary file.

  Returns:
    a list mapping from id to word.
  """
  with open(filename, 'r') as fp:
    vocab = ['UNK'] + [x.strip('\n').split('\t')[0] for x in fp.readlines()]
  return vocab


class AdsMemExamplesTest(tf.test.TestCase):
  def setUp(self):
    logging.set_verbosity(tf.logging.INFO)

    config_string = """
      debug: true
      batch_size: 32
      feature_dimensions: 1536
      max_stmts_per_image: 5
      max_stmt_len: 21
      number_of_regions: 10
      shuffle_buffer_size: 1000
      max_densecap_len: 9
      max_densecaps_per_image: 5
      max_symbols_per_image: 7
      use_single_densecap: true
      image_feature_path: "output/img_features_train.npy"
      region_feature_path: "output/roi_features_train.npy"
      statement_vocab_path: "output/action_reason_vocab_200d.txt"
      statement_annot_path: "data/train/QA_Combined_Action_Reason_train.json"
      densecap_vocab_path: "output/densecap_vocab_200d.txt"
      densecap_annot_path: "output/densecap_train.json"
      symbol_annot_path: "output/symbol_train.json"
      symbol_cluster_path: "data/additional/clustered_symbol_list.json"
    """
    self.default_config = ads_mem_examples_pb2.AdsMemExamples()
    text_format.Merge(config_string, self.default_config)

    self.stmt_vocab = _load_vocab(self.default_config.statement_vocab_path)
    self.densecap_vocab = _load_vocab(self.default_config.densecap_vocab_path)
    word_to_id, id_to_symbol = load_symbol_cluster(self.default_config.symbol_cluster_path)
    self.symbol_vocab = id_to_symbol

  def test_get_examples(self):
    g = tf.Graph()
    with g.as_default():
      examples, init_fn = ads_mem_examples.get_examples(
          self.default_config, split='valid')

    with self.test_session(graph=g) as sess:
      init_fn(sess)
      examples = sess.run(examples)
      self.assertEqual(examples['image_id'].shape, (32,))
      self.assertEqual(examples['img_features'].shape, (32, 1536))
      self.assertEqual(examples['roi_features'].shape, (32, 10, 1536))
      self.assertEqual(examples['number_of_statements'].shape, (32,))
      self.assertEqual(examples['statement_lengths'].shape, (32,))
      self.assertEqual(examples['statement_strings'].shape, (32, 21))
      self.assertEqual(examples['number_of_densecaps'].shape, (32,))
      self.assertEqual(examples['densecap_lengths'].shape, (32,))
      self.assertEqual(examples['densecap_strings'].shape, (32, 45))
      self.assertEqual(examples['number_of_symbols'].shape, (32,))
      self.assertEqual(examples['symbols'].shape, (32, 7))
      self.assertEqual(examples['eval_statement_lengths'].shape, (32, 15))
      self.assertEqual(examples['eval_statement_strings'].shape, (32, 15, 21))

    for i in xrange(32):
      statement = examples['statement_strings'][i][:examples['statement_lengths'][i]]
      densecap = examples['densecap_strings'][i][:examples['densecap_lengths'][i]]
      symbol = examples['symbols'][i][:examples['number_of_symbols'][i]]
      logging.info('=' * 128)
      logging.info(examples['image_id'][i])
      logging.info(' '.join([self.stmt_vocab[x] for x in statement]))
      logging.info(' '.join([self.densecap_vocab[x] for x in densecap]))
      logging.info(' '.join([self.symbol_vocab[x] for x in symbol]))
      for j in xrange(15):
        statement = examples['eval_statement_strings'][i, j][:examples['eval_statement_lengths'][i, j]]
        logging.info('%i: %s', j, ' '.join([self.stmt_vocab[x] for x in statement]))

if __name__ == '__main__':
    tf.test.main()
