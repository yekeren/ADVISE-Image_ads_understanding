
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import nltk
import numpy as np
import tensorflow as tf

from tensorflow import logging

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

from protos import ads_mem_examples_pb2
from readers.utils import load_vocab
from readers.utils import load_raw_annots
from readers.utils import load_action_reason_annots
from readers.utils import load_densecap_annots
from readers.utils import load_symbol_cluster
from readers.utils import tokenize


def encode_and_pad_sentences(
    vocab, sentences, max_sents_per_image, max_sent_len):
  """Encodes and pads sentences.

  Args:
    vocab: a dict mapping from word to id.
    sentences: a list of python string.
    max_sents_per_image: maximum number of sentences.
    max_sent_len: maximum length of sentence.

  Returns:
    num_sents: a integer denoting the number of sentences.
    sent_mat: a [max_sents_per_image, max_sent_len] numpy array pad with zero.
    sent_len: a [max_sents_per_image] numpy array indicating the length of each
      sentence in the matrix.
  """
  encode_fn = lambda x: [vocab.get(w, 0) for w in tokenize(x)]

  sentences = [encode_fn(s) for s in sentences]
  sent_mat = np.zeros((max_sents_per_image, max_sent_len), np.int32)
  sent_len = np.zeros((max_sents_per_image,), np.int32)

  for index, sent in enumerate(sentences[:max_sents_per_image]):
    sent_len[index] = min(max_sent_len, len(sent))
    sent_mat[index][:sent_len[index]] = sent[:sent_len[index]]

  return len(sentences), sent_mat, sent_len


def _get_data_placeholders(config, split):
  """Returns data placeholder to feed the dataset.

  Args:
    config: an instance of ads_mem_examples_pb2.AdsMemExamples.

  Returns:
    data_placeholders: a dict mapping from name to placeholders.
    feed_dict: a dict mapping from name to data.
  """
  # Create placeholders.
  data_placeholders = {
    'image_id': 
      tf.placeholder(tf.string, [None]),
    'img_features': 
      tf.placeholder(tf.float32, [None, config.feature_dimensions]),
    'roi_features': 
      tf.placeholder(tf.float32, [None, 
          config.number_of_regions, config.feature_dimensions]),
    'number_of_statements':
      tf.placeholder(tf.int32, [None]),
    'statement_strings': 
      tf.placeholder(tf.int32, [None, 
          config.max_stmts_per_image, config.max_stmt_len]),
    'statement_lengths': 
      tf.placeholder(tf.int32, [None, config.max_stmts_per_image]),
    'number_of_symbols':
      tf.placeholder(tf.int32, [None]),
    'symbols':
      tf.placeholder(tf.int32, [None, config.max_symbols_per_image]),
  }
  if not config.use_single_densecap:
    data_placeholders.update({ 
        'number_of_densecaps':
          tf.placeholder(tf.int32, [None]),
        'densecap_strings': 
          tf.placeholder(tf.int32, [None, 
              config.max_densecaps_per_image, config.max_densecap_len]),
        'densecap_lengths': 
          tf.placeholder(tf.int32, [None, 
            config.max_densecaps_per_image]), 
        })
  else:
    data_placeholders.update({ 
        'number_of_densecaps':
          tf.placeholder(tf.int32, [None]),
        'densecap_strings': 
          tf.placeholder(tf.int32, [None, 
              1, config.max_densecaps_per_image * config.max_densecap_len]),
        'densecap_lengths': 
          tf.placeholder(tf.int32, [None, 1]), 
        })

  if split != 'train':
    data_placeholders.update({
      'eval_statement_strings': 
        tf.placeholder(tf.int32, [None, 
            config.number_of_val_stmts_per_image, config.max_stmt_len]),
      'eval_statement_lengths': 
        tf.placeholder(tf.int32, [None, config.number_of_val_stmts_per_image]),
      })

  # Load annotations and image features.
  assert tf.gfile.Exists(config.image_feature_path)
  assert tf.gfile.Exists(config.region_feature_path)
  assert tf.gfile.Exists(config.statement_vocab_path)
  assert tf.gfile.Exists(config.statement_annot_path)
  assert tf.gfile.Exists(config.densecap_vocab_path)
  assert tf.gfile.Exists(config.densecap_annot_path)
  assert tf.gfile.Exists(config.symbol_annot_path)
  assert tf.gfile.Exists(config.symbol_cluster_path)

  # Image features.
  start = time.time()
  image_features = np.load(config.image_feature_path).item()
  region_features = np.load(config.region_feature_path).item()
  logging.info('Image features are loaded, cost=%is, img_len=%i, roi_len=%i.', 
      time.time() - start, len(image_features), len(region_features))

  # Action-reason annotations.
  start = time.time()
  stmt_annots = load_action_reason_annots(config.statement_annot_path)
  logging.info('Annotations are loaded, cost=%is, path=%s, len=%i.', 
      time.time() - start, config.statement_annot_path, len(stmt_annots))

  stmt_vocab = load_vocab(config.statement_vocab_path)
  logging.info('Load vocab from %s, vocab_size=%i', 
      config.statement_vocab_path, len(stmt_vocab))

  # Densecap annotations.
  start = time.time()
  dense_annots = load_densecap_annots(
      config.densecap_annot_path, config.max_densecaps_per_image)
  logging.info('Dense annotations are loaded, cost=%is, path=%s, len=%i.', 
      time.time() - start, config.densecap_annot_path, len(dense_annots))

  dense_vocab = load_vocab(config.densecap_vocab_path)
  logging.info('Load vocab from %s, vocab_size=%i', 
      config.densecap_vocab_path, len(dense_vocab))

  # Symbol annotations.
  start = time.time()
  symbol_annots = load_raw_annots(config.symbol_annot_path)
  logging.info('Symbol annotations are loaded, cost=%is, path=%s, len=%i.', 
      time.time() - start, config.symbol_annot_path, len(symbol_annots))
  word_to_id, id_to_symbol = load_symbol_cluster(config.symbol_cluster_path)

  # Initialize feed_dict.
  feed_dict = {
    'image_id': [],
    'img_features': [],
    'roi_features': [],
    'number_of_statements': [],
    'statement_strings': [],
    'statement_lengths': [],
    'number_of_densecaps': [],
    'densecap_strings': [],
    'densecap_lengths': [],
    'number_of_symbols': [],
    'symbols': [],
  }
  if split != 'train':
    feed_dict.update({
        'eval_statement_strings': [],
        'eval_statement_lengths': [],
        })

  total_images = total_statements = 0

  # Split training data for validation purpose.
  stmt_annots = stmt_annots.items()
  if split == 'valid':
    stmt_annots = stmt_annots[:config.number_of_val_examples]
  elif split == 'train':
    stmt_annots = stmt_annots[config.number_of_val_examples:]
  logging.info('Processing %i %s records...', len(stmt_annots), split)

  if config.debug:
    logging.warn('DEBUG MODE!!!!!!!')
    stmt_annots = stmt_annots[:100]

  for index, (image_id, annot) in enumerate(stmt_annots):
    # Pad action-reason.
    (number_of_statements, statement_strings, statement_lengths
     ) = encode_and_pad_sentences(
       stmt_vocab, 
       annot['pos_examples'],
       config.max_stmts_per_image,
       config.max_stmt_len)

    # Pad densecap.
    if not config.use_single_densecap:
      (number_of_densecaps, densecap_strings, densecap_lengths
       ) = encode_and_pad_sentences(
         dense_vocab, 
         dense_annots[image_id],
         config.max_densecaps_per_image, 
         config.max_densecap_len)
    else:  # Concatenate all densecaps to form a single sentence.
      dense_string_concat = ' '.join(dense_annots[image_id])
      (number_of_densecaps, densecap_strings, densecap_lengths
       ) = encode_and_pad_sentences(
         dense_vocab, 
         [dense_string_concat],
         1, 
         config.max_densecap_len * config.max_densecaps_per_image)

    # Pad symbols.
    symbols = symbol_annots.get(image_id, [])
    number_of_symbols = len(symbols)
    symbols += [0] * config.max_symbols_per_image
    symbols = symbols[:config.max_symbols_per_image]

    feed_dict['image_id'].append(image_id)
    feed_dict['img_features'].append(image_features[image_id])
    feed_dict['roi_features'].append(region_features[image_id])
    feed_dict['number_of_statements'].append(
        np.array(number_of_statements, dtype=np.int32))
    feed_dict['statement_strings'].append(statement_strings)
    feed_dict['statement_lengths'].append(statement_lengths)
    feed_dict['number_of_densecaps'].append(
        np.array(number_of_densecaps, dtype=np.int32))
    feed_dict['densecap_strings'].append(densecap_strings)
    feed_dict['densecap_lengths'].append(densecap_lengths)
    feed_dict['number_of_symbols'].append(
        np.array(number_of_symbols, dtype=np.int32))
    feed_dict['symbols'].append(np.array(symbols))

    if split != 'train':
      # Pad strings for evaluation purpose.
      (number_of_eval_statements, eval_statement_strings, eval_statement_lengths
       ) = encode_and_pad_sentences(
         stmt_vocab, 
         annot['all_examples'],
         config.number_of_val_stmts_per_image,
         config.max_stmt_len)
      assert number_of_eval_statements == config.number_of_val_stmts_per_image
      feed_dict['eval_statement_strings'].append(eval_statement_strings)
      feed_dict['eval_statement_lengths'].append(eval_statement_lengths)

    total_images += 1
    total_statements += number_of_statements

    if index % 1000 == 0:
      logging.info('Load on %i/%i', index, len(stmt_annots))

  logging.info('Load %i images with %i statements.', 
      total_images, total_statements)

  # Legacy: GPU or CPU mode.
  if config.data_provider_mode == ads_mem_examples_pb2.AdsMemExamples.FROM_CPU:
    for k, v in feed_dict.items():
      feed_dict[data_placeholders[k]] = np.stack(v)
      del feed_dict[k]
    return data_placeholders, feed_dict

#  elif config.data_provider_mode == ads_mem_examples_pb2.AdsMemExamples.FROM_GPU:
#    data_tensors = {}
#    for k, v in feed_dict.items():
#      data_tensors[k] = tf.constant(np.stack(v))
#    return data_tensors, {}

  raise ValueError('Unknown mode %i.' % config.data_provider_mode)


def _map_func(dataset):
  """Transforms the shape of dataset.

  Args:
    dataset: a dict mapping from name to tensors.

  Returns:
    dataset: a dict mapping from name to transformed tensors.
  """
  # For testing data without positive statements, always return the 0-th entry.
  # Else return a random entry.
  pred = tf.greater(dataset['number_of_statements'], 0)
  index = tf.cond(pred,
      true_fn=lambda: tf.random_uniform(
        shape=[], maxval=dataset['number_of_statements'], dtype=tf.int32),
      false_fn=lambda: tf.constant(0, shape=[]))
  dataset['statement_strings'] = dataset['statement_strings'][index]
  dataset['statement_lengths'] = dataset['statement_lengths'][index]

  index = tf.random_uniform(
      shape=[], maxval=dataset['number_of_densecaps'], dtype=tf.int32)
  dataset['densecap_strings'] = dataset['densecap_strings'][index]
  dataset['densecap_lengths'] = dataset['densecap_lengths'][index]
  return dataset


def get_examples(config, split='train'):
  """Gets batched tensors for train/valid/test data.

  Args:
    config: an instance of AdsExample proto.
    split: train/valid/test

  Returns:
    tensor_dict: a dictionary mapping names to tensors.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, ads_mem_examples_pb2.AdsMemExamples):
    raise ValueError(
        'Config has to be an instance of ads_mem_examples_pb2.AdsMemExamples.')

  if not split in ['train', 'valid', 'test']:
    raise ValueError('Split has to be one of train, valid, or test.')

  # Create dataset object.
  data_placeholders, feed_dict = _get_data_placeholders(config, split)

  dataset = tf.data.Dataset.from_tensor_slices(data_placeholders)
  if split != 'test':
    dataset = dataset.filter(lambda x: x['number_of_statements'] > 0)
  dataset = dataset.map(_map_func)
  if split == 'train':
    dataset = dataset.repeat()
  dataset = dataset.shuffle(config.shuffle_buffer_size)
  dataset = dataset.batch(config.batch_size)

  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  # Initialize feed_dict.
  def _init_fn(sess):
    """Feeds data to the dataset placeholders.

    Args:
      sess: a tf.Session object.
    """
    logging.info('Initializing iterator...')
    sess.run(iterator.initializer, feed_dict=feed_dict)

  return next_element, _init_fn
