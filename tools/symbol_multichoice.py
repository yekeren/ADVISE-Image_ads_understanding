
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random

import nltk
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from sklearn.metrics import average_precision_score

from utils.ads_api import AdsApi

from protos import feature_extractors_pb2
from feature_extractors import builder
from feature_extractors import fc_extractor

flags = tf.app.flags
flags.DEFINE_string('api_config', 'configs/ads_api.ads.config.0', 'Path to config file.')
flags.DEFINE_string('symbol_path', 'symbols/symbols.0.npz', 'Path to the output file.')
flags.DEFINE_integer('symbol_top_k', 0, '')
flags.DEFINE_float('symbol_threshold', 0, '')

FLAGS = flags.FLAGS
slim = tf.contrib.slim

random.seed(286)
num_positive_statements = 3

PSAs = ['environment', 
     'animal_right', 
     'human_right', 
     'safety', 
     'smoking_alcohol_abuse', 
     'domestic_violence', 
     'self_esteem', 
     'political', 
     'charities']

root = '/afs/cs.pitt.edu/usr0/yekeren/work2/learning_ads_embedding/output/'

def _init_glove(file_txt, file_npz):
  with open(file_npz, 'rb') as fp:
    w2v = np.load(fp)
    pad = np.zeros([1, w2v.shape[1]], dtype=np.float32)
    w2v = np.concatenate([pad, w2v], 0)

  with open(file_txt, 'r') as fp:
    words = dict([(x.strip('\n'), w2v[i + 1]) for i, x in
        enumerate(fp.readlines())])
  words['<UNK>'] = w2v[0]
  tf.logging.info('Load %d words from glove.', len(words))
  return words

w2v = _init_glove(
    '/afs/cs.pitt.edu/usr0/yekeren/work2/learning_ads_embedding/output/glove_w2v.txt', 
    '/afs/cs.pitt.edu/usr0/yekeren/work2/learning_ads_embedding/output/glove_w2v.npz')

def _tokenize(caption):
  caption = caption.replace('<UNK>', '')
  caption = nltk.word_tokenize(caption.lower())
  return caption

def _get_embedding(words):
  emb = np.zeros((200,))
  c = 0
  for w in words:
    if w in w2v:
      emb += w2v[w]
      c += 1
  if c > 0:
    emb /= np.linalg.norm(emb)
  return emb

def _sim_fn(caption, words):
  """Measures similarity using glove embedding."""
  words_emb = _get_embedding(words)
  caption_emb = _get_embedding(caption)
  return float(1 - (words_emb * caption_emb).sum())

def _argsort(captions, words):
  captions = [(i, _tokenize(c)) for i, c in enumerate(captions)]
  random.shuffle(captions)

  captions = sorted(captions, 
      lambda x, y: cmp(_sim_fn(x[1], words), _sim_fn(y[1], words)))
  indices = [x[0] for x in captions]
  return indices

def map_probs_to_words(symbol_to_name, probs):
  if FLAGS.symbol_top_k > 0:
    indices = probs.argsort()[::-1][:FLAGS.symbol_top_k]
  elif FLAGS.symbol_threshold > 0:
    indices = np.nonzero(probs >= FLAGS.symbol_threshold)[0]
    if len(indices) == 0:
      indices = probs.argsort()[::-1][:1]
  symbols = [symbol_to_name[i] for i in indices]
  return symbols

def _multichoice(predictions):
  # Start evaluation.
  api = AdsApi(FLAGS.api_config)
  meta_list = api.get_meta_list(split='test')

  meta_list = [meta for meta in meta_list if 'statements' in meta]
  tf.logging.info('Start to evaluate %s records.', len(meta_list))

  symbol_to_name = api.get_symbol_to_name()

  recalls = {3: [], 5: [], 10:[]}
  recalls_cat = {}
  recalls_type = {}

  minrank = []
  minrank_cat = {}
  minrank_type = {}

  topics = sorted([v for k, v in api.get_topic_to_name().iteritems()])
  for topic in topics:
    recalls_cat.setdefault(topic, {3: [], 5: [], 10:[]})
    minrank_cat.setdefault(topic, [])

  num_eval = 0
  for meta_index, meta in enumerate(meta_list):
    if meta_index % 100 == 0:
      tf.logging.info('On image %d/%d', meta_index, len(meta_list))

    image_id = meta['image_id']
    if image_id in predictions:
      # Evaluation.
      num_eval += 1
      statements = meta['statements'] + meta['negative_statements']
      #statements = meta['statements'] + meta['hard_negative_statements']
      #statements = meta['slogans'] + meta['negative_slogans']

      symbols = map_probs_to_words(symbol_to_name, predictions[image_id])
      #tf.logging.info('%s: %s', image_id, ','.join(symbols))
      indices = _argsort(captions=statements, words=symbols)
      indices = np.array(indices)
      topic_name = meta.get('topic_name', 'unclear')

      for at_k in recalls.keys():
        recall = (indices[:at_k] < num_positive_statements).sum()
        recall = float(1.0 * recall)
        recalls[at_k].append(recall)
        recalls_cat.setdefault(
            topic_name, {3: [], 5: [], 10:[]})[at_k].append(recall)

        if topic_name in PSAs:
          recalls_type.setdefault('psa', {3: [], 5: [], 10:[]})[at_k].append(recall)
        elif topic_name != 'unclear':
          recalls_type.setdefault('prod', {3: [], 5: [], 10:[]})[at_k].append(recall)

      rank = np.where(indices < num_positive_statements)[0]
      minrank.append(float(rank.min()))
      minrank_cat.setdefault(topic_name, []).append(float(rank.min()))

      if topic_name in PSAs:
        minrank_type.setdefault('psa', []).append(float(rank.min()))
      elif topic_name != 'unclear':
        minrank_type.setdefault('prod', []).append(float(rank.min()))

  tf.logging.info('Evaluated %d/%d examples.', num_eval, len(meta_list))


  # New eval csv file.
  split = FLAGS.api_config.split('.')[-1]
  if FLAGS.symbol_top_k > 0:
    result_file = 'symbols/symbol.top_%d.%s.csv' % (FLAGS.symbol_top_k, split)
  else:
    result_file = 'symbols/symbol.threshold_%.2lf.%s.csv' % (FLAGS.symbol_threshold, split)

  with open(result_file, 'w') as fp:
    n_examples = num_eval
      
    line = ''
    for at_k in sorted(recalls.keys()):
      line += ',recall@%d' % (at_k) if line != '' else 'recall@%d' % (at_k)
    line += ',minrank'
    for adstype in sorted(recalls_type.keys()):
      for at_k in sorted(recalls_type[adstype].keys()):
        line += ',%s recall@%d' % (adstype, at_k)
      line += ',%s minrank' % (adstype)
    for topic in sorted(recalls_cat.keys()):
      for at_k in sorted(recalls_cat[topic].keys()):
        line += ',%s recall@%d' % (topic, at_k)
      line += ',%s minrank' % (topic)
    fp.write(line + '\n')

    for i in xrange(n_examples):
      line = ''

      # General recall and minrank.
      for at_k in sorted(recalls.keys()):
        value = '%d' % (recalls[at_k][i]) if i < len(recalls[at_k]) else ''
        line += ',' + value if line != '' else value
      value = '%d' % (minrank[i]) if i < len(minrank) else ''
      line += ',' + value

      # Type recall and minrank.
      for adstype in sorted(recalls_type.keys()):
        for at_k in sorted(recalls_type[adstype].keys()):
          value = '%d' % (recalls_type[adstype][at_k][i]) if i < len(recalls_type[adstype][at_k]) else ''
          line += ',' + value
        value = '%d' % (minrank_type[adstype][i]) if i < len(minrank_type[adstype]) else ''
        line += ',' + value

      # Category recall and minrank.
      for topic in sorted(recalls_cat.keys()):
        for at_k in sorted(recalls_cat[topic].keys()):
          value = '%d' % (recalls_cat[topic][at_k][i]) if i < len(recalls_cat[topic][at_k]) else ''
          line += ',' + value
        value = '%d' % (minrank_cat[topic][i]) if i < len(minrank_cat[topic]) else ''
        line += ',' + value

      fp.write(line + '\n')

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  assert FLAGS.symbol_top_k > 0 or FLAGS.symbol_threshold > 0

  data = np.load(FLAGS.symbol_path).item()
  image_ids, symbol_predictions = data['image_ids'], data['symbols_data']
  tf.logging.info('Load %s symbol predictions from %s, shape=%s.', 
      len(data['image_ids']), FLAGS.symbol_path, symbol_predictions.shape)

  predictions = {}
  for image_id, symbol_prediction in zip(image_ids, symbol_predictions):
    predictions[image_id] = symbol_prediction

  _multichoice(predictions)

if __name__ == '__main__':
  tf.app.run()
