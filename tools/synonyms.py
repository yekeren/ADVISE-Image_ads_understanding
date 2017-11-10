from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import cv2
import numpy as np
import tensorflow as tf

from sklearn.externals import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from utils import ads_api
from utils import vis

flags = tf.app.flags
flags.DEFINE_string('api_config', 'configs/ads_api.empty.config', 'Path to config file.')
flags.DEFINE_integer('num_clusters', 5, 'Number of clusters.')
flags.DEFINE_string('image_emb_path', 'output/image_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('word_txt_path', 'output/glove_w2v.txt', 'Path to the feature data file.')
flags.DEFINE_string('word_emb_path', 'learned_embs/word_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('densecap_emb_path', 'learned_embs/densecap_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('symbol_emb_path', 'learned_embs/symbol_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('topic', 'sports', 'A topic name.')
flags.DEFINE_string('vis_file', 'cluster.html', 'Path to the output html file.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim

stop_words = [
  'use', 'make',
  'all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'fifty',
  'four', 'not', 'own', 'through', 'yourselves', 'go', 'where', 'mill', 'only',
  'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere', 'with',
  'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming',
  'under', 'ours', 'has', 'might', 'thereafter', 'latterly', 'do', 'them',
  'his', 'around', 'than', 'get', 'very', 'de', 'none', 'cannot', 'every',
  'whether', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name',
  'several', 'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone',
  'either', 'each', 'become', 'thereupon', 'sometime', 'side', 'two', 'therein',
  'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up',
  'namely', 'towards', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out',
  'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please',
  'forty', 'per', 'its', 'everything', 'behind', 'un', 'above', 'between', 'it',
  'neither', 'seemed', 'ever', 'across', 'she', 'somehow', 'be', 'we', 'full',
  'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon',
  'nowhere', 'although', 'found', 'alone', 're', 'along', 'fifteen', 'by',
  'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could',
  'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence',
  'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly',
  'within', 'seems', 'into', 'others', 'while', 'whatever', 'except', 'down',
  'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
  'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few',
  'together', 'top', 'there', 'due', 'been', 'next', 'anyone', 'eleven', 'much',
  'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred',
  'was', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on',
  'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part', 'everywhere',
  'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty',
  'these', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below',
  'anywhere', 'nine', 'can', 'of', 'your', 'toward', 'my', 'something', 'and',
  'whereafter', 'whenever', 'give', 'almost', 'wherever', 'is', 'describe',
  'beforehand', 'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem',
  'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin',
  'no', 'perhaps', 'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein',
  'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if',
  'nobody', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon',
  'eight', 'but', 'serious', 'nothing', 'such', 'why', 'a', 'off', 'whereby',
  'third', 'i', 'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours',
  'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'whereas', 'once']

def _get_meta_list():
  api = ads_api.AdsApi(FLAGS.api_config)
  return api.get_meta_list()


def _get_data(meta_list, feature_data, split, topic):
  """Splits data to get train, valid, or test partition.

  Args:
    meta_list: a python list containing meta info for all images.
    feature_data: a mapping from image_id to embedding vector.
    split: 'train', 'valid', or 'test'
    topic: name of the topic.

  Returns:
    image_ids: a list of image_id.
    x: a [batch, emb_size] numpy array denoting features.
  """
  meta_list = [meta for meta in meta_list if meta['split'] == split]
  meta_list = [meta for meta in meta_list if meta.get('topic_name', None) == topic]
  assert len(meta_list) > 0, 'Invalid topic name %s.' % (topic)

  image_metas, x = [], []
  for meta in meta_list:
    image_metas.append(meta)
    x.append(np.expand_dims(feature_data[meta['image_id']], axis=0))

  x = np.concatenate(x, axis=0)
  tf.logging.info('%s: x_shape=%s', split, x.shape)

  return image_metas, x


def _get_feature(file_path):
  """Read feature dict from file.
  """
  feature_data = np.load(file_path).item()
  for key in feature_data.keys():
    if type(feature_data[key]) == dict:
      feature_data[key] = feature_data[key]['image_emb']
  tf.logging.info('Load %s feature vectors.', len(feature_data))
  return feature_data


def _load_vocab(file_path):
  with open(file_path, 'r') as fp:
    vocab = ['UNK'] + [x.strip('\n') for x in fp.readlines()]
  vocab_r = dict(((w, i) for i, w in enumerate(vocab)))
  return vocab, vocab_r


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #with open('raw_data/ads/annotations/clustered_symbol_list.json', 'r') as fp:
  #  data = json.loads(fp.read())
  #symbols = set()
  #for elem in data['data']:
  #  for symbol in elem['symbols']:
  #    symbols.add(symbol)

  #meta_list = _get_meta_list()
  #tf.logging.info('Load %s records.', len(meta_list))

  ## Read data file.
  #feature_data = _get_feature(FLAGS.image_emb_path)
  word_embs = np.load(FLAGS.word_emb_path)
  densecap_embs = np.load(FLAGS.densecap_emb_path)
  symbol_embs = np.load(FLAGS.symbol_emb_path)

  tf.logging.info('Shape of word embedding: %s', word_embs.shape)
  tf.logging.info('Shape of densecap embedding: %s', densecap_embs.shape)
  tf.logging.info('Shape of symbol embedding: %s', symbol_embs.shape)

  vocab, vocab_r = _load_vocab(FLAGS.word_txt_path)

  api = ads_api.AdsApi(FLAGS.api_config)
  symbol_to_name = api.get_symbol_to_name()

  top_n = 500

  with open('output/densecap_vocab.txt', 'r') as fp:
    dense_words = [line.strip('\n').split('\t')[0] for line in fp.readlines()]
    dense_words = [w for w in dense_words if not w in stop_words and w[0] >= 'a' and w[0] <= 'z']

  with open('output/vocab.txt', 'r') as fp:
    ads_words = [line.strip('\n').split('\t')[0] for line in fp.readlines()]
    ads_words = [w for w in ads_words if not w in stop_words and w[0] >= 'a' and w[0] <= 'z']

  # Synonums of ads words.
  fp = open('ads_synonyms.txt', 'w')
  for word in ads_words[:500]:
    if not word in vocab_r: continue
    emb = word_embs[vocab_r[word]]

    # Densecap synonyms.
    scores = 1 - (emb * densecap_embs).sum(1)
    indices = scores.argsort()
    densecap_synonyms = []
    for i in indices:
      if vocab[i] in dense_words:
        densecap_synonyms.append(vocab[i])
        if len(densecap_synonyms) >= 5:
          break

    # Symbols synonyms.
    scores = 1 - (emb * symbol_embs).sum(1)
    indices = scores.argsort()
    symbol_synonyms = []
    for i in indices:
      symbol_synonyms.append(symbol_to_name[i])
      if len(symbol_synonyms) >= 5:
        break

    fp.write('%s\t%s\t%s\n' % (word, 
          ','.join(densecap_synonyms), ','.join(symbol_synonyms)))
  fp.close()

  # Synonums of densecap words.
  fp = open('densecap_synonyms.txt', 'w')
  for word in dense_words[:500]:
    if not word in vocab_r: continue
    emb = densecap_embs[vocab_r[word]]

    # Ads synonyms.
    scores = 1 - (emb * word_embs).sum(1)
    indices = scores.argsort()
    ads_synonyms = []
    for i in indices:
      if vocab[i] in ads_words:
        ads_synonyms.append(vocab[i])
        if len(ads_synonyms) >= 5:
          break

    # Symbols synonyms.
    scores = 1 - (emb * symbol_embs).sum(1)
    indices = scores.argsort()
    symbol_synonyms = []
    for i in indices:
      symbol_synonyms.append(symbol_to_name[i])
      if len(symbol_synonyms) >= 5:
        break

    fp.write('%s\t%s\t%s\n' % (word, 
          ','.join(ads_synonyms), ','.join(symbol_synonyms)))
  fp.close()

  # Synonums of symbol words.
  fp = open('symbol_synonyms.txt', 'w')
  for i, word in symbol_to_name.iteritems():
    if i == 0: continue
    emb = symbol_embs[i]

    # Ads synonyms.
    scores = 1 - (emb * word_embs).sum(1)
    indices = scores.argsort()
    ads_synonyms = []
    for i in indices:
      if vocab[i] in ads_words:
        ads_synonyms.append(vocab[i])
        if len(ads_synonyms) >= 5:
          break

    # Densecap synonyms.
    scores = 1 - (emb * densecap_embs).sum(1)
    indices = scores.argsort()
    densecap_synonyms = []
    for i in indices:
      if vocab[i] in dense_words:
        densecap_synonyms.append(vocab[i])
        if len(densecap_synonyms) >= 5:
          break

    fp.write('%s\t%s\t%s\n' % (word, 
          ','.join(densecap_synonyms), ','.join(ads_synonyms)))

  fp.close()

if __name__ == '__main__':
  tf.app.run()
