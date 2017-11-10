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
flags.DEFINE_string('api_config', 'configs/ads_api.config', 'Path to config file.')
flags.DEFINE_integer('num_clusters', 5, 'Number of clusters.')
flags.DEFINE_string('image_emb_path', 'output/image_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('word_emb_path', 'output/word_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('word_txt_path', 'output/glove_w2v.txt', 'Path to the feature data file.')
flags.DEFINE_string('densecap_emb_path', 'output/densecap_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('densecap_txt_path', 'output/glove_w2v.txt', 'Path to the feature data file.')
flags.DEFINE_string('topic', 'sports', 'A topic name.')
flags.DEFINE_string('vis_file', 'cluster.html', 'Path to the output html file.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


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
  return vocab


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  with open('raw_data/ads/annotations/clustered_symbol_list.json', 'r') as fp:
    data = json.loads(fp.read())
  symbols = set()
  for elem in data['data']:
    for symbol in elem['symbols']:
      symbols.add(symbol)

  meta_list = _get_meta_list()
  tf.logging.info('Load %s records.', len(meta_list))

  # Read data file.
  feature_data = _get_feature(FLAGS.image_emb_path)
  word_embs = np.load(FLAGS.word_emb_path)
  densecap_embs = np.load(FLAGS.densecap_emb_path)

  image_metas, image_embs = _get_data(meta_list, feature_data, 
      split='valid', topic=FLAGS.topic)
  tf.logging.info('Load %s records for %s.', len(image_metas), FLAGS.topic)

  vocab = _load_vocab(FLAGS.word_txt_path)

  # Cluster the data.
  Z = linkage(image_embs, method='average', metric='cosine')
  print(Z[0])
  print(Z[-1])
  #labels = fcluster(Z, FLAGS.num_clusters, criterion='maxclust')
  labels = fcluster(Z, 0.8, criterion='distance')

  print(labels)

  # Visualize data.
  vis_dict = {}
  for meta, label in zip(image_metas, labels):
    if label >= 0: vis_dict.setdefault(label, []).append(meta)
  vis_dict = sorted(vis_dict.iteritems(), lambda x, y: -cmp(len(x[1]), len(y[1])))

  tf.logging.info('Got %d clusters.', len(vis_dict))

  html = ''
  html += '<table border=1>'
  html += '<tr>'
  html += '<th>cluster id</th>'
  html += '<th>ads words</th>'
  html += '<th>densecap words</th>'
  html += '</tr>'
  label_id = 0
  for label, meta_list in vis_dict:
    #if len(meta_list) <= 2: 
    #  continue

    label_id += 1

    # Retrieve the related words.
    image_emb = [np.expand_dims(feature_data[meta['image_id']], 0) for meta in meta_list]
    image_emb = np.concatenate(image_emb, axis=0).sum(0)
    image_emb = image_emb / np.linalg.norm(image_emb)
    word_scores = 1.0 - (image_emb*word_embs).sum(1)
    word_indices = word_scores.argsort()[:10]
    densecap_scores = 1.0 - (image_emb*densecap_embs).sum(1)
    densecap_indices = densecap_scores.argsort()[:10]

    html += '<tr id="%s">' % (label_id)
    html += '<td><a href="#%d">%d</a></td>' % (label_id, label_id)
    html += '<td>'
    for i in word_indices:
      if vocab[i] in symbols:
        html += '%.2lf <b>%s</b></br>' % (word_scores[i], vocab[i])
      else:
        html += '%.2lf %s</br>' % (word_scores[i], vocab[i])
    html += '</td>'
    html += '<td>'
    for i in densecap_indices:
      if vocab[i] in symbols:
        html += '%.2lf <b>%s</b></br>' % (densecap_scores[i], vocab[i])
      else:
        html += '%.2lf %s</br>' % (densecap_scores[i], vocab[i])
    html += '</td>'
    meta_list = meta_list[:30]
    for meta in meta_list:
      image = vis.image_load(meta['file_path'])
      image = cv2.resize(image, (300, 300))
      statements = meta.get('statements', [])
      statements = ['<p>' + x + '</p>' for x in statements]
      html += '<td><img src="data:image/jpg;base64,%s"></br>%s</td>' % (
          vis.image_uint8_to_base64(image), ''.join(statements))
    html += '</tr>'
  html += '</table>'
  with open(FLAGS.vis_file, 'w') as fp:
    fp.write(html)

if __name__ == '__main__':
  tf.app.run()
