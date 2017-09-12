import os
import sys
import json
import argparse
import random
import shutil

import urlparse
import numpy as np
import tensorflow as tf
import cv2
import scipy

import BaseHTTPServer
from sklearn.neighbors import KDTree
import nltk

from utils import vis
from utils import ads_dataset_api
from utils.ads_dataset_api import AdsDatasetApi
from utils.ads_dataset_api import is_validation_example

from train import load_vocab

flags = tf.app.flags
flags.DEFINE_string('visu_emb_path', '', 'Path to the visual embedding data.')
flags.DEFINE_string('word_emb_path', '', 'Path to the word embedding data.')
flags.DEFINE_string('image_dir', '', 'Directory to ads dataset.')
flags.DEFINE_string('entity_annot_path', '', 'File path to entity annotations.')
flags.DEFINE_string('qa_action_annot_path', '', 'File path to ads action annotations.')
flags.DEFINE_string('qa_reason_annot_path', '', 'File path to ads reason annotations.')
flags.DEFINE_string('qa_action_reason_annot_path', '', 'File path to ads action reason annotations.')

flags.DEFINE_string('http_host_name', '', 'Hostname of HTTP server.')
flags.DEFINE_integer('http_port_number', 8001, 'Port number of HTTP server.')

flags.DEFINE_integer('image_size', 300, 'Size of the image to be displayed.')
flags.DEFINE_integer('patch_size', 150, 'Size of the patch to be displayed.')
flags.DEFINE_float('score_threshold', 0.5, 'Score threshold for filtering bounding boxes.')
flags.DEFINE_integer('num_negative_statements', 20, 'Number of negative statements.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _tokenize(caption):
  return nltk.word_tokenize(caption.lower())


def _get_data(raw_data, data_type='train'):
  """Get embeddings and associated meta info.

  Args:
    raw_data: embedding data read from file.
    data_type: type of the split of the data, could be train|valid|test.

  Returns:
    embeddings: a [num_patches, embedding_size] np array.
    meta_list: a list containing meta info for each embedding vector.
  """
  embeddings = []
  meta_list = []
  for k, v in raw_data.iteritems():
    emb_list = None
    if data_type == 'train' and is_training_example(k):
      emb_list = v['entity_emb_list']
    if data_type == 'valid' and is_validation_example(k):
      emb_list = v['entity_emb_list']
    if data_type == 'test' and is_testing_example(k):
      emb_list = v['entity_emb_list']

    if emb_list is not None:
      embeddings.append(emb_list)
      for entity_id, emb in enumerate(emb_list):
        meta_list.append({
            'image_id': k,
            'entity_id': entity_id})
        
  tf.logging.info('Loaded %s images for %s.', len(embeddings), data_type)
  embeddings = np.concatenate(embeddings, axis=0)
  tf.logging.info('Loaded %s patches for %s.', len(embeddings), data_type)

  assert len(embeddings) == len(meta_list)
  return embeddings, meta_list


def get_similar_word_by_emb(query, vocab, vocab_r, word_emb, neighbors=20):
  """Get similar word by embedding vector.

  Args:
    query: a (embedding_size,) embedding.
    vocab: a list mapping from id to word.
    vocab_r: a dictionary mapping from word to (id, freq).
    word_emb: a [vocab_size, embedding_size] np array denoting embedding matrix.
    neighbors: number of nearest neighbors.

  Returns:
    word_list: a list of word info in which each contains:
      word_id: word integer id.
      word: word in text format.
      cosine_similarity: i.e., 1 - cosine(x, y).
  """
  query_emb = query
  distance = 1.0 - np.dot(word_emb, query_emb)

  word_list = []
  for index in np.argsort(distance)[:neighbors]:
    w_info = {
        'word_id': index,
        'word': vocab[index],
        'freq': vocab_r[vocab[index]][1],
        'cosine_similarity': float(distance[index])
        }
    word_list.append(w_info)
  return word_list


def get_similar_word_by_word(query, vocab, vocab_r, word_emb, neighbors=20):
  """Get similar word by word.

  Args:
    query: a query word.
    vocab: a list mapping from id to word.
    vocab_r: a dictionary mapping from word to (id, freq).
    word_emb: a [vocab_size, embedding_size] np array denoting embedding matrix.
    neighbors: number of nearest neighbors.

  Returns:
    word_list: a list of word info in which each contains:
      word_id: word integer id.
      word: word in text format.
      cosine_similarity: i.e., 1 - cosine(x, y).
  """
  query_id, freq = vocab_r.get(query, (None, None))
  if query_id is None:
    tf.logging.warning('Query word %s is not found.', query)
    return []

  query_emb = word_emb[query_id]
  return get_similar_word_by_emb(
      query_emb, vocab, vocab_r, word_emb, neighbors)


def get_similar_visual_patch_by_word(query, vocab_r, word_emb, visu_emb, meta_list, neighbors=20):
  """Get similar visual patch by word.

  Args:
    query: a query word.
    vocab_r: a dictionary mapping from word to (id, freq).
    word_emb: a [vocab_size, embedding_size] np array denoting embedding matrix.
    visu_emb: a [num_patches, embedding_size] np array denoting embeddings of image patches.
    meta_list: a list containing meta info associated to visu_emb.
    neighbors: number of nearest neighbors.

  Returns:
    patch_list: a list of patch info in which each contains:
      image_id: ads image_id, e.g. 3/11943.jpg.
      entity_id: id of the entity.
      cosine_similarity: i.e., 1 - cosine(x, y).
  """
  query_id, freq = vocab_r.get(query, (None, None))
  if query_id is None:
    tf.logging.warning('Query word %s is not found.', query)
    return []

  query_emb = word_emb[query_id]
  distance = 1.0 - np.dot(visu_emb, query_emb)

  patch_list = []
  for index in np.argsort(distance)[:neighbors]:
    patch_list.append({
        'image_id': meta_list[index]['image_id'],
        'entity_id': meta_list[index]['entity_id'],
        'cosine_similarity': float(distance[index])
        })
  return patch_list


def _unit_norm(embeddings):
  """Normalize rows of embedding matrix.

  Args:
    embeddings: a [vocab_size, embedding_size] np array.

  Returns:
    embeddings_unit: normalized [vocab_size, embedding_size] embedding array.
  """
  embedding_size = embeddings.shape[1]
  norm = 1e-12 + np.sqrt((embeddings ** 2).sum(axis=1))
  return embeddings / np.tile(np.expand_dims(norm, 1), [1, embedding_size])


class MyServer(BaseHTTPServer.HTTPServer):
  def init(self, data=None):
    self._data = data


class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):
  def do_GET(self):
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()

    if self.path[:len('/get_knn_info_by_word')] == '/get_knn_info_by_word':
      self.do_text_KNN(self.server._data,
          urlparse.parse_qs(urlparse.urlparse(self.path).query))

    elif self.path[:len('/get_knn_info_by_image_id')] == '/get_knn_info_by_image_id':
      self.do_visu_KNN(self.server._data,
          urlparse.parse_qs(urlparse.urlparse(self.path).query))

  def do_visu_KNN(self, server_data, args):
    # Parse arguments.
    neighbors = 20
    if 'neighbors' in args:
      neighbors = int(args['neighbors'][0])
    neighbors = min(neighbors, 100)

    if 'image_id' not in args:
      tf.logging.warning('Image_id is not specified in the request.')
      return
    image_id = args['image_id'][0]

    # Parameters stored in server instance.
    raw_data, vocab, vocab_r, word_emb, visu_emb, \
    meta_list, api, image_size, patch_size  = \
    server_data['raw_data'], server_data['vocab'], \
    server_data['vocab_r'], server_data['word_emb'], server_data['visu_emb'], \
    server_data['meta_list'], server_data['api'], server_data['image_size'], \
    server_data['patch_size']

    meta = api.get_meta_list_by_ids([image_id])[0]
    entity_emb_list = raw_data[image_id]['entity_emb_list']
    entity_annot_list = [e for e in meta['entities_ex'] if e['score'] >= FLAGS.score_threshold]
    if len(entity_emb_list) != len(entity_annot_list):
      raise ValueError('Embedding list does not matches to entity list.')

    html = ''
    html += '<html>'
    html += '<h1>Query: %s</h1>' % (image_id)

    # Statements.
    image_emb = entity_emb_list.mean(axis=0)

    html += '<h2>Statements:</h2>'
    positives = meta['action_reason_captions']
    negatives = meta['action_reason_captions_neg'][:FLAGS.num_negative_statements - len(positives)]
    statements = map(lambda x: (x, True), positives) + map(lambda x: (x, False), negatives)

    dists = []
    for statement, is_true in statements:
      embs = []
      for word in _tokenize(statement):
        w_id, freq = vocab_r.get(word, (0, 0))
        embs.append(word_emb[w_id])
      stmt_emb = np.stack(embs).mean(axis=0)
      dists.append(scipy.spatial.distance.cosine(stmt_emb, image_emb))

    statements = [(stmt, is_true, dist) for (stmt, is_true), dist in zip(statements, dists)]
    statements = sorted(statements, lambda x, y: cmp(x[2], y[2]))
    for stmt, is_true, dist in statements:
      if not is_true:
        html += '<p>[%.4lf] %s</p>' % (dist, stmt)
      else:
        html += '<p style="background-color:Yellow">[%.4lf] %s</p>' % (dist, stmt)
      
    # Region proposals.
    html += '<h2>Region proposals:</h2>'
    html += '<table border=1>'

    image_data = vis.image_load(meta['filename'])
    height, width, _ = image_data.shape

    image_resized = cv2.resize(image_data, (image_size, image_size))
    image_resized_with_annot = np.copy(image_resized)

    html += '<tr>'
    html += '<td><a href="get_knn_info_by_image_id?neighbors=%s&image_id=%s">%s</a></td>' % (
        neighbors, image_id, image_id)
    html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
        vis.image_uint8_to_base64(image_resized, disp_size=(image_size, image_size)))
    scale_fn = lambda x: int(x * image_size)

    patches = []
    scores = []
    synonyms = []
    for entity, entity_embedding  in zip(entity_annot_list, entity_emb_list):
      # Crop patch.
      score = entity['score']
      x1, y1 = entity['xmin'], entity['ymin']
      x2, y2 = entity['xmax'], entity['ymax']
      scores.append(score)
      patches.append(
          cv2.resize(
            image_data[int(y1 * height):int(y2 * height), int(x1 * width):int(x2 * width), :], 
            (patch_size, patch_size)
          ))

      # KNN search for synonyms.
      entity_embedding = entity_embedding / (1e-12 + np.sqrt((entity_embedding ** 2).sum()))
      synonym_list = get_similar_word_by_emb(
          entity_embedding, vocab, vocab_r, word_emb, neighbors=5)
      synonyms.append(synonym_list)

      # Draw bounding boxes on original image.
      color = (0, 255, 0)
      cv2.rectangle(image_resized_with_annot,
          (scale_fn(x1), scale_fn(y1)),
          (scale_fn(x2), scale_fn(y2)),
          color, thickness=2)
      cv2.putText(image_resized_with_annot,
          '%.2lf' % (score),
          (scale_fn(x1), scale_fn(y1) + 20),
          cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), thickness=2)
      cv2.putText(image_resized_with_annot,
          '%.2lf' % (score),
          (scale_fn(x1), scale_fn(y1) + 20),
          cv2.FONT_HERSHEY_COMPLEX, 0.6, color, thickness=1)

    html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
        vis.image_uint8_to_base64(image_resized_with_annot, disp_size=(image_size, image_size)))
    html += '</table>'

    # Patches and similar words.
    norm = 1e-12 + np.sqrt((image_emb ** 2).sum())
    image_emb = image_emb / norm

    html += '<h2>Image patches and similar words:</h2>'
    synonym_list = get_similar_word_by_emb(
        image_emb, vocab, vocab_r, word_emb, neighbors=5)
    for w_info in synonym_list:
      url = "get_knn_info_by_word?neighbors=%s&query=%s" % (neighbors, w_info['word'])
      html += '%.4lf: <a href="%s">%s</a></br>' % (w_info['cosine_similarity'], url, w_info['word'])

    html += '<table border=1>'
    html += '<tr><td>IMAGE_PATCH</td>'
    for patch in patches:
      html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
          vis.image_uint8_to_base64(patch, disp_size=(patch_size, patch_size)))
    html += '</tr>'

    html += '<tr><td>DETECTION SCORE</td>'
    for score in scores:
      html += '<td>%.2lf</td>' % (score)
    html += '</tr>'

    html += '<tr><td>SIMILAR WORDS</td>'
    for synonym_list in synonyms:
      html += '<td>'
      for w_info in synonym_list:
        url = "get_knn_info_by_word?neighbors=%s&query=%s" % (neighbors, w_info['word'])
        html += '%.4lf: <a href="%s">%s</a></br>' % (w_info['cosine_similarity'], url, w_info['word'])
      html += '</td>'
    html += '</tr>'

    html += '</table>'
    html += '</html>'

    self.wfile.write(html)

  def do_text_KNN(self, server_data, args):
    # Parse arguments.
    neighbors = 20
    if 'neighbors' in args:
      neighbors = int(args['neighbors'][0])
    neighbors = min(neighbors, 100)

    if 'query' not in args:
      tf.logging.warning('Query is not specified in the request.')
      return
    query = args['query'][0]

    # Parameters stored in server instance.
    raw_data, vocab, vocab_r, word_emb, visu_emb, \
    meta_list, api, image_size, patch_size  = \
    server_data['raw_data'], server_data['vocab'], \
    server_data['vocab_r'], server_data['word_emb'], server_data['visu_emb'], \
    server_data['meta_list'], server_data['api'], server_data['image_size'], \
    server_data['patch_size']

    word_list = get_similar_word_by_word(
        query, vocab, vocab_r, word_emb, neighbors)
    patch_list = get_similar_visual_patch_by_word(
        query, vocab_r, word_emb, visu_emb, meta_list, neighbors)
    html = ''
    html += '<h1>Query: %s</h1>' % (query)

    # Similar words.
    html += '<h2>Similar words:</h2>'
    html += '<table border=1>'
    html += '<tr>'
    html += '<td>INDEX</td>'
    html += '<td>WORD_ID</td>'
    html += '<td>WORD</td>'
    html += '<td>FREQUENCY</td>'
    html += '<td>DISTANCE</td>'
    html += '</tr>'
    for i, w_info in enumerate(word_list):
      html += '<tr>'
      html += '<td>%d</td>' % (i)
      html += '<td>%d</td>' % (w_info['word_id'])
      html += '<td><a href="get_knn_info_by_word?neighbors=%s&query=%s">%s</a></td>' % (
          neighbors, w_info['word'], w_info['word'])
      html += '<td>%d</td>' % (w_info['freq'])
      html += '<td>%.4lf</td>' % (w_info['cosine_similarity'])
      html += '</tr>'
    html += '</table>'

    # Similar image patches.
    html += '<h2>Similar image patches:</h2>'
    html += '<table border=1>'
    html += '<tr>'
    html += '<td>INDEX</td>'
    html += '<td>IMAGE_ID</td>'
    html += '<td>ENTITY_ID</td>'
    html += '<td>DISTANCE</td>'
    html += '<td>VISUALIZATION</td>'
    html += '</tr>'
    for i, p_info in enumerate(patch_list):
      # Load and draw bounding box on image.
      image_id = p_info['image_id']
      meta = api.get_meta_list_by_ids([image_id])[0]
      image_data = vis.image_load(meta['filename'])
      image_resized = cv2.resize(image_data, (image_size, image_size))

      entity = meta['entities_ex'][p_info['entity_id']]
      score = entity['score']
      x1, y1 = entity['xmin'], entity['ymin']
      x2, y2 = entity['xmax'], entity['ymax']

      color = (0, 255, 0)
      scale_fn = lambda x: int(x * image_size)
      cv2.rectangle(image_resized,
          (scale_fn(x1), scale_fn(y1)),
          (scale_fn(x2), scale_fn(y2)),
          color, thickness=2)

      html += '<tr>'
      html += '<td>%d</td>' % (i)
      html += '<td><a href="get_knn_info_by_image_id?neighbors=%s&image_id=%s">%s</a></td>' % (
          neighbors, image_id, image_id)
      html += '<td>%s</td>' % (p_info['entity_id'])
      html += '<td>%.4lf</td>' % (p_info['cosine_similarity'])
      html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
          vis.image_uint8_to_base64(image_resized))
      html += '</tr>'

    self.wfile.write(html)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  vocab, vocab_r = load_vocab(FLAGS.vocab_path)
  tf.logging.info('Load %s words in the vocabulary.', len(vocab))

  # Init ads api, load entity annotations.
  api = AdsDatasetApi()
  api.init(
      images_dir=FLAGS.image_dir, 
      entity_annot_file_ex=FLAGS.entity_annot_path,
      qa_action_annot_file=FLAGS.qa_action_annot_path,
      qa_reason_annot_file=FLAGS.qa_reason_annot_path,
      qa_action_reason_annot_file=FLAGS.qa_action_reason_annot_path)
  api.sample_negative_action_reason_captions(FLAGS.num_negative_statements)

  # Read word embeddings.
  word_emb = _unit_norm(np.load(FLAGS.word_emb_path))
  
  # Read visual embeddings.
  raw_data = np.load(FLAGS.visu_emb_path).item()
  visu_emb, meta_list = _get_data(raw_data, data_type='valid')
  visu_emb = _unit_norm(visu_emb)

  tf.logging.info('Indices inititalized, now start the HTTP server.')

  httpd = MyServer((FLAGS.http_host_name, FLAGS.http_port_number), MyHandler)
  httpd.init(data={
      'raw_data': raw_data,
      'vocab': vocab,
      'vocab_r': vocab_r,
      'word_emb': word_emb,
      'visu_emb': visu_emb,
      'meta_list': meta_list,
      'api': api,
      'image_size': FLAGS.image_size,
      'patch_size': FLAGS.patch_size
      })
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass

  httpd.server_close()
  tf.logging.info('HTTP server exits!')

if __name__ == '__main__':
  tf.app.run()
