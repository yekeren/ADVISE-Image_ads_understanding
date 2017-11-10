
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import json

import cv2

from google.protobuf import text_format
from protos import ads_emb_model_pb2

import numpy as np
import tensorflow as tf

from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans

from source import ads_emb_model

from utils import ads_dataset_api
from utils import vis
from utils.ads_dataset_api import AdsDatasetApi
from utils.ads_dataset_api import is_training_example
from utils.ads_dataset_api import is_validation_example

flags = tf.app.flags

flags.DEFINE_string('model_config', 
    'models/mobilenet_v1.from_feat.patch_repr.densecap_x0.5.image_relu6.caption_none-1/model.pbtxt', 
    'Path to the configuration file.')

flags.DEFINE_string('saved_ckpts_dir',
    'log5/mobilenet_v1.from_feat.patch_repr.densecap_x0.5.image_relu6.caption_none-1.adagrad_2.0/saved_ckpts/', 
    'The directory used to save the current best model.')

flags.DEFINE_string('visual_feature_path', 
    'output/mobilenet_v1_features.npz', 
    'Path to the feature data file.')

flags.DEFINE_string('caption_vocab_path', 
    'output/vocab.txt', 
    'Path to vocab file.')

flags.DEFINE_string('densecap_vocab_path', 
    'output/densecap_vocab.txt', 
    'Path to densecap vocab file.')

flags.DEFINE_integer('kmeans_num_clusters', 
    100, 'Number of clusters of Kmeans.')

flags.DEFINE_string('kmeans_model_path', 
    'output/kmeans.npz', 
    'Path to the kmeans model file.')

flags.DEFINE_string('images_dir', 
    'raw_data/ads/images', 
    'Path to ads images.')

flags.DEFINE_string('entity_annot_path', 
    'raw_data/ads/annotations/ssd_proposals.json', 
    'Path to entity annotations.')

flags.DEFINE_string('visual_word_path', 
    'output/visual_words.json', 
    'Path to visual words output.')

flags.DEFINE_string('image_html', 
    'output/image.html', 
    'Path to the image html file.')

flags.DEFINE_string('visual_word_html', 
    'output/visual_words.html', 
    'Path to the visual word html file.')

flags.DEFINE_string('visual_word_text', 
    'output/visual_words.text', 
    'Path to the visual word text file.')

FLAGS = flags.FLAGS


def default_session_config_proto():
  """Get the default config proto for tensorflow session.

  Returns:
    config: The default config proto for tf.Session.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 1.0
  return config


def load_vocab(vocab_path):
  """Load vocabulary from file.

  Args:
    vocab_path: path to the vocab file.

  Returns:
    vocab: a list mapping from id to text.
    vocab_r: a dictionary mapping from text to id.
  """
  with open(vocab_path, 'r') as fp:
    lines = fp.readlines()

  vocab_r = {'<UNK>': (0, 0)}
  vocab = ['<UNK>'] * (len(lines) + 1)

  for line in lines:
    word, index, freq = line.strip('\n').split('\t')
    vocab[int(index)] = word
    vocab_r[word] = (int(index), int(freq))
  return vocab, vocab_r


caption_vocab, caption_vocab_r = load_vocab(FLAGS.caption_vocab_path)
densecap_vocab, densecap_vocab_r = load_vocab(FLAGS.densecap_vocab_path)


def _tokenize(caption):
  return nltk.word_tokenize(caption.lower())

def _get_visual_features(filter_fn):
  raw_data = np.load(FLAGS.visual_feature_path).item()
  image_ids = []
  examples = []
  for k, v in raw_data.iteritems():
    if filter_fn(k):
      examples.append(v['entity_emb_list'])
      image_ids.append(k)
  return image_ids, examples


def _get_model_path():
  file_path_list = []
  for file_path in tf.gfile.Glob(os.path.join(FLAGS.saved_ckpts_dir, 'model.ckpt-*.meta')):
    file_path_list.append(file_path)

  if len(file_path_list) == 0:
    raise ValueError('No checkpoint was found in %s.' % (FLAGS.saved_ckpts_dir))

  ckpt_fn = lambda x: int(re.findall('ckpt-(\d+).meta', x)[0])
  model_path = sorted(file_path_list, lambda x, y: -cmp(ckpt_fn(x), ckpt_fn(y)))[0]
  model_path = model_path[:-5]

  return model_path


def _embed_visual_features(examples):
  g = tf.Graph()
  with g.as_default():
    model_proto = ads_emb_model_pb2.AdsEmbModel()
    with open(FLAGS.model_config, 'r') as fp:
      text_format.Merge(fp.read(), model_proto)
    model = ads_emb_model.AdsEmbModel(model_proto)

    feature_placeholder = tf.placeholder(
        dtype=tf.float32, shape=[None, examples[0].shape[1]])
    embedding_tensor = model.embed_feature(
        feature_placeholder,
        embedding_size=model_proto.embedding_size,
        weight_decay=0.0,
        is_training=False)
    embedding_tensor = tf.nn.l2_normalize(embedding_tensor, 1)

    invalid_tensor_names = tf.report_uninitialized_variables()
    saver = tf.train.Saver()

  with tf.Session(graph=g, config=default_session_config_proto()) as sess:
    model_path = _get_model_path()
    saver.restore(sess, model_path)

    invalid_tensor_names = sess.run(invalid_tensor_names)
    assert len(invalid_tensor_names) == 0

    embs_list = []
    for i, example in enumerate(examples):
      embs = sess.run(embedding_tensor, feed_dict={feature_placeholder: example})
      embs_list.append(embs)

      if i % 100 == 0:
        tf.logging.info('On image %d of %d.', i, len(examples))
  return embs_list


def _get_kmeans_model():
  # Process kmeans clustering and generate model file.
  if not os.path.isfile(FLAGS.kmeans_model_path):
    tf.logging.info('Training kmeans...')
    _, examples = _get_visual_features(is_training_example)
    examples_embeded = _embed_visual_features(examples)

    kmeans = MiniBatchKMeans(n_clusters=FLAGS.kmeans_num_clusters, 
        max_iter=50, 
        batch_size=10000,
        max_no_improvement=None, 
        verbose=1)

    kmeans.fit(np.concatenate(examples_embeded, axis=0))
    joblib.dump(kmeans, FLAGS.kmeans_model_path)

  # Load kmeans model.
  else:
    kmeans = joblib.load(FLAGS.kmeans_model_path)
    tf.logging.info('Load kmeans model from file %s.', FLAGS.kmeans_model_path)

  return kmeans


def _get_visual_words(kmeans, filter_fn):
  if not os.path.isfile(FLAGS.visual_word_path):
    tf.logging.info('Generating visual words...')
    api = ads_dataset_api.AdsDatasetApi()
    api.init(images_dir=FLAGS.images_dir,
        entity_annot_file_ex=FLAGS.entity_annot_path)

    image_ids, examples = _get_visual_features(filter_fn)
    examples_embeded = _embed_visual_features(examples)

    assert len(image_ids) == len(examples_embeded)

    visual_words = {}
    for image_id, example_embed in zip(image_ids, examples_embeded):
      visual_words[image_id] = kmeans.predict(example_embed).tolist()

    with open(FLAGS.visual_word_path, 'w') as fp:
      fp.write(json.dumps(visual_words))
  else:
    with open(FLAGS.visual_word_path, 'r') as fp:
      visual_words = json.loads(fp.read())
    tf.logging.info('Load visual words from %s.', FLAGS.visual_word_path)
  return visual_words

def _index_examples(visual_words):
  api = AdsDatasetApi()
  api.init(images_dir=FLAGS.images_dir,
      entity_annot_file_ex=FLAGS.entity_annot_path)

  examples = []

  for image_id, words in visual_words.iteritems():
    meta = api.get_meta_list_by_ids([image_id])[0]

    nms_words = set()
    for entity_id, word in enumerate(words):
      box = meta['entities_ex'][entity_id]
      if box['score'] < 0.2:
        break
      if word not in nms_words:
        nms_words.add(word)
        examples.append({
            'image_id': image_id,
            'visual_word': word,
            'filename': meta['filename'],
            'box': box
            })
  return examples

class WordSim(object):
  def __init__(self):
    self.g = tf.Graph()
    with self.g.as_default():
      model_proto = ads_emb_model_pb2.AdsEmbModel()
      with open(FLAGS.model_config, 'r') as fp:
        text_format.Merge(fp.read(), model_proto)
      model = ads_emb_model.AdsEmbModel(model_proto)

      model.caption_encoder.build_weights(
          vocab_size=len(caption_vocab),
          embedding_size=200)
      model.densecap_encoder.build_weights(
          vocab_size=len(densecap_vocab),
          embedding_size=200)

      caption_weights = tf.nn.l2_normalize(
          model.caption_encoder.embedding_weights, 1)
      densecap_weights = tf.nn.l2_normalize(
          model.densecap_encoder.embedding_weights, 1)

      feature_placeholder = tf.placeholder(
          dtype=tf.float32, shape=[200])
      #embedding_tensor = model.embed_feature(
      #    tf.expand_dims(feature_placeholder, 0),
      #    embedding_size=model_proto.embedding_size,
      #    weight_decay=0.0,
      #    is_training=False)
      #embedding_tensor = tf.nn.l2_normalize(embedding_tensor, 1)[0]
      embedding_tensor = feature_placeholder

      caption_scores = 1 - tf.reduce_sum(embedding_tensor * caption_weights, 1)
      densecap_scores = 1 - tf.reduce_sum(embedding_tensor * densecap_weights, 1)
      
      caption_scores, caption_indices = tf.nn.top_k(-caption_scores, k=400)
      densecap_scores, densecap_indices = tf.nn.top_k(-densecap_scores, k=400)
      self.result_tensors = {
        'caption_scores': -caption_scores,
        'densecap_scores': -densecap_scores,
        'caption_indices': caption_indices,
        'densecap_indices': densecap_indices
      }
      self.feature_placeholder = feature_placeholder

      invalid_tensor_names = tf.report_uninitialized_variables()
      saver = tf.train.Saver()

    self.sess = tf.Session(graph=self.g, config=default_session_config_proto())
    model_path = _get_model_path()
    saver.restore(self.sess, model_path)


    invalid_tensor_names = self.sess.run(invalid_tensor_names)
    assert len(invalid_tensor_names) == 0

  def get_sim_words(self, feature):
    return self.sess.run(self.result_tensors, 
        feed_dict={self.feature_placeholder: feature})


def _show_visual_words(kmeans, examples):
  sim = WordSim()

  vis_dict = {}
  for example in examples:
    patches = vis_dict.setdefault(example['visual_word'], [])
    if len(patches) < 20:
      patches.append(example)

  centers = kmeans.cluster_centers_

  html = ''
  html += '<table border=1>'
  html += '<tr>'
  html += '<th>visual word</th>'
  html += '<th width="200px">qa word ([distance freq] word)</th>'
  html += '<th width="200px">densecap word ([distance] word)</th>'
  html += '<th>image patches</th>'
  html += '</tr>'

  text = ''

  count = 0
  for word, patches in vis_dict.iteritems():
    html += '<tr id="%s">' % (word)
    html += '<td><a href="#%s">%s</a></td>' % (word, word)
    text += '%s' % (word)

    result = sim.get_sim_words(centers[word])

    caption_scores = result['caption_scores'][:20]
    caption_indices = result['caption_indices'][:20]
    densecap_scores = result['densecap_scores'][:20]
    densecap_indices = result['densecap_indices'][:20]

    html += '<td><table>'
    word_list = []
    for word_id, score in zip(caption_indices, caption_scores):
      freq = caption_vocab_r[caption_vocab[word_id]][1]
      if freq >= 1:
        html += '<tr><td>%.4lf</td><td>%d</td><td>%s</td></tr>' % (score, freq, caption_vocab[word_id])
        word_list.append(caption_vocab[word_id])
    html += '</table></td>'
    text += '\t%s' % (','.join(word_list))

    html += '<td><table>'
    word_list = []
    for word_id, score in zip(densecap_indices, densecap_scores):
      html += '<tr><td>%.4lf</td><td>%s</td></tr>' % (score, densecap_vocab[word_id])
      word_list.append(densecap_vocab[word_id])
    html += '</table></td>'
    text += '\t%s\n' % (','.join(word_list))

    for patch in patches:
      image = vis.image_load(patch['filename'])
      score = patch['box']['score']
      box = [patch['box']['xmin'], patch['box']['ymin'], patch['box']['xmax'], patch['box']['ymax']]

      patch = vis.image_crop_and_resize(image, box, crop_size=(160, 160))
      html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
          vis.image_uint8_to_base64(patch))

    html += '</tr>'
    tf.logging.info('On word %d of %d.', word, len(vis_dict))

  html += '</table>'

  with open(FLAGS.visual_word_html, 'w') as fp:
    fp.write(html)
  with open(FLAGS.visual_word_text, 'w') as fp:
    fp.write(text)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  kmeans = _get_kmeans_model()
  visual_words = _get_visual_words(kmeans, is_validation_example)

  examples = _index_examples(visual_words)
  tf.logging.info('Indexed %d examples.', len(examples))

  # Show visual words.
  _show_visual_words(kmeans, examples)

if __name__ == '__main__':
  tf.app.run()
