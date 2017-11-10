
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import nltk
from utils.ads_dataset_api import AdsDatasetApi
from utils.embedding_converter import *

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Directory to ads dataset.')
flags.DEFINE_string('densecap_annot_file', '', 'File path to densecap annotations.')
flags.DEFINE_string('qa_action_annot_file', '', 'File path to ads action annotations.')
flags.DEFINE_string('qa_reason_annot_file', '', 'File path to ads reason annotations.')
flags.DEFINE_string('qa_action_reason_annot_file', '', 'File path to ads action reason annotations.')

FLAGS = flags.FLAGS


def _get_captions(api):
  captions = []
  for meta in api.get_meta_list():
    if 'action_reason_captions' in meta:
      captions.extend(meta['action_reason_captions'])
  return captions


def _get_densecap_captions(api):
  captions = []
  for meta in api.get_meta_list():
    if 'densecap_entities' in meta:
      captions.extend([x['caption'] for x in meta['densecap_entities'][:5]])
  return captions


def _tokenize(caption):
  caption = caption.replace('<UNK>', '')
  caption = nltk.word_tokenize(caption.lower())
  #caption = [ps.stem(w) for w in caption]
  return caption


def _create_vocab(captions, min_count=1):
  vocab = {}
  for caption in captions:
    for word in _tokenize(caption):
      vocab[word] = vocab.get(word, 0) + 1

  tf.logging.info('number of words: %d.', len(vocab))
  for k in vocab.keys():
    if vocab[k] < min_count:
      del vocab[k]
  tf.logging.info('after filtering low frequency words: %d.', len(vocab))
  return vocab


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  api = AdsDatasetApi()
  api.init(
      images_dir=FLAGS.data_dir,
      densecap_annot_file=FLAGS.densecap_annot_file,
      qa_action_annot_file=FLAGS.qa_action_annot_file,
      qa_reason_annot_file=FLAGS.qa_reason_annot_file,
      qa_action_reason_annot_file=FLAGS.qa_action_reason_annot_file)

  captions = []
  captions.extend(_get_captions(api))
  captions.extend(_get_densecap_captions(api))

  vocab = _create_vocab(captions)

  converter = GloveConverter()
  converter.load('models/zoo/glove.6B.100d.txt', words=vocab)
  converter.saveVocab('output/glove_w2v_100d.txt')
  converter.saveEmbedding('output/glove_w2v_100d.npz')


if __name__ == '__main__':
  tf.app.run()
