
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import cv2
import nltk
from nltk.stem import PorterStemmer

import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util

from utils import image_coder
from utils.ads_api import AdsApi

flags = tf.app.flags
flags.DEFINE_string('ads_config', '', 'Directory to ads dataset config file.')
flags.DEFINE_string('feature_path', '', 'Path to the feature data file.')

flags.DEFINE_string('w2v_vocab_path', '', 'File path to the word2vec vocab file.')
flags.DEFINE_string('vocab_path', '', 'File path to the vocab file.')
flags.DEFINE_string('densecap_vocab_path', '', 'File path to the densecap vocab file.')
flags.DEFINE_string('output_path', '', 'Path of the output TFRecord file.')

flags.DEFINE_integer('max_entities_per_image', 10, 'Maximum number of entities per image.')
flags.DEFINE_integer('max_annots_per_image', 5, 'Maximum number of QA annotations per image.')
flags.DEFINE_integer('max_caption_len', 30, 'Maximum length of QA caption.')
flags.DEFINE_integer('max_densecap_annots_per_image', 10, 'Maximum number of QA annotations per image.')
flags.DEFINE_integer('max_densecap_caption_len', 10, 'Maximum length of QA caption.')
flags.DEFINE_integer('max_symbols_per_image', 5, 'Maximum number of symbols per image.')

flags.DEFINE_integer('image_size', 500, 'Maximum height/widht of the image.')

FLAGS = flags.FLAGS
coder = image_coder.ImageCoder()


def _get_captions(meta_list, split='train'):
  captions = []
  for meta in meta_list:
    if meta['split'] == split and 'statements' in meta:
      captions.extend(meta['statements'])
  return captions

def _get_densecap_captions(meta_list, split='train'):
  captions = []
  for meta in meta_list:
    if meta['split'] == split and 'densecap_objects' in meta:
      # Export only top-5 densecap captions.
      captions.extend([x['caption'] for x in meta['densecap_objects'][:5]])
  return captions


def _tokenize(caption):
  caption = caption.replace('<UNK>', '')  # For the densecap case.
  caption = nltk.word_tokenize(caption.lower())
  return caption

def _load_vocab(file_path):
  with open(file_path, 'r') as fp:
    lines = fp.readlines()
  words = [w.strip('\n') for w in lines]
  return dict([(w, i + 1) for (i, w) in enumerate(words)])


def _create_vocab(captions, min_count=2):
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


def _check_coverage(vocab, captions):
  uncover = 0
  for caption in captions:
    for word in _tokenize(caption):
      if not word in vocab:
        uncover += 1
        break
  tf.logging.info('coverage: %.4lf', 1.0 - 1.0 * uncover / len(captions))


def _save_and_index_vocab(vocab, vocab_path):
  vocab = sorted(vocab.iteritems(), lambda x, y: -cmp(x[1], y[1]))
  index = {}
  with open(vocab_path, 'w') as fp:
    for idx, (k, v) in enumerate(vocab):
      fp.write('%s\t%d\t%d\n' % (k, idx + 1, v))
      index[k] = idx + 1
  return index


def dict_to_tf_example(data):
  caption_strings = reduce(lambda x, y: x + y, data['caption_strings'])
  assert len(caption_strings) == FLAGS.max_annots_per_image * FLAGS.max_caption_len

  densecap_caption_strings = reduce(lambda x, y: x + y, data['densecap_caption_strings'])
  assert len(densecap_caption_strings) == FLAGS.max_densecap_annots_per_image * FLAGS.max_densecap_caption_len

  # Encode image.
  full_path = data['file_path']
  encoded_data = tf.gfile.FastGFile(full_path, 'r').read()
  try:
    if full_path[-4:].lower() == '.png':
      image = coder.decode_png(encoded_data)
    else:
      image = coder.decode_jpeg(encoded_data)
  except Exception as ex:
    logging.warning('failed to decode %s.', full_path)
    return None

  height, width, channels = image.shape
  image = cv2.resize(image, (FLAGS.image_size, FLAGS.image_size))
  encoded_data = coder.encode_jpeg(image)
  key = hashlib.sha256(encoded_data).hexdigest()

  # Precomputed embeddings.
  image_emb = data['image_emb']
  entity_embs = data['entity_embs'].reshape((-1))
  assert entity_embs.shape[0] == FLAGS.max_entities_per_image * data['entity_embs'].shape[1]

  entity_scores = data['entity_scores'].reshape((-1))
  assert entity_scores.shape[0] == FLAGS.max_entities_per_image

  example = tf.train.Example(features=tf.train.Features(feature={
        'image/source_id': dataset_util.bytes_feature(data['image_id'].encode('utf8')),
#        'image/sha256': dataset_util.bytes_feature(key.encode('utf8')),
#        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/embeddings': dataset_util.float_list_feature(image_emb.tolist()),
        'topic/topic_id': dataset_util.int64_feature(data['topic_id']),
        'entity/num_entities': dataset_util.int64_feature(data['num_entities']),
        'entity/embeddings': dataset_util.float_list_feature(entity_embs.tolist()),
        'entity/scores': dataset_util.float_list_feature(entity_scores.tolist()),
        'caption/num_captions': dataset_util.int64_feature(data['num_captions']),
        'caption/caption_lengths': dataset_util.int64_list_feature(data['caption_lengths']),
        'caption/caption_strings': dataset_util.int64_list_feature(caption_strings),
        'densecap_caption/num_captions': dataset_util.int64_feature(data['num_densecap_captions']),
        'densecap_caption/caption_lengths': dataset_util.int64_list_feature(data['densecap_caption_lengths']),
        'densecap_caption/caption_strings': dataset_util.int64_list_feature(densecap_caption_strings),
        'symbols/num_symbols': dataset_util.int64_feature(data['num_symbols']),
        'symbols/symbol_ids': dataset_util.int64_list_feature(data['symbol_ids']),
        }))
  return example


def _pad_captions(caption_strings, vocab, max_annots_per_image, max_caption_len):
  """Pad multiple captions for a single image patch.
    
  Returns:
    num_captions: number of captions, should be less than max_annots_per_image.
    lengths: length for each of the captions, shape=[max_annots_per_image].
    captions: ids for each of the captions, shape=[max_annots_per_image, max_caption_len].
  """
  unk_id = 0

  caption_strings = caption_strings[:max_annots_per_image]

  num_captions = len(caption_strings)
  lengths = []
  captions = []

  for caption in caption_strings:
    caption = [vocab.get(w, unk_id) for w in _tokenize(caption)]
    caption = caption[:max_caption_len]

    if len(caption) > 0:
      lengths.append(len(caption))
      while len(caption) != max_caption_len:
        caption.append(0)
      captions.append(caption)

  while len(captions) != max_annots_per_image:
    captions.append([0] * max_caption_len)
  while len(lengths) != max_annots_per_image:
    lengths.append(0)

  assert num_captions <= max_annots_per_image
  assert len(lengths) == max_annots_per_image
  assert len(captions) == max_annots_per_image
  for caption in captions:
    assert len(caption) == max_caption_len

  return num_captions, lengths, captions

def _pad_entities(entity_emb_list):
  entity_emb_list = entity_emb_list[:FLAGS.max_entities_per_image]
  num_entities = entity_emb_list.shape[0]

  if num_entities < FLAGS.max_entities_per_image:
    entity_emb_list = np.concatenate([entity_emb_list, 
        np.zeros((FLAGS.max_entities_per_image - num_entities, entity_emb_list.shape[1])) ], 
        axis=0)

  assert num_entities <= FLAGS.max_entities_per_image
  #assert entity_emb_list.shape == (FLAGS.max_entities_per_image, 1024)
  return num_entities, entity_emb_list

def create_tf_record(output_path, raw_features, meta_list, vocab, densecap_vocab):
  count = 0
  writer = tf.python_io.TFRecordWriter(output_path)

  for meta_index, meta in enumerate(meta_list):
    if meta_index % 100 == 0:
      tf.logging.info('On image %d of %d.' % (meta_index, len(meta_list)))

    image_id = meta['image_id']
    raw_feature = raw_features[image_id]

    image_emb = raw_feature['image_emb']
    entity_emb_list = raw_feature['object_emb_list']
    entity_score_list = raw_feature['object_score_list']
    assert len(entity_emb_list) == 10
    assert len(entity_score_list) == 10

    # Get entities annotations.
    entity_emb_list = np.concatenate(
        [np.expand_dims(image_emb, 0), entity_emb_list], axis=0)
    entity_score_list = np.concatenate(
        [[0], entity_score_list], axis=0)

    num_entities, entity_emb_list = _pad_entities(entity_emb_list)
    entity_score_list = entity_score_list[:FLAGS.max_entities_per_image].tolist()
    while len(entity_score_list) < FLAGS.max_entities_per_image:
      entity_score_list.append(0)
    entity_score_list = np.array(entity_score_list)

    # Get statements annotations.
    captions = meta.get('statements', [])
    num_captions, lengths, captions = _pad_captions(
        captions, vocab,
        max_annots_per_image=FLAGS.max_annots_per_image,
        max_caption_len=FLAGS.max_caption_len)

    # Get densecap annotations.
    densecap_captions = [x['caption'] for x in meta['densecap_objects'][:FLAGS.max_densecap_annots_per_image] if x['score'] > 0]
    num_densecap_captions, densecap_lengths, densecap_captions = _pad_captions(
        densecap_captions, densecap_vocab,
        max_annots_per_image=FLAGS.max_densecap_annots_per_image,
        max_caption_len=FLAGS.max_densecap_caption_len)

    # Get symbol annotations.
    symbols = meta.get('symbol_ids', [])
    while len(symbols) < FLAGS.max_symbols_per_image:
      symbols.append(0)
    symbols = symbols[:FLAGS.max_symbols_per_image]
    num_symbols = len(symbols)

    example = {
      'image_id': image_id,
      'image_emb': image_emb,
      'file_path': meta['file_path'],
      'topic_id': meta.get('topic_id', 0),
      'num_captions': num_captions,
      'caption_lengths': lengths,
      'caption_strings': captions,
      'num_densecap_captions': num_densecap_captions,
      'densecap_caption_lengths': densecap_lengths,
      'densecap_caption_strings': densecap_captions,
      'num_symbols': num_symbols,
      'symbol_ids': symbols,
      'num_entities': num_entities,
      'entity_embs': entity_emb_list,
      'entity_scores': entity_score_list,
    }

    if num_entities > 0 and num_captions > 0 and num_densecap_captions > 0:
      tf_example = dict_to_tf_example(example)
      writer.write(tf_example.SerializeToString())

      count += 1
  writer.close()
  tf.logging.info("Write %s records.", count)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Read data file.
  api = AdsApi(FLAGS.ads_config)

  # Create vocabulary.
  meta_list = api.get_meta_list()
  train_captions = _get_captions(meta_list, split='train')
  valid_captions = _get_captions(meta_list, split='valid')

  if FLAGS.w2v_vocab_path:
    vocab = _load_vocab(FLAGS.w2v_vocab_path)
  else:
    vocab = _create_vocab(train_captions, min_count=2)
    vocab = _save_and_index_vocab(vocab, FLAGS.vocab_path)
  _check_coverage(vocab, train_captions)
  _check_coverage(vocab, valid_captions)


  # Create densecap vocabulary.
  train_densecap_captions = _get_densecap_captions(meta_list, split='train')
  valid_densecap_captions = _get_densecap_captions(meta_list, split='valid')

  if FLAGS.w2v_vocab_path:
    densecap_vocab = _load_vocab(FLAGS.w2v_vocab_path)
  else:
    densecap_vocab = _create_vocab(train_densecap_captions)
    densecap_vocab = _save_and_index_vocab(
        densecap_vocab, FLAGS.densecap_vocab_path)
  _check_coverage(densecap_vocab, train_densecap_captions)
  _check_coverage(densecap_vocab, valid_densecap_captions)

  # Create tf record
  meta_list = [meta for meta in meta_list if meta['split'] == 'train' and 'statements' in meta]
  raw_features = np.load(FLAGS.feature_path).item()
  create_tf_record(FLAGS.output_path, 
      raw_features, meta_list, vocab, densecap_vocab)

if __name__ == '__main__':
  tf.app.run()
