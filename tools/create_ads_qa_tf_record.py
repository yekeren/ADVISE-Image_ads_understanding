
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import logging
import cv2
import nltk

import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util

from utils import image_coder
from utils.ads_dataset_api import AdsDatasetApi
from utils.ads_dataset_api import is_training_example
from utils.ads_dataset_api import is_validation_example
from utils.ads_dataset_api import is_testing_example

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Directory to ads dataset.')
flags.DEFINE_string('emb_path', '', 'Path to the feature data file.')
flags.DEFINE_string('qa_action_annot_file', '', 'File path to ads action annotations.')
flags.DEFINE_string('qa_reason_annot_file', '', 'File path to ads reason annotations.')
flags.DEFINE_string('qa_action_reason_annot_file', '', 'File path to ads action reason annotations.')

flags.DEFINE_string('vocab_path', '', 'File path to the vocab file.')
flags.DEFINE_string('output_dir', '', 'Directory to output TFRecord')

flags.DEFINE_integer('max_annots_per_image', 5, 'Maximum number of QA annotations per image.')
flags.DEFINE_integer('max_caption_len', 30, 'Maximum length of QA caption.')
flags.DEFINE_integer('max_entities_per_image', 10, 'Maximum number of entities per image.')

FLAGS = flags.FLAGS
coder = image_coder.ImageCoder()


def _get_data(raw_embs, export_type='training'):
  examples = []
  for k, v in raw_embs.iteritems():
    example = {'image_id': k}
    if export_type == 'training' and is_training_example(k):
      example['entity_emb_list'] = v['entity_emb_list']
      examples.append(example)
    if export_type == 'validation' and is_validation_example(k):
      example['entity_emb_list'] = v['entity_emb_list']
      examples.append(example)
    if export_type == 'testing' and is_testing_example(k):
      example['entity_emb_list'] = v['entity_emb_list']
      examples.append(example)

  logging.info('Loaded %s images for %s.', len(examples), export_type)
  return examples


def _get_captions(api, image_ids):
  meta_list = api.get_meta_list_by_ids(image_ids)
  captions = []
  for meta in meta_list:
    if 'action_reason_captions' in meta:
      captions.extend(meta['action_reason_captions'])
  return captions


def _tokenize(caption):
  return nltk.word_tokenize(caption.lower())


def _create_vocab(captions, min_count=2):
  vocab = {}
  for caption in captions:
    for word in _tokenize(caption):
      vocab[word] = vocab.get(word, 0) + 1

  logging.info('number of words: %d.', len(vocab))
  for k in vocab.keys():
    if vocab[k] < min_count:
      del vocab[k]
  logging.info('after filtering low frequency words: %d.', len(vocab))
  return vocab


def _check_coverage(vocab, captions):
  uncover = 0
  for caption in captions:
    for word in _tokenize(caption):
      if not word in vocab:
        uncover += 1
        break
  logging.info('coverage: %.4lf', 1.0 - 1.0 * uncover / len(captions))


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

  entity_embs = data['entity_embs'].reshape((-1))
  assert entity_embs.shape[0] == FLAGS.max_entities_per_image * data['entity_embs'].shape[1]

  example = tf.train.Example(features=tf.train.Features(feature={
        'image/source_id': dataset_util.bytes_feature(data['image_id'].encode('utf8')),
        'entity/num_entities': dataset_util.int64_feature(data['num_entities']),
        'entity/embeddings': dataset_util.float_list_feature(entity_embs.tolist()),
        'caption/num_captions': dataset_util.int64_feature(data['num_captions']),
        'caption/caption_lengths': dataset_util.int64_list_feature(data['caption_lengths']),
        'caption/caption_strings': dataset_util.int64_list_feature(caption_strings),
        }))
  return example


def _pad_captions(caption_strings, vocab):
  """Pad multiple captions for a single image patch.
    
  Returns:
    num_captions: number of captions, should be less than max_annots_per_image.
    lengths: length for each of the captions, shape=[max_annots_per_image].
    captions: ids for each of the captions, shape=[max_annots_per_image, max_caption_len].
  """
  unk_id = 0

  caption_strings = caption_strings[:FLAGS.max_annots_per_image]

  num_captions = len(caption_strings)
  lengths = []
  captions = []

  for caption in caption_strings:
    caption = [vocab.get(w, unk_id) for w in _tokenize(caption)]
    caption = caption[:FLAGS.max_caption_len]

    lengths.append(len(caption))
    while len(caption) != FLAGS.max_caption_len:
      caption.append(0)
    captions.append(caption)
  while len(captions) != FLAGS.max_annots_per_image:
    captions.append([0] * FLAGS.max_caption_len)
  while len(lengths) != FLAGS.max_annots_per_image:
    lengths.append(0)

  assert num_captions <= FLAGS.max_annots_per_image
  assert len(lengths) == FLAGS.max_annots_per_image
  assert len(captions) == FLAGS.max_annots_per_image
  for caption in captions:
    assert len(caption) == FLAGS.max_caption_len

  return num_captions, lengths, captions

def _pad_entities(entity_emb_list):
  entity_emb_list = entity_emb_list[:FLAGS.max_entities_per_image]
  num_entities = entity_emb_list.shape[0]

  if num_entities < FLAGS.max_entities_per_image:
    entity_emb_list = np.concatenate([entity_emb_list, 
        np.zeros((FLAGS.max_entities_per_image - num_entities, entity_emb_list.shape[1])) ], 
        axis=0)

  assert num_entities <= FLAGS.max_entities_per_image
  assert entity_emb_list.shape == (FLAGS.max_entities_per_image, 1536)
  return num_entities, entity_emb_list

def create_tf_record(output_path, emb_list, api, vocab):
  count = 0
  writer = tf.python_io.TFRecordWriter(output_path)
  for elem_index, elem in enumerate(emb_list):
    if elem_index % 100 == 0:
      logging.info('On image %d of %d.' % (elem_index, len(emb_list)))

    image_id = elem['image_id']
    entity_emb_list = elem['entity_emb_list']

    meta = api.get_meta_list_by_ids([image_id])[0]
    captions = meta.get('action_reason_captions', [])

    num_captions, lengths, captions = _pad_captions(captions, vocab)
    num_entities, entity_emb_list = _pad_entities(entity_emb_list)

    example = {
      'image_id': image_id,
      'num_entities': num_entities,
      'entity_embs': entity_emb_list,
      'num_captions': num_captions,
      'caption_lengths': lengths,
      'caption_strings': captions,
    }
    if num_entities > 0 and num_captions > 0:
      tf_example = dict_to_tf_example(example)
      writer.write(tf_example.SerializeToString())
      count += 1
  writer.close()
  tf.logging.info("Write %s records.", count)

def main(_):
  logging.basicConfig(level=logging.DEBUG)

  api = AdsDatasetApi()
  api.init(
      images_dir=FLAGS.data_dir,
      qa_action_annot_file=FLAGS.qa_action_annot_file,
      qa_reason_annot_file=FLAGS.qa_reason_annot_file,
      qa_action_reason_annot_file=FLAGS.qa_action_reason_annot_file)

  # Read data file.
  raw_embs = np.load(FLAGS.emb_path).item()
  train_embs = _get_data(raw_embs, export_type='training')
  valid_embs = _get_data(raw_embs, export_type='validation')
  test_embs = _get_data(raw_embs, export_type='testing')

  # Create vocabulary.
  train_captions = _get_captions(api, [example['image_id'] for example in train_embs])
  valid_captions = _get_captions(api, [example['image_id'] for example in valid_embs])
  test_captions = _get_captions(api, [example['image_id'] for example in test_embs])

  vocab = _create_vocab(train_captions)
  _check_coverage(vocab, train_captions)
  _check_coverage(vocab, valid_captions)
  _check_coverage(vocab, test_captions)

  vocab = _save_and_index_vocab(vocab, FLAGS.vocab_path)

  train_output_path = os.path.join(FLAGS.output_dir,
      'ads_entities_qa.train.record')
  val_output_path = os.path.join(FLAGS.output_dir,
      'ads_entities_qa.val.record')

  create_tf_record(train_output_path, train_embs, api, vocab)
  create_tf_record(val_output_path, valid_embs, api, vocab)

if __name__ == '__main__':
  tf.app.run()
