
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import cv2
import time
import nltk
from nltk.stem import PorterStemmer

from google.protobuf import text_format
from protos import ads_emb_model_pb2

import numpy as np
import tensorflow as tf

import ads_emb_model

from utils import vis
from utils import ads_api

from train import FLAGS
from train import default_session_config_proto

flags = tf.app.flags

flags.DEFINE_string('api_config', 'configs/ads_api_topics.config', 'Path to config file.')
flags.DEFINE_integer('max_image_size', 800, 'Maximum value of image size.')
flags.DEFINE_string('output_path', '', 'Path to the output file.')

slim = tf.contrib.slim

def _get_meta_list():
  api = ads_api.AdsApi(FLAGS.api_config)
  return api.get_meta_list()

def _export_emb_vecs(image_placeholder, image_emb, sess, meta_list):
  """Exports embedding vectors.

  Args:
    image_placeholder: a [height, width, 3] placeholder.
    image_emb: a [emb_size] tensor.
    sess: tensorflow session object.
    meta_list: a list of meta info.

  Returns:
    result: a python dict mapping from image_id to [emb_size] numpy arry
  """
  result = {}
  for i, meta in enumerate(meta_list):
    image = vis.image_load(meta['file_path'], True)

    if i % 50 == 0:
      tf.logging.info('On image %d/%d.', i, len(meta_list))

    height, width, _ = image.shape
    if height > FLAGS.max_image_size or width > FLAGS.max_image_size:
      image = cv2.resize(image, (FLAGS.max_image_size, FLAGS.max_image_size))

    emb = sess.run(image_emb, feed_dict={image_placeholder: image})
    result[meta['image_id']] = emb

  return result


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  meta_list = _get_meta_list()
  tf.logging.info('Load %s records.', len(meta_list))

  g = tf.Graph()
  with g.as_default():
    image_placeholder = tf.placeholder(
        dtype=tf.uint8, shape=[None, None, 3])

    # Create ads embedding model.
    model_proto = ads_emb_model_pb2.AdsEmbModel()
    with open(FLAGS.model_config, 'r') as fp:
      text_format.Merge(fp.read(), model_proto)
    model = ads_emb_model.AdsEmbModel(model_proto)

    # Get image embedding vector.
    image_embs, assign_fn_img = model.build_image_model(
        tf.expand_dims(image_placeholder, 0), is_training=False)
    image_embs = tf.nn.l2_normalize(image_embs, 1)
    image_emb = image_embs[0, :]

    global_step = slim.get_or_create_global_step()

    # Variables to restore, ignore variables in the pre-trained model.
    variables_to_restore = tf.global_variables()
    variables_to_restore = filter(
        lambda x: 'MobilenetV1' not in x.op.name, variables_to_restore)
    variables_to_restore = filter(
        lambda x: 'InceptionV4' not in x.op.name, variables_to_restore)
    variables_to_restore = filter(
        lambda x: 'BoxPredictor' not in x.op.name, variables_to_restore)
    invalid_tensor_names = tf.report_uninitialized_variables()
    saver = tf.train.Saver(variables_to_restore)

  with tf.Session(graph=g, config=default_session_config_proto()) as sess:
    assign_fn_img(sess)

    model_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)
    assert model_path is not None

    saver.restore(sess, model_path)
    names = sess.run(invalid_tensor_names)
    assert len(names) == 0

    step = sess.run(global_step)
    tf.logging.info('Using model at %s steps.', step)

    result = _export_emb_vecs(
        image_placeholder, image_emb, sess, meta_list)

  with open(FLAGS.output_path, 'wb') as fp:
    np.save(fp, result)

if __name__ == '__main__':
  tf.app.run()

