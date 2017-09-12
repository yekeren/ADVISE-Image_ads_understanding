from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import tensorflow as tf

import cv2
import numpy as np
from utils import vis

from autoencoder import AutoEncoder
from bow import BOW

from train import FLAGS
from train import config
from train import config_bow
from train import mine_positives
from train import mine_negatives
from train import average_entity_embeddings
from train import unit_norm
from train import compute_triplet_loss
from train import default_session_config_proto

flags = tf.app.flags
flags.DEFINE_string('emb_path', '', 'Path to the feature data file.')
flags.DEFINE_string('ckpt_path', '', 'Path to the checkpoint files.')
flags.DEFINE_string('output_visu_emb', '', 'Path to the output visual embedding file.')
flags.DEFINE_string('output_word_emb', '', 'Path to the output word embedding file.')


FLAGS = flags.FLAGS


def main(_):
  logging.basicConfig(level=logging.DEBUG)

  # Read data file.
  raw_embs = np.load(FLAGS.emb_path).item()

  g = tf.Graph()
  with g.as_default():
    # Creates model.
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 1536))

    # Get hidden activations from autoencoder model.
    model = AutoEncoder(config)
    hidden, reconstruction = model.build(inputs, is_training=False)

    # Get embedding weights from bow model.
    model_txt = BOW(config=config_bow)
    embedding_weights = model_txt.build_weights()

    saver = tf.train.Saver()
    invalid_tensor_names = tf.report_uninitialized_variables()

  # Get the visual embeddings.
  output = {}
  with tf.Session(graph=g, config=default_session_config_proto()) as sess:
    # Restore from checkpoint.
    ckpt_path = FLAGS.ckpt_path
    if not os.path.isfile(FLAGS.ckpt_path):
      ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    saver.restore(sess, ckpt_path)

    invalid_tensor_names = sess.run(invalid_tensor_names)
    if len(invalid_tensor_names) > 0:
      raise ValueError('There are uninitialized variables!')

    # Get the word embeddings.
    embedding_weights = sess.run(embedding_weights)

    # Get the visual embeddings.
    for index, (image_id, example) in enumerate(raw_embs.iteritems()):
      if index % 100 == 0:
        logging.info('On image %d of %d.', index, len(raw_embs))
      emb_data, output_data = sess.run([hidden, reconstruction], feed_dict={inputs: example['entity_emb_list']})
      output[image_id] = {
        'image_emb': example['image_emb'],
        'entity_emb_list': emb_data
      }
  with open(FLAGS.output_word_emb, 'wb') as fp:
    np.save(fp, embedding_weights)

  with open(FLAGS.output_visu_emb, 'wb') as fp:
    np.save(fp, output)


if __name__ == '__main__':
  tf.app.run()
