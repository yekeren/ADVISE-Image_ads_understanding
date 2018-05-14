
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random

import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging

from google.protobuf import text_format
from sklearn.metrics import average_precision_score

from readers.utils import load_raw_annots
from readers.utils import load_symbol_cluster

from utils import mlp
from utils import train_utils

from protos import mlp_pb2

flags.DEFINE_string('model_proto', 
                    'configs/symbol_classifier.pbtxt', 
                    'Path to the model proto file.')

flags.DEFINE_string('symbol_cluster_path', 
                    'data/additional/clustered_symbol_list.json', 
                    'Path to the symbol cluster json files.')

flags.DEFINE_string('symbol_annot_path', 
                    'output/symbol_train.json', 
                    'Path to the symbol annotation file.')

flags.DEFINE_string('feature_path', 'output/img_features_train.npy', 
                    'Path to the feature data file.')

flags.DEFINE_string('output_model_path', 'output/symbol_classifier/model.ckpt', 
                    'Path to the output model files.')

flags.DEFINE_integer('number_of_val_examples', 1500, 
                     'Number of validation examples.')

flags.DEFINE_integer('max_iters', 8000, 'Maximum iterations.')

flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')


FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _load_model_proto(filename):
  """Loads model proto file.

  Args:
    filename: path to the model proto file.

  Returns:
    model_proto: an instance of mlp_pb2.MLPModel
  """
  model_proto = mlp_pb2.MLPModel()
  with open(filename, 'r') as fp:
    text_format.Merge(fp.read(), model_proto)
  return model_proto


def _get_data(annots, img_features, number_of_classes):
  """Processes to get the features and labels.

  Args:
    annots: a dict mapping from image_id to a list of integers.
    img_features: a dict mapping from image_id to features.

  Returns:
    x: a [total, feature_dims] np array.
    y: a [total, n_labels] np array.
  """
  x, y = [], []
  for image_id, annot in annots.iteritems():
    label = np.zeros((number_of_classes), dtype=np.float32)
    for i in annot:
      label[i] = 1
    x.append(img_features[image_id])
    y.append(label)
  return np.stack(x, axis=0), np.stack(y, axis=0)


def main(_):
  logging.set_verbosity(tf.logging.INFO)

  model_proto = _load_model_proto(FLAGS.model_proto)
  logging.info('Model proto: \n%s', model_proto)

  # Load vocab.
  word_to_id, id_to_symbol = load_symbol_cluster(FLAGS.symbol_cluster_path)
  logging.info('Number of classes: %i.', len(id_to_symbol))

  # Load image features.
  img_features = np.load(FLAGS.feature_path).item()
  logging.info('Load %i features.', len(img_features))

  # Note that ZERO is reserved for 'unclear'.
  annots = load_raw_annots(FLAGS.symbol_annot_path)
  x, y = _get_data(annots, img_features, len(id_to_symbol))

  number_of_val_examples = FLAGS.number_of_val_examples
  x_train, y_train= x[number_of_val_examples:], y[number_of_val_examples:]
  x_valid, y_valid = x[:number_of_val_examples], y[:number_of_val_examples]

  logging.info('Load %d train examples.', len(x_train))
  logging.info('Load %d valid examples.', len(x_valid))

  # Build graph to train symbol classifier.
  g = tf.Graph()
  with g.as_default():
    # For training
    logits, init_fn = mlp.model(model_proto, x_train, is_training=True)
    loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_train, logits=logits)
    loss_op = tf.reduce_mean(loss_op)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = slim.learning.create_train_op(loss_op, optimizer=optimizer)
    init_op = tf.global_variables_initializer()

    # For evaluation.
    tf.get_variable_scope().reuse_variables()
    logits_val, init_fn_val = mlp.model(model_proto, x_valid, is_training=False)

    saver = tf.train.Saver()

  with tf.Session(graph=g, config=train_utils.default_session_config()) as sess:
    sess.run(init_op)

    max_v, metrics = 0, []
    for i in xrange(FLAGS.max_iters):
      (_, loss, pred_train, pred_valid
       ) = sess.run([train_op, loss_op, logits, logits_val])

      mAP_micro = average_precision_score(
          y_valid[:, 1:], pred_valid[:, 1:], average='micro')
      mAP_macro = average_precision_score(
          y_valid[:, 1:], pred_valid[:, 1:], average='macro')
      metric = mAP_macro

      if i % 100 == 0:
        logging.info('step=%d, loss=%.4lf, mAP_micro=%.4lf, mAP_macro=%.4lf.', 
            i + 1, loss, mAP_micro, mAP_macro)

        if metric >= max_v:
          saver.save(sess, FLAGS.output_model_path)
          max_v = metric

        if len(metrics) >= 3:
          if metric < metrics[-1] and metrics[-1] < metrics[-2] and metrics[-2] < metrics[-3]:
            logging.info('Process early stopping.')
            break

        metrics.append(metric)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
