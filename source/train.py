
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import ads_entity_qa_examples
from autoencoder import AutoEncoder
from bow import BOW

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to tf record file for training.')
flags.DEFINE_string('vocab_path', '', 'Path to vocab file.')

flags.DEFINE_integer('number_of_steps', 10000, 'Maximum number of steps.')
flags.DEFINE_integer('batch_size', 32, 'Batch size of the training process.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training.')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training, adam|adagrad.')
flags.DEFINE_integer('embedding_size', 256, 'Embedding size.')
flags.DEFINE_float('reconstruction_loss_weight', 1.0, 'Learning rate for training.')
flags.DEFINE_float('triplet_alpha', 0.3, 'Margin for optimizing triplet loss.')

flags.DEFINE_string('train_log_dir', '', 'The directory where the graph and checkpoints are saved.')
flags.DEFINE_integer('log_every_n_steps', 100, 'Log every n steps.')
flags.DEFINE_integer('save_interval_secs', 180, 'Save checkpoint secs.')
flags.DEFINE_integer('save_summaries_secs', 60, 'Save summaries secs.')

flags.DEFINE_bool('moving_average', True, 'Whether to use moving average.')
FLAGS = flags.FLAGS
slim = tf.contrib.slim


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


config = {
  'n_hidden': FLAGS.embedding_size,
  'weight_decay': 0.0,
  'keep_prob': 0.7,
  'use_batch_norm': True,
  'reconstruction_loss': 'l1_loss',
}
vocab, _ = load_vocab(FLAGS.vocab_path)
config_bow = {
  'embedding_size': FLAGS.embedding_size,
  'vocab_size': len(vocab),
  'weight_decay': 0.0,
  'keep_prob': 0.7
}


def default_session_config_proto():
  """Get the default config proto for tensorflow session.

  Returns:
    config: The default config proto for tf.Session.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.48
  return config


def mine_positives(num_captions):
  """Mine positive examples from training data.

  Args:
    num_captions: a [batch] tensor indicating number of captions for each image.

  Returns:
    caption_indices: a [batch, 2] indicating selected caption.
  """
  batch_size = FLAGS.batch_size
  main_indices = tf.range(batch_size, dtype=tf.int64)
  caption_indices = tf.mod(
      tf.random_uniform([batch_size], maxval=10000, dtype=tf.int64),
      num_captions)
  caption_indices = tf.stack([main_indices, caption_indices], axis=1)
  return caption_indices


def mine_negatives(positives):
  """Mine negative examples from positive examples.

  Args:
    positives: a [batch, embedding_size] tensor indicating positive examples.

  Returns:
    negatives: a [batch, embedding_size] tensor indicating negative examples.
  """
  batch_size = FLAGS.batch_size
  indices = tf.add(
      tf.range(batch_size, dtype=tf.int64),
      tf.random_uniform(
        shape=[batch_size], minval=1, maxval=batch_size, dtype=tf.int64))
  indices = tf.mod(indices, batch_size)
  return tf.gather(positives, indices)


def average_entity_embeddings(patch_embeddings, boolean_masks):
  """Average entity embeddings to get anchors.

  Args:
    patch_embeddings: a [batch, max_entities, embedding_size] tensor indicating embeddings of each patch.
    boolean_masks: a [batch, max_entities] indicating valid entities.

  Returns:
    embeddings_averaged: a [batch, embedding_size] tensor storing averaged patch embeddings for each image.
  """
  max_entities = patch_embeddings.get_shape()[1].value
  weights = tf.cast(boolean_masks, tf.float32)
  num_entities = tf.reduce_sum(weights, axis=1)
  weights = tf.div(
      weights, 
      1e-12 + tf.tile(tf.expand_dims(num_entities, 1), [1, max_entities])
      )
  weights = tf.expand_dims(weights, axis=1)
  embeddings_averaged = tf.squeeze(tf.matmul(weights, patch_embeddings), [1])
  return embeddings_averaged


def unit_norm(x):
  """Compute unit norm for tensor x.

  Args:
    x: a [batch, embedding_size] tensor.

  Returns:
    x_unit: a [batch, embedding_size] tensor that is normalized.
  """
  embedding_size = x.get_shape()[1].value
  x_norm = tf.tile(tf.norm(x, axis=1, keep_dims=True), [1, embedding_size])
  return x / (x_norm + 1e-12)


def compute_triplet_loss(anchors, positives, negatives, alpha=0.3):
  """Compute triplet loss.

  Args:
    anchors: a [batch, embedding_size] tensor.
    positives: a [batch, embedding_size] tensor.
    negatives: a [batch, embedding_size] tensor.

  Returns:
    triplet_loss: a scalar tensor.
  """
  cosine_distance_fn = lambda x, y: 1 - tf.reduce_sum(tf.multiply(x, y), axis=1)

  dist1 = cosine_distance_fn(anchors, positives)
  dist2 = cosine_distance_fn(anchors, negatives)

  losses = tf.maximum(dist1 - dist2 + alpha, 0)
  losses = tf.boolean_mask(losses, losses > 0)

  loss = tf.cond(
      tf.shape(losses)[0] > 0,
      lambda: tf.reduce_mean(losses),
      lambda: 0.0)
  tf.losses.add_loss(loss)

  # Gather statistics.
  loss_ratio = tf.count_nonzero(
      dist1 + alpha >= dist2, dtype=tf.float32) / FLAGS.batch_size
  good_ratio = tf.count_nonzero(
      dist1 < dist2, dtype=tf.float32) / FLAGS.batch_size
  bad_ratio = 1 - good_ratio

  return {
    'losses/triplet_loss': loss,
    'triplet/good_ratio': good_ratio,
    'triplet/bad_ratio': bad_ratio,
    'triplet/loss_ratio': loss_ratio,
  }


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  g = tf.Graph()
  with g.as_default():
    # Gets input data.
    examples = ads_entity_qa_examples.get_examples(
        filename=FLAGS.input_path,
        batch_size=FLAGS.batch_size)

    num_entities = examples['num_entities']
    inc_v4_embs = examples['embeddings']
    num_captions = examples['num_captions']
    caption_lengths = examples['caption_lengths']
    caption_strings = examples['caption_strings']

    _, max_num_entities, embedding_dims = inc_v4_embs.get_shape().as_list()

    boolean_masks = tf.less(
      tf.range(max_num_entities, dtype=tf.int64),
      tf.expand_dims(num_entities, 1))

    # Build autoencoder model.
    inc_v4_embs = tf.boolean_mask(inc_v4_embs, boolean_masks)

    model_vis = AutoEncoder(config)
    hidden, reconstruction = model_vis.build(inc_v4_embs, is_training=True)
    autoencoder_loss_summaries = model_vis.build_loss(inc_v4_embs,
        reconstruction, weight=FLAGS.reconstruction_loss_weight)
    for k, v in autoencoder_loss_summaries.iteritems():
      tf.summary.scalar(k, v)

    # Get anchors by averaging embeddings of patches.
    sparse_indices = tf.where(boolean_masks)
    lookup = tf.sparse_to_dense(sparse_indices, 
        output_shape=[FLAGS.batch_size, max_num_entities], 
        sparse_values=tf.range(tf.shape(hidden)[0]))

    patch_embeddings = tf.nn.embedding_lookup(hidden, lookup)
    anchors = average_entity_embeddings(patch_embeddings, boolean_masks)

    # Mine positive examples, one caption per image.
    caption_indices = mine_positives(num_captions)
    caption_lengths = tf.gather_nd(caption_lengths, caption_indices)
    caption_strings = tf.gather_nd(caption_strings, caption_indices)

    model_txt = BOW(config=config_bow)
    caption_embeddings, word_embeddings, embedding_weights = model_txt.build(
        caption_lengths, caption_strings, is_training=True)
    positives = caption_embeddings

    # Mine negative examples (unrelated captions), one caption per image.
    negatives = mine_negatives(positives)

    anchors = unit_norm(anchors)
    positives = unit_norm(positives) 
    negatives = unit_norm(negatives) 

    triplet_loss_summaries = compute_triplet_loss(
        anchors, positives, negatives, alpha=FLAGS.triplet_alpha)
    for k, v in triplet_loss_summaries.iteritems():
      tf.summary.scalar(k, v)

    # Logs info.
    tf.logging.info('*' * 128)
    tf.logging.info('global variables:')
    for v in tf.global_variables():
      tf.logging.info('%s: %s', v.op.name, v.get_shape().as_list())

    regularization_loss = tf.losses.get_regularization_loss()
    tf.summary.scalar('losses/regularization_loss', regularization_loss)

    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Optimizer.
    optimizer = None
    if FLAGS.optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    else:
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.moving_average:
      optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer,
          average_decay=0.99)
    train_op = tf.contrib.training.create_train_op(total_loss,
        optimizer=optimizer)

    saver = None
    if FLAGS.moving_average:
      saver = optimizer.swapping_saver()

#    with tf.Session() as sess:
#      sess.run(tf.global_variables_initializer())
#      coord = tf.train.Coordinator()
#      threads = tf.train.start_queue_runners(coord=coord)
#
#      for i in xrange(10000):
#        sess.run(train_op)
#
#        if i % 100 == 0:
#          example = {
#            'dist1': dist1,
#            'dist2': dist2,
#            'anchors': anchors,
#            'positives': positives,
#            'negatives': negatives,
#          }
#          example = sess.run(example)
#
#      coord.request_stop()
#      coord.join(threads, stop_grace_period_secs=10)

  # Starts training.
  slim.learning.train(train_op, 
      logdir=FLAGS.train_log_dir,
      graph=g,
      number_of_steps=FLAGS.number_of_steps,
      log_every_n_steps=FLAGS.log_every_n_steps,
      save_interval_secs=FLAGS.save_interval_secs,
      save_summaries_secs=FLAGS.save_summaries_secs,
      session_config=default_session_config_proto(),
      saver=saver)

if __name__ == '__main__':
  tf.app.run()
