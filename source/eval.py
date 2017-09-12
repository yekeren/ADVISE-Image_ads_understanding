
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import math

import numpy as np
import tensorflow as tf

import ads_entity_qa_examples
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
flags.DEFINE_integer('eval_n_examples', 10000, 'Number of examples to evaluate.')
flags.DEFINE_string('eval_log_dir', '', 'The directory where the graph is saved.')
flags.DEFINE_integer('sleep_secs', 180, 'Seconds to sleep when there is no checkpoint.')

slim = tf.contrib.slim
ckpt_path = None

def evaluate_once(sess, writer, global_step, metrics):
  num_batches = int(math.ceil(FLAGS.eval_n_examples / FLAGS.batch_size))
  final_results = {}
  for i in xrange(num_batches):
    results = sess.run(metrics)
    for k, v in results.iteritems():
      final_results.setdefault(k, []).append(float(v))

  summary = tf.Summary()
  for k, v in final_results.iteritems():
    v = sum(v) / len(v)
    summary.value.add(tag=k, simple_value=v)
    logging.info('%s: %.4lf', k, v)
  writer.add_summary(summary, global_step=global_step)
  writer.flush()


def evaluation_loop(sess, saver, writer, global_step, metrics):
  global ckpt_path

  while True:
    time.sleep(FLAGS.sleep_secs)

    model_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)
    if model_path and ckpt_path != model_path:
      ckpt_path = model_path
      saver.restore(sess, model_path)
      logging.info('*' * 128)
      logging.info('Load checkpoint %s.', model_path)

      # Evaluate.
      step = sess.run(global_step)
      logging.info('Global step=%s.', step)
      evaluate_once(sess, writer, step, metrics)

      logging.info('Finish evaluation.')
      if step >= FLAGS.number_of_steps:
        logging.info('Break evaluation_loop.')
        break
    else:
      logging.info('No new checkpoint was found in %s.', FLAGS.train_log_dir)

def main(_):
  logging.basicConfig(level=logging.DEBUG)

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
    hidden, reconstruction = model_vis.build(inc_v4_embs, is_training=False)
    autoencoder_loss_summaries = model_vis.build_loss(inc_v4_embs, reconstruction)

    # Get anchors by averaging embeddings of patch.
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
    caption_embeddings, word_embeddings, embedding_weights  = model_txt.build(
        caption_lengths, caption_strings, is_training=False)
    positives = caption_embeddings

    # Mine negative examples (unrelated captions), one caption per image.
    negatives = mine_negatives(positives)

    anchors = unit_norm(anchors)
    positives = unit_norm(positives) 
    negatives = unit_norm(negatives) 

    triplet_loss_summaries = compute_triplet_loss(
        anchors, positives, negatives, alpha=FLAGS.triplet_alpha)

    # Logs info.
    tf.logging.info('*' * 128)
    tf.logging.info('global variables:')
    for v in tf.global_variables():
      tf.logging.info('%s: %s', v.op.name, v.get_shape().as_list())

    regularization_loss = tf.losses.get_regularization_loss()
    total_loss = tf.losses.get_total_loss()
    metrics = {
      'losses/total_loss': total_loss,
      'losses/regularization_loss': regularization_loss,
    }
    metrics.update(autoencoder_loss_summaries)
    metrics.update(triplet_loss_summaries)

    global_step = slim.get_or_create_global_step()
    saver = tf.train.Saver()

  with tf.Session(graph=g, config=default_session_config_proto()) as sess:
    writer = tf.summary.FileWriter(FLAGS.eval_log_dir, g)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    evaluation_loop(sess, saver, writer, global_step, metrics)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    writer.close()


if __name__ == '__main__':
  tf.app.run()
