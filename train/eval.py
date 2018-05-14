
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import nltk
import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging

from google.protobuf import text_format

from protos import pipeline_pb2
from readers import ads_mem_examples
from models import builder
from utils import train_utils
from readers.utils import load_action_reason_annots

import eval_utils

flags.DEFINE_float('per_process_gpu_memory_fraction', 0.5, 
                   'GPU usage limitation.')

flags.DEFINE_string('pipeline_proto', '', 
                    'Path to the pipeline proto file.')

flags.DEFINE_string('action_reason_annot_path', '', 
                    'Path to the action-reason annotation file.')

flags.DEFINE_string('train_log_dir', '', 
                    'The directory where the graph and checkpoints are saved.')

flags.DEFINE_string('eval_log_dir', '', 
                    'The directory where the eval event files are saved.')

flags.DEFINE_string('saved_ckpt_dir', '', 
                    'The directory where the checkpoints of the best model is saved.')

flags.DEFINE_boolean('continuous_evaluation', True, 
                    '''If True, continously evaluate the latest model using the
                    validation set and move it to saved_ckpt_dir if it is
                    improved; otherwise, evaluate only the best model on the test
                    set.''')

flags.DEFINE_integer('number_of_steps', 0, 
                    'If none zero, use this value instead of that in the config file .')

flags.DEFINE_string('final_results_path', '', 
                    'Path to the saved result file.')


FLAGS = flags.FLAGS
slim = tf.contrib.slim


def load_pipeline_proto(filename):
  """Loads pipeline proto file.

  Args:
    filename: path to the pipeline proto file.

  Returns:
    pipeline_proto: an instance of pipeline_pb2.Pipeline
  """
  pipeline_proto = pipeline_pb2.Pipeline()
  with open(filename, 'r') as fp:
    text_format.Merge(fp.read(), pipeline_proto)
  if FLAGS.number_of_steps > 0:
    pipeline_proto.train_config.number_of_steps = FLAGS.number_of_steps
    pipeline_proto.eval_config.number_of_steps = FLAGS.number_of_steps
  return pipeline_proto


def _load_vocab(filename):
  """Loads vocabulary.

  Args:
    filename: path to the vocabulary file.

  Returns:
    a list mapping from id to word.
  """
  with open(filename, 'r') as fp:
    vocab = ['UNK'] + [x.strip('\n').split('\t')[0] for x in fp.readlines()]
  return vocab


def export_inference(results, groundtruths, filename):
  """Exports results to a specific file.

  Args:
    results: 
    groundtruths:
    filename: the path to the output json file.
  """
  final_results = {}
  for image_id, result in results.iteritems():
    pred = np.array(result['distances']).argmin()
    final_results[image_id] = groundtruths[image_id]['all_examples'][pred]

  with open(filename, 'w') as fp:
    fp.write(json.dumps(final_results))

def evaluate_once(sess, writer, global_step, predictions, groundtruths):
  """Evaluates model.

  Args:
    annots: a list of ads annotations.
    sess: the tf.Session.
    writer: summary writer object.
    global_step: global step of current model.
    predictions: the tensor predictions.
  """
  # Loop through the evaluation dataset.
  results = {}
  try:
    while True:
      pred_vals = sess.run(predictions)
      for image_id, distances in zip(
          pred_vals['image_id'], pred_vals['distance']):

        results[image_id] = {
          'distances': map(lambda x: round(x, 5), distances.tolist()),
        }

  except tf.errors.OutOfRangeError:
    logging.info('Done evaluating -- epoch limit reached')

  if not FLAGS.continuous_evaluation:
    export_inference(results, groundtruths, FLAGS.final_results_path)
    return None

  metrics = eval_utils.evaluate(results, groundtruths)
  logging.info("Evaluation results at %i: \n%s", 
      global_step, json.dumps(metrics, indent=2))

  if writer is not None:
    summary = tf.Summary()
    for k, v in metrics.iteritems():
      summary.value.add(tag='metrics/{}'.format(k), simple_value=v)
    writer.add_summary(summary, global_step=global_step)

  # Save results.
  return metrics['accuracy']

def main(_):
  logging.set_verbosity(tf.logging.INFO)

  assert os.path.isfile(FLAGS.pipeline_proto)
  assert os.path.isfile(FLAGS.action_reason_annot_path)

  pipeline_proto = load_pipeline_proto(FLAGS.pipeline_proto)
  logging.info("Pipeline configure: %s", '=' * 128)
  logging.info(pipeline_proto)

  groundtruths = load_action_reason_annots(FLAGS.action_reason_annot_path)

  g = tf.Graph()
  with g.as_default():
    # Get examples from reader.
    split = 'valid'
    if not FLAGS.continuous_evaluation:
      split = 'test'

    examples, feed_init_fn = ads_mem_examples.get_examples(
        pipeline_proto.example_reader, split)

    # Build model for training.
    global_step = slim.get_or_create_global_step()

    model = builder.build(pipeline_proto.model, is_training=False)
    predictions = model.build_evaluation_graph(examples)

    init_fn = model.get_init_fn()
    uninitialized_variable_names = tf.report_uninitialized_variables()

    saver = tf.train.Saver()
    init_op = tf.group(tf.local_variables_initializer(), 
        tf.global_variables_initializer())

  session_config = train_utils.default_session_config( 
      FLAGS.per_process_gpu_memory_fraction)

  # evaluation on test set.
  logging.info('Start evaluating.')
  eval_config = pipeline_proto.eval_config

  # One time evaluation and inference.
  if not FLAGS.continuous_evaluation:
    model_path = train_utils.get_latest_model(FLAGS.saved_ckpt_dir)
    with tf.Session(graph=g, config=session_config) as sess:
      feed_init_fn(sess)
      sess.run(init_op)
      saver.restore(sess, model_path)
      logging.info('Restore model from %s.', model_path)
      warn_names = sess.run(uninitialized_variable_names)
      assert len(warn_names) == 0

      # Evaluate the best model in terms of recall@3.
      step = sess.run(global_step)
      evaluate_once(sess, None, step, predictions, groundtruths)

    logging.info('Done')

    exit(0)

  # Continuous evaluation on valid set.
  writer = tf.summary.FileWriter(FLAGS.eval_log_dir, g)
  step = prev_step = -1
  while True:
    start = time.time()

    try:
      model_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)

      if model_path is not None:
        with tf.Session(graph=g, config=session_config) as sess:
          # Restore model.
          feed_init_fn(sess)
          sess.run(init_op)
          saver.restore(sess, model_path)
          logging.info('Restore model from %s.', model_path)

          warn_names = sess.run(uninitialized_variable_names)
          assert len(warn_names) == 0

          step = sess.run(global_step)
          if step != prev_step and step > eval_config.eval_min_global_steps:
            # Evaluate the latest model.
            prev_step = step
            metric = evaluate_once(sess, writer, step, predictions, groundtruths)

            step_best, metric_best = train_utils.save_model_if_it_is_better(
                step, metric, model_path, FLAGS.saved_ckpt_dir, reverse=False)

            if step_best == step:
              summary = tf.Summary()
              summary.value.add(tag='metrics/model_metric', simple_value=metric_best)
              writer.add_summary(summary, global_step=step)
            writer.flush()

        # with tf.Session
      # if model_path is not None
    except Exception as ex:
      pass

    if step >= eval_config.number_of_steps:
      break

    sleep_secs = eval_config.eval_interval_secs - (time.time() - start)
    if sleep_secs > 0:
      logging.info('Now sleep for %.2lf secs.', sleep_secs)
      time.sleep(sleep_secs)

  writer.close()
  logging.info('Done')

if __name__ == '__main__':
  app.run()
