
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow import logging

from protos import train_config_pb2

slim = tf.contrib.slim


def build_multipler(config):
  """Builds gradient multipliers from config.

  Args:
    config: an instance of train_config_pb2.GradientMultiplier.

  Raises:
    ValueError: if config is invalid.

  Returns:
    a dict mapping from variable name to multiplier.
  """
  multipliers = {}
  for multiplier_config in config:
    scope = multiplier_config.scope
    multiplier = multiplier_config.multiplier

    variables = [x for x in tf.trainable_variables() if scope in x.op.name]
    for v in variables:
      multipliers[v.op.name] = multiplier
  return multipliers


def build_optimizer(config):
  """Builds optimizer from config.

  Args:
    config: an instance of train_config_pb2.TrainConfig.

  Raises:
    ValueError: if config is invalid.

  Returns:
    a tensorflow optimizer instance.
  """
  if not isinstance(config, train_config_pb2.TrainConfig):
    raise ValueError('The config has to be an instance of TrainConfig.')

  learning_rate = config.learning_rate

  global_step = slim.get_or_create_global_step()

  if config.learning_rate_decay_rate < 1.0:
    learning_rate = tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_steps=config.learning_rate_decay_steps,
        decay_rate=config.learning_rate_decay_rate,
        staircase=config.learning_rate_staircase)

  tf.summary.scalar('losses/learning_rate', learning_rate)

  optimizer = config.optimizer.WhichOneof('optimizer')

  if 'adagrad' == optimizer:
    return tf.train.AdagradOptimizer(learning_rate)

  if 'adam' == optimizer:
    return tf.train.AdamOptimizer(learning_rate)

  if 'rmsprop' == optimizer:
    return tf.train.RMSPropOptimizer(
        learning_rate,
        decay=config.optimizer.rmsprop.decay,
        momentum=config.optimizer.rmsprop.momentum)

  raise ValueError('Invalid optimizer: {}.'.format(config.optimizer))


def default_session_config(per_process_gpu_memory_fraction=1.0):
  """Get the default session config for tensorflow session.

  Args:
    per_process_gpu_memory_fraction: the maximum fraction of gpu memory that 
      the process can use.

  Returns:
    config: The default config proto for tf.Session.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  #config.log_device_placement = True
  config.gpu_options.per_process_gpu_memory_fraction \
    = per_process_gpu_memory_fraction
  return config


def save_model_if_it_is_better(global_step, model_metric, 
    model_path, saved_ckpts_dir, reverse=False):
  """Save model if it is better than previous best model.

  The function backups model checkpoint if it is a better model.

  Args:
    global_step: a integer denoting current global step.
    model_metric: a float number denoting performance of current model.
    model_path: current model path.
    saved_ckpt_dir: the directory used to save the best model.
    reverse: if True, smaller value means better model.

  Returns:
    step_best: global step of the best model.
    metric_best: performance of the best model.
  """
  # Read the record file to get the previous best model.
  filename = 'saved_info.txt'
  filename = os.path.join(saved_ckpts_dir, filename)

  step_best, metric_best = None, None
  if tf.gfile.Exists(filename):
    with open(filename, 'r') as fp:
      step_best, metric_best = fp.readline().strip().split('\t')
    step_best, metric_best = int(step_best), float(metric_best)

  condition = lambda x, y: (x > y) if not reverse else (x < y)

  if metric_best is None or condition(model_metric, metric_best):
    logging.info(
        'Current model[%.4lf] is better than the previous best one[%.4lf].',
        model_metric, 0.0 if metric_best is None else metric_best)
    step_best, metric_best = global_step, model_metric

    # Backup checkpoint files.
    logging.info('Copying files...')
    tf.gfile.MakeDirs(saved_ckpts_dir)

    with open(filename, 'w') as fp:
      fp.write('%d\t%.8lf' % (global_step, model_metric))

    for existing_path in tf.gfile.Glob(
        os.path.join(saved_ckpts_dir, 'model.ckpt*')):
      tf.gfile.Remove(existing_path)
      logging.info('Remove %s.', existing_path)

    for source_path in tf.gfile.Glob(model_path + '*'):
      dest_path = os.path.join(saved_ckpts_dir, os.path.split(source_path)[1])
      tf.gfile.Copy(source_path, dest_path, overwrite=True)
      logging.info('Copy %s to %s.', source_path, dest_path)
  return step_best, metric_best

def get_latest_model(saved_ckpts_dir):
  """Returns the latest model (in terms of checkpoint number) in the directory.

  Args:
    saved_ckpts_dir: the path to the directory that saves checkpoints.

  Returns:
    model_path: path to the latest model.
  """
  path_list = []
  for file_path in tf.gfile.Glob(os.path.join(saved_ckpts_dir, 'model.ckpt-*.meta')):
    path_list.append(file_path)

  if len(path_list) == 0:
    raise ValueError('No checkpoint was found in %s.' % (saved_ckpts_dir))

  ckpt_fn = lambda x: int(re.findall('ckpt-(\d+).meta', x)[0])
  model_path = sorted(path_list, lambda x, y: -cmp(ckpt_fn(x), ckpt_fn(y)))[0]
  model_path = model_path[:-5]
  return model_path
