
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import text_format
from protos import ads_emb_model_pb2

import numpy as np
import tensorflow as tf

from source import ads_emb_model
from source import ads_qa_examples

flags = tf.app.flags
flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of hostname:port pairs.')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of hostname:port pairs.')
flags.DEFINE_string('job_name', 'worker', 'Either "ps" or "worker".')
flags.DEFINE_integer('task_index', 0, 'Index of task within the job, by default, worker 0 is the chief worker.')

flags.DEFINE_string('model_config', 'configs/ads_emb_model.pbtxt', 'Path to the configuration file.')

flags.DEFINE_integer('number_of_steps', 10000, 'Maximum number of steps.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training.')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training, adam|adagrad.')
flags.DEFINE_bool('moving_average', True, 'Whether to use moving average.')

flags.DEFINE_bool('tune_caption_model', True, 'If True, train caption model.')

flags.DEFINE_string('train_log_dir', '', 'The directory where the graph and checkpoints are saved.')
flags.DEFINE_integer('log_every_n_steps', 1, 'Log every n steps.')
flags.DEFINE_integer('save_interval_secs', 600, 'Save checkpoint secs.')
flags.DEFINE_integer('save_summaries_secs', 60, 'Save summaries secs.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def default_session_config_proto():
  """Get the default config proto for tensorflow session.

  Returns:
    config: The default config proto for tf.Session.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  return config


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Create a cluster from the parameter server and worker hosts.
  if FLAGS.ps_hosts:
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    cluster = tf.train.ClusterSpec({
        "ps": ps_spec, 
        "worker": worker_spec})

    server = tf.train.Server(cluster, 
        job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
      server.join()

  g = tf.Graph()
  with g.as_default():
    # Assigns ops to the local worker by default.
    device_fn = None
    if FLAGS.ps_hosts:
      device_fn = tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % (FLAGS.task_index),
            ps_device="/job:ps/cpu:0",
            cluster=cluster)

    with tf.device(device_fn):
      # Create ads embedding model.
      model_proto = ads_emb_model_pb2.AdsEmbModel()
      with open(FLAGS.model_config, 'r') as fp:
        text_format.Merge(fp.read(), model_proto)

      # Gets input data.
      assert model_proto.HasField('examples_reader')
      examples = ads_qa_examples.get_examples(
          model_proto.examples_reader)

      model = ads_emb_model.AdsEmbModel(model_proto)
      triplet_loss_summaries, assign_fn = model.build(
          images=examples['image'],
          num_captions=examples['num_captions'],
          caption_lengths=examples['caption_lengths'],
          caption_strings=examples['caption_strings'],
          num_detections=examples.get('num_detections', None),
          proposed_features=examples.get('proposed_features', None),
          topics=examples['topic'],
          densecap_num_captions=examples.get('densecap_num_captions', None),
          densecap_caption_lengths=examples.get('densecap_caption_lengths', None),
          densecap_caption_strings=examples.get('densecap_caption_strings', None),
          is_training=True)

      # Losses.
      for k, v in triplet_loss_summaries.iteritems():
        tf.summary.scalar(k, v)

      regularization_loss = tf.losses.get_regularization_loss()
      tf.summary.scalar('losses/regularization_loss', regularization_loss)

      total_loss = tf.losses.get_total_loss()
      #total_loss = triplet_loss_summaries['losses/triplet_loss_densecap_img']
      tf.summary.scalar('losses/total_loss', total_loss)

      # for v in tf.trainable_variables():
      #   print(v.op.name)

      # reader = tf.train.NewCheckpointReader('log3/mobilenet_v1.from_image.patch_repr.stn.adagrad_2.0/train/model.ckpt-3838')
      # variable_map = reader.get_variable_to_shape_map()
      # print('\n'.join(sorted(variable_map.keys())))

      # Optimizer.
      optimizer = None
      if FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
      elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

      if FLAGS.moving_average:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer,
            average_decay=0.99)

      variables_to_train = tf.trainable_variables()
      variables_to_train = filter(
          lambda x: 'MobilenetV1' not in x.op.name, variables_to_train)
      variables_to_train = filter(
          lambda x: 'InceptionV4' not in x.op.name, variables_to_train)
      variables_to_train = filter(
          lambda x: 'BoxPredictor' not in x.op.name, variables_to_train)
      if not FLAGS.tune_caption_model:
        variables_to_train = filter(
            lambda x: 'caption_encoder' not in x.op.name, variables_to_train)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      update_ops = filter(
          lambda x: 'MobilenetV1' not in x.op.name, update_ops)
      update_ops = filter(
          lambda x: 'InceptionV4' not in x.op.name, update_ops)
      update_ops = filter(
          lambda x: 'BoxPredictor' not in x.op.name, update_ops)
      if not FLAGS.tune_caption_model:
        update_ops = filter(
            lambda x: 'caption_encoder' not in x.op.name, update_ops)

      # Logs info.
      tf.logging.info('*' * 128)
      tf.logging.info('variables_to_train:')
      for v in variables_to_train:
        tf.logging.info('%s: %s', v.op.name, v.get_shape().as_list())

      tf.logging.info('*' * 128)
      tf.logging.info('update_ops:')
      for v in update_ops:
        tf.logging.info('%s', v.op.name)

      # Gradient multipliers.
      gradient_multipliers = {}
      for v in variables_to_train:
        if 'affine_transformer' in v.op.name:
          gradient_multipliers[v] = 0.004

        v_flat = tf.reshape(v, [-1])
        tf.summary.scalar('summarize_variables/%s_avg' % (v.op.name),
            tf.reduce_mean(v_flat))
        tf.summary.scalar('summarize_variables/%s_std' % (v.op.name),
            tf.sqrt(tf.maximum(
                1e-12, 
                tf.reduce_mean(tf.square(v_flat - tf.reduce_mean(v_flat))))))
        tf.summary.histogram('summarize_variables/%s' % (v.op.name), v_flat)

      train_op = slim.learning.create_train_op(total_loss,
          update_ops=update_ops,
          variables_to_train=variables_to_train, 
          clip_gradient_norm=0.0,
          gradient_multipliers=gradient_multipliers,
          summarize_gradients=True,
          optimizer=optimizer)

      saver = None
      if FLAGS.moving_average:
        saver = optimizer.swapping_saver()

    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   assign_fn(sess)
    #   #saver.restore(sess, 'log/mobilenet_v1/train/model.ckpt-9251')
    #   
    #   coord = tf.train.Coordinator()
    #   threads = tf.train.start_queue_runners(coord=coord)

    #   for i in xrange(10000):
    #     sess.run(train_op)

    #     if i % 10 == 0:
    #       example = sess.run(model.tensors)
    #       print('*' * 128)
    #       print(example['densecap_caption_embs'][0, :10])
    #       print(example['image_embs'][0, :10])
    #       #print('densecap_norm:', example['densecap_norm'][:5])
    #       print(example['loss_ratio'])
    #       print(example['densecap_loss_ratio'])

    #       print(example['img_densecap'])
    #       print(example['densecap_img'])

    #   coord.request_stop()
    #   coord.join(threads, stop_grace_period_secs=10)

    # exit(0)

  # Starts training.
  master = None
  if FLAGS.ps_hosts:
    master = server.target

  tf.logging.info('Start session.')
  slim.learning.train(train_op, 
      logdir=FLAGS.train_log_dir,
      graph=g,
      master=master,
      is_chief=(FLAGS.task_index == 0),
      number_of_steps=FLAGS.number_of_steps,
      log_every_n_steps=FLAGS.log_every_n_steps,
      save_interval_secs=FLAGS.save_interval_secs,
      save_summaries_secs=FLAGS.save_summaries_secs,
      session_config=default_session_config_proto(),
      init_fn=assign_fn,
      saver=saver)

if __name__ == '__main__':
  tf.app.run()
