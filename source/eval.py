
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import cv2
import time
import nltk

from google.protobuf import text_format
from protos import ads_emb_model_pb2

import numpy as np
import tensorflow as tf

import ads_emb_model

from utils import vis
from utils import ads_dataset_api

from train import FLAGS
from train import default_session_config_proto

flags = tf.app.flags

flags.DEFINE_string('saved_ckpts_dir', '', 'The directory used to save the current best model.')
flags.DEFINE_string('eval_log_dir', '', 'The directory where the graph is saved.')
flags.DEFINE_integer('eval_interval_secs', 180, 'Seconds to sleep when there is no checkpoint.')
flags.DEFINE_integer('eval_min_global_steps', 400, 'Minimum global steps for evaluation.')

flags.DEFINE_string('vocab_path', '', 'Path to vocab file.')
flags.DEFINE_string('image_dir', '', 'Directory to ads dataset.')
flags.DEFINE_string('topic_list_path', '', 'File path to ads topic list.')
flags.DEFINE_string('topic_annot_path', '', 'File path to ads topic annotations.')
flags.DEFINE_string('qa_action_annot_path', '', 'File path to ads action annotations.')
flags.DEFINE_string('qa_reason_annot_path', '', 'File path to ads reason annotations.')
flags.DEFINE_string('qa_action_reason_annot_path', '', 'File path to ads action reason annotations.')
flags.DEFINE_integer('num_positive_statements', 3, 'Number of positive statements.')
flags.DEFINE_integer('num_negative_statements', 17, 'Number of negative statements.')
flags.DEFINE_integer('max_string_len', 30, 'Maximum length of strings.')

flags.DEFINE_integer('max_val_examples', 1000, 'Maximum number of examples to validate.')
flags.DEFINE_integer('max_vis_examples', 1000, 'Maximum number of examples to visualize.')

flags.DEFINE_boolean('continuous_evaluation', True, 
    'If true, continously evaluate the latest model. Otherwise, evalute the current best model only.')

slim = tf.contrib.slim
ckpt_path = None

topic_to_name = None

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

vocab, vocab_r = load_vocab(FLAGS.vocab_path)

def _tokenize(caption):
  return nltk.word_tokenize(caption.lower())


def visualize(global_step, vis_examples):
  html = ''
  html += '<html>'
  html += '<table border=1>'

  html += '<tr>'
  html += '<th>Image ID</th>'
  html += '<th>Image Data</th>'
  if 'scores_topic' in vis_examples[0]:
    html += '<th>Topics</th>'
  html += '<th>Statements</th>'
  html += '</tr>'

  for example in vis_examples:
    html += '<tr id="%s">' % (example['image_id'])

    # Anchor for the image_id.
    html += '<td><a href="#%s">%s</a></td>' % (example['image_id'], example['image_id'])

    # Image with proposal annotations.
    image = cv2.resize(example['image'], (500, 500))
    for i in xrange(example['num_detections'][0]):
      y1, x1, y2, x2 = example['detection_boxes'][0, i].tolist()
      vis.image_draw_bounding_box(image, [x1, y1, x2, y2])
      vis.image_draw_text(image, [x1, y1],
          '%.3lf' % (example['detection_scores'][0, i]),
          color=(0, 0, 0))
    html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
        vis.image_uint8_to_base64(image, convert_to_bgr=True))

    # Topic predictions.
    if 'scores_topic' in example:
      html += '<td>'
      scores = example['scores_topic']
      for topic_id in scores.argsort()[:20]:
        if topic_id == example['topic_id']:
          html += '<p style="background-color:Yellow">[%.4lf] %s</p>' % (
              scores[topic_id], topic_to_name[topic_id])
        else:
          html += '<p>[%.4lf] %s</p>' % (
              scores[topic_id], topic_to_name[topic_id])
      html += '</td>'

    # Statements with similarity scores.
    html += '<td>'
    statements = []
    scores = example['scores']
    caption_strings = example['caption_strings']
    caption_lengths = example['caption_lengths']

    for i in xrange(caption_lengths.shape[0]):
      caption_string = caption_strings[i]
      caption_length = int(caption_lengths[i])
      statement = [vocab[wid] for wid in caption_string[:caption_length]]
      statements.append((' '.join(statement)))

    for index in scores.argsort():
      if index >= FLAGS.num_positive_statements:
        html += '<p>[%.4lf] %s</p>' % (scores[index], statements[index])
      else:
        html += '<p style="background-color:Yellow">[%.4lf] %s</p>' % (
            scores[index], statements[index])
    html += '</td>'

    html += '</tr>'
  html += '</table>'
  html += '</html>'

  vis_path = os.path.join(FLAGS.saved_ckpts_dir, 'vis.html')
  with open(vis_path, 'w') as fp:
    fp.write(html)
  tf.logging.info('Visualize model at global step %s.', global_step)


def evaluate_once(sess, writer, global_step, 
    result_dict, eval_placeholders, eval_data):
  """Evaluate the model on the evaluation dataset.

  Args:
    sess: a tf.Session instance used to execute the graph.
    writer: a tf.summary.FileWriter instance used to write summaries.
    global_step: a integer number used to indecate steps for the summaries.
    eval_scores: a [num_captions] tensor denoting similarities between image and
      each caption.
    eval_placeholders: a dictionary mapping from name to input placeholders.
    eval_data: a dictionary mapping from name to numpy data.

  Returns:
    model_metric: a float number using to pick the best model.
    vis_examples: a list of examples for visualization.
  """
  # Multi-choice metrics.
  recalls = {3: [], 5: [], 10:[]}
  recalls_cat = {}

  minrank = []
  minrank_cat = {}

  # Topic-prediction metrics.
  precision = []
  precision_cat = {}

  vis_examples = []
  for example_id, example in enumerate(eval_data):

    result = sess.run(result_dict, feed_dict={
        eval_placeholders['image']: example['image'],
        eval_placeholders['caption_strings']: example['caption_strings'],
        eval_placeholders['caption_lengths']: example['caption_lengths'],
        })
    if example_id % 50 == 0:
      tf.logging.info('On image %d of %d.', example_id, len(eval_data))

    if len(vis_examples) < FLAGS.max_vis_examples:
      vis_example = example
      vis_example.update(result)
      vis_examples.append(vis_example)

    # Multi-choice task.
    scores = result['scores']
    for at_k in recalls.keys():
      recall = (scores.argsort()[:at_k] < FLAGS.num_positive_statements).sum()
      recall = float(1.0 * recall)
      recalls[at_k].append(recall)
      recalls_cat.setdefault(example['topic'], {3: [], 5: [], 10:[]})[at_k].append(recall)

    rank = np.where(scores.argsort() < 3)[0]
    minrank.append(float(rank.min()))
    minrank_cat.setdefault(example['topic'], []).append(float(rank.min()))

    # Topic-prediction task.
    if 'scores_topic' in result:
      correct = 0.0
      if int(result['scores_topic'].argsort()[0]) == example['topic_id']:
        correct = 1.0
      precision.append(correct)
      precision_cat.setdefault(example['topic'], []).append(correct)

  # Use recall@3 as the metric to pick the best model.
  model_metric = 1.0 * sum(recalls[3]) / len(recalls[3])

  # Write summary.
  if writer is not None:
    summary = tf.Summary()
    # General recall.
    for at_k in recalls.keys():
      recall = 1.0 * sum(recalls[at_k]) / len(recalls[at_k])
      tag = 'metrics/recall@%d' % (at_k)
      summary.value.add(tag=tag, simple_value=recall)
      tf.logging.info('%s: %.4lf', tag, recall)

    # General minrank.
    minrank = 1.0 * sum(minrank) / len(minrank)
    tag = 'metrics/minrank'
    summary.value.add(tag=tag, simple_value=minrank)
    tf.logging.info('%s: %.4lf', tag, minrank)

    # General precision.
    if len(precision):
      precision = 1.0 * sum(precision) / len(precision)
      tag = 'metrics/precision[topics]'
      summary.value.add(tag=tag, simple_value=precision)
      tf.logging.info('%s: %.4lf', tag, precision)

    # Recall for different categories.
    recalls_avg = {3: [], 5: [], 10:[]}
    for topic, recalls in recalls_cat.iteritems():
      for at_k in recalls.keys():
        recall = 1.0 * sum(recalls[at_k]) / len(recalls[at_k])
        tag = 'recall@%d/%s' % (at_k, topic)
        summary.value.add(tag=tag, simple_value=recall)
        # tf.logging.info('%s: %.4lf', tag, recall)
        recalls_avg[at_k].append(recall)

    # Minrank for different categories.
    minrank_avg = []
    for topic, minrank in minrank_cat.iteritems():
      minrank = 1.0 * sum(minrank) / len(minrank)
      tag = 'minrank/%s' % (topic)
      summary.value.add(tag=tag, simple_value=minrank)
      # tf.logging.info('%s: %.4lf', tag, minrank)
      minrank_avg.append(minrank)

    # Precision for different categories.
    precision_avg = []
    for topic, precision in precision_cat.iteritems():
      if len(precision):
        precision = 1.0 * sum(precision) / len(precision)
        tag = 'precision[topics]/%s' % (topic)
        summary.value.add(tag=tag, simple_value=precision)
        #tf.logging.info('%s: %.4lf', tag, precision)
        precision_avg.append(precision)

    # Averaged recall.
    for at_k in recalls_avg.keys():
      recall = 1.0 * sum(recalls_avg[at_k]) / len(recalls_avg[at_k])
      tag = 'metrics/recall@%d_avg' % (at_k)
      summary.value.add(tag=tag, simple_value=recall)
      tf.logging.info('%s: %.4lf', tag, recall)

    # Averaged minrank.
    minrank = 1.0 * sum(minrank_avg) / len(minrank_avg)
    tag = 'metrics/minrank_avg'
    summary.value.add(tag=tag, simple_value=minrank)
    tf.logging.info('%s: %.4lf', tag, minrank)

    # Averaged precision.
    if len(precision_avg):
      precision = 1.0 * sum(precision_avg) / len(precision_avg)
      tag = 'metrics/precision_avg[topics]'
      summary.value.add(tag=tag, simple_value=precision)
      tf.logging.info('%s: %.4lf', tag, precision)

    writer.add_summary(summary, global_step=global_step)
    writer.flush()

  # Write eval results.
  else:
    result_file = os.path.join(FLAGS.saved_ckpts_dir, 'model_eval.csv')
    with open(result_file, 'w') as fp:
      for at_k in sorted(recalls.keys()):
        fp.write(',recall@%d' % (at_k))
      fp.write(',minrank')
      fp.write('\n')

      fp.write('general')
      for at_k in sorted(recalls.keys()):
        recall = 1.0 * sum(recalls[at_k]) / len(recalls[at_k])
        fp.write(',%.4lf' % (recall))
      minrank = 1.0 * sum(minrank) / len(minrank)
      fp.write(',%.4lf' % (minrank))
      fp.write('\n')

      recalls_avg = {3: [], 5: [], 10:[]}
      minrank_avg = []
      for topic in sorted(recalls_cat.keys()):
        fp.write('%s' % (topic))
        recalls = recalls_cat[topic]
        minrank = minrank_cat[topic]
        for at_k in sorted(recalls.keys()):
          recall = 1.0 * sum(recalls[at_k]) / len(recalls[at_k])
          recalls_avg[at_k].append(recall)
          fp.write(',%.4lf' % (recall))
        minrank = 1.0 * sum(minrank) / len(minrank)
        fp.write(',%.4lf' % (minrank))
        minrank_avg.append(minrank)
        fp.write('\n')

      fp.write('averaged')
      for at_k in sorted(recalls_avg.keys()):
        recall = 1.0 * sum(recalls_avg[at_k]) / len(recalls_avg[at_k])
        fp.write(',%.4lf' % (recall))
      minrank = 1.0 * sum(minrank_avg) / len(minrank_avg)
      fp.write(',%.4lf' % (minrank))
      fp.write('\n')

  return model_metric, vis_examples

def evaluate_best_model(sess, saver, writer, global_step, 
    result_dict, eval_placeholders, eval_data):
  """Evaluate the model on the evaluation dataset.

  Args:
    sess: a tf.Session instance used to execute the graph.
    global_step: a integer number used to indecate steps for the summaries.
    eval_scores: a [num_captions] tensor denoting similarities between image and
      each caption.
    eval_placeholders: a dictionary mapping from name to input placeholders.
    eval_data: a dictionary mapping from name to numpy data.

  Returns:
    model_metric: a float number using to pick the best model.
  """
  file_path_list = []
  for file_path in tf.gfile.Glob(os.path.join(FLAGS.saved_ckpts_dir, 'model.ckpt-*.meta')):
    file_path_list.append(file_path)

  if len(file_path_list) == 0:
    raise ValueError('No checkpoint was found in %s.' % (FLAGS.saved_ckpts_dir))

  ckpt_fn = lambda x: int(re.findall('ckpt-(\d+).meta', x)[0])
  model_path = sorted(file_path_list, lambda x, y: -cmp(ckpt_fn(x), ckpt_fn(y)))[0]
  model_path = model_path[:-5]

  saver.restore(sess, model_path)
  tf.logging.info('*' * 128)
  tf.logging.info('Load checkpoint %s.', model_path)

  step = sess.run(global_step)
  tf.logging.info('Global step=%s.', step)
  model_metric, _ = evaluate_once(
      sess, writer, step, result_dict, eval_placeholders, eval_data)


def save_model_if_it_is_better(global_step, model_path, model_metric):
  """Save model if it is better than previous best model.

  Args:
    global_step: a integer denoting current global step.
    model_path: current model path.
    model_metric: a float number denoting performance of current model.

  Returns:
    global_step_best: global step of the best model.
    model_metric_best: performance of the best model.
  """
  tf.gfile.MakeDirs(FLAGS.saved_ckpts_dir)

  # Read the record file to get the previous best model.
  global_step_best = None
  model_metric_best = None

  record_file = os.path.join(FLAGS.saved_ckpts_dir, 'model_record.txt')
  if tf.gfile.Exists(record_file):
    with open(record_file, 'r') as fp:
      global_step_best, model_metric_best = fp.readline().strip().split('\t')
    global_step_best = int(global_step_best)
    model_metric_best = float(model_metric_best)

  # Save model if it beats the previous best one.
  if model_metric_best is None or model_metric > model_metric_best + 0.0001:
    tf.logging.info('Current model is better than previous best one.')
    global_step_best = global_step
    model_metric_best = model_metric

    with open(record_file, 'w') as fp:
      fp.write('%d\t%.8lf' % (global_step, model_metric))

    for file_path in tf.gfile.Glob(model_path + '*'):
      dest_path = os.path.join(FLAGS.saved_ckpts_dir, os.path.split(file_path)[1])
      tf.gfile.Copy(file_path, dest_path, overwrite=True)
      tf.logging.info('Copy %s to %s', file_path, dest_path)

  else:
    tf.logging.info('Current model[%.4lf] is not better than previous best one[%.4lf].',
        model_metric, model_metric_best)

  return global_step_best, model_metric_best


def evaluation_loop(sess, saver, writer, global_step,
    eval_scores, eval_placeholders, eval_data):
  global ckpt_path

  while True:
    model_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)

    start = time.time()
    if model_path and ckpt_path != model_path:
      ckpt_path = model_path
      saver.restore(sess, model_path)
      tf.logging.info('*' * 128)
      tf.logging.info('Load checkpoint %s.', model_path)

      # Evaluate.
      step = sess.run(global_step)
      if step < FLAGS.eval_min_global_steps:
        tf.logging.info('Global step=%s < %s.', step, FLAGS.eval_min_global_steps)
        continue

      tf.logging.info('Global step=%s.', step)
      model_metric, vis_examples = evaluate_once(
          sess, writer, step, eval_scores, eval_placeholders, eval_data)

      # Pick the best model.
      step_best, _ = save_model_if_it_is_better(
          step, model_path, model_metric)

      if step_best == step:
        # We improved the model, record it.
        summary = tf.Summary()
        summary.value.add(
            tag='metrics/model_metric', 
            simple_value=model_metric)
        writer.add_summary(summary, global_step=step)
        writer.flush()

        # Visualize the current best model.
      visualize(step, vis_examples)

      tf.logging.info('Finish evaluation.')
      if step >= FLAGS.number_of_steps:
        tf.logging.info('Break evaluation_loop.')
        break
    else:
      tf.logging.info('No new checkpoint was found in %s.', FLAGS.train_log_dir)

    eval_secs = time.time() - start
    if FLAGS.eval_interval_secs - eval_secs > 0:
      tf.logging.info('Now sleep for %s secs.', 
          FLAGS.eval_interval_secs - eval_secs)
      time.sleep(FLAGS.eval_interval_secs - eval_secs)

def _get_eval_data(data_type, image_level_feature=True):
  api = ads_dataset_api.AdsDatasetApi()
  api.init(
      images_dir=FLAGS.image_dir, 
      topic_list_file=FLAGS.topic_list_path, 
      topic_annot_file=FLAGS.topic_annot_path,
      qa_action_annot_file=FLAGS.qa_action_annot_path,
      qa_reason_annot_file=FLAGS.qa_reason_annot_path,
      qa_action_reason_annot_file=FLAGS.qa_action_reason_annot_path,
      qa_action_reason_padding=FLAGS.num_positive_statements)
  api.sample_negative_action_reason_captions(FLAGS.num_negative_statements)

  global topic_to_name
  topic_to_name = api.topic_to_name

  if 'testing' == data_type:
    filter_fn = ads_dataset_api.is_testing_example
  elif 'validation' == data_type:
    filter_fn = ads_dataset_api.is_validation_example

  examples = []
  meta_list = api.get_meta_list()
  for meta in meta_list:
    image_id = meta['image_id']
    if filter_fn(image_id) and 'action_reason_captions' in meta:
      image = vis.image_load(meta['filename'], True)
      topic_id = meta.get('topic_id', 0)
      topic_name = meta.get('topic_name', 'unclear')

      captions = meta['action_reason_captions'] + meta['action_reason_captions_neg']
      caption_strings = np.zeros((len(captions), FLAGS.max_string_len), dtype=np.int64)
      caption_lengths = np.zeros((len(captions)))

      for c_index, caption in enumerate(captions):
        caption = [vocab_r.get(w, (0, 0))[0] for w in
        _tokenize(caption)][:FLAGS.max_string_len]
        caption_strings[c_index, :len(caption)] = caption
        caption_lengths[c_index] = len(caption)

      examples.append({
          'image_id': image_id,
          'image': image,
          'topic_id': topic_id,
          'topic': topic_name,
          'caption_lengths': caption_lengths,
          'caption_strings': caption_strings,
          })

      if len(examples) >= FLAGS.max_val_examples:
        break

  tf.logging.info('Loaded %s %s examples.', len(examples), data_type)
  return examples


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  g = tf.Graph()
  with g.as_default():
    image_placeholder = tf.placeholder(
        dtype=tf.uint8, shape=[None, None, 3])
    caption_strings_placeholder = tf.placeholder(
        dtype=tf.int64, shape=[None, FLAGS.max_string_len])
    caption_lengths_placeholder = tf.placeholder(
        dtype=tf.int64, shape=[None])

    eval_placeholders = {
      'image': image_placeholder,
      'caption_strings': caption_strings_placeholder,
      'caption_lengths': caption_lengths_placeholder
    }
    # Create ads embedding model.
    model_proto = ads_emb_model_pb2.AdsEmbModel()
    with open(FLAGS.model_config, 'r') as fp:
      text_format.Merge(fp.read(), model_proto)
    model = ads_emb_model.AdsEmbModel(model_proto)

    # Get image embedding vector.
    image_embs, assign_fn_img = model.build_image_model(
        tf.expand_dims(image_placeholder, 0), is_training=False)
    image_emb = ads_emb_model.unit_norm(image_embs)[0, :]

    # Get caption embedding vectors.
    caption_embs, assign_fn_cap = model.build_caption_model(
        caption_lengths_placeholder,
        caption_strings_placeholder, 
        is_training=False)
    caption_embs = ads_emb_model.unit_norm(caption_embs)

    scores = 1 - tf.reduce_sum(
        tf.multiply(image_emb, caption_embs), axis=1)

    # Get topic embedding vectors.
    scores_topic = None
    if model.topic_embedder is not None:
      topics_const = tf.range(
          model_proto.topic_embedder.bow_embedder.vocab_size, 
          dtype=tf.int64)
      topic_embs, assign_fn_top = model.build_topic_model(
          topics_const, is_training=False)
      topic_embs = ads_emb_model.unit_norm(topic_embs)
      scores_topic = 1 - tf.reduce_sum(
          tf.multiply(image_emb, topic_embs), axis=1)

    global_step = slim.get_or_create_global_step()

    # Variables to restore, ignore variables in the pre-trained model.
    variables_to_restore = tf.global_variables()
    variables_to_restore = filter(
        lambda x: 'MobilenetV1' not in x.op.name, variables_to_restore)
    variables_to_restore = filter(
        lambda x: 'InceptionV4' not in x.op.name, variables_to_restore)
    variables_to_restore = filter(
        lambda x: 'BoxPredictor' not in x.op.name, variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)

  if FLAGS.continuous_evaluation:
    eval_data = _get_eval_data('validation')
  else:
    eval_data = _get_eval_data('testing')

  # Build result dict for both evaluation and visualization.
  result_dict = {
    'scores': scores,
    'image_emb': image_emb,
    'caption_embs': caption_embs,
  }
  if scores_topic is not None:
    result_dict['scores_topic'] = scores_topic
  result_dict.update(model.tensors)

  def assign_fn(sess):
    assign_fn_img(sess)
    assign_fn_cap(sess)

  with tf.Session(graph=g, config=default_session_config_proto()) as sess:
    assign_fn(sess)

    writer = tf.summary.FileWriter(FLAGS.eval_log_dir, g)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if FLAGS.continuous_evaluation:
      evaluation_loop(sess, saver, writer, global_step, 
          result_dict, eval_placeholders, eval_data)
    else:
      evaluate_best_model(sess, saver, None, global_step, 
          result_dict, eval_placeholders, eval_data)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    writer.close()


if __name__ == '__main__':
  tf.app.run()

