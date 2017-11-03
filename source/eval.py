
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

flags.DEFINE_string('saved_ckpts_dir', '', 'The directory used to save the current best model.')
flags.DEFINE_string('eval_log_dir', '', 'The directory where the graph is saved.')
flags.DEFINE_integer('eval_interval_secs', 180, 'Seconds to sleep when there is no checkpoint.')
flags.DEFINE_integer('eval_min_global_steps', 200, 'Minimum global steps for evaluation.')

flags.DEFINE_string('ads_config', '', 'File path to ads config file.')
flags.DEFINE_string('feature_path', '', 'File path to image features.')
flags.DEFINE_string('vocab_path', '', 'Path to vocab file.')

flags.DEFINE_integer('num_positive_statements', 3, 'Maximum value of image size.')
flags.DEFINE_integer('max_image_size', 500, 'Maximum value of image size.')
flags.DEFINE_integer('max_string_len', 30, 'Maximum length of strings.')
flags.DEFINE_integer('max_val_examples', 1000, 'Maximum number of validation examples.')
flags.DEFINE_integer('max_vis_examples', 1000, 'Maximum number of examples to visualize.')

flags.DEFINE_boolean('continuous_evaluation', True, 
    'If true, continously evaluate the latest model. Otherwise, evalute the current best model only.')

slim = tf.contrib.slim
ckpt_path = None

topic_to_name = None
symbol_to_name = None

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

  if '\t' in lines[0]:
    for line in lines:
      word, index, freq = line.strip('\n').split('\t')
      vocab[int(index)] = word
      vocab_r[word] = (int(index), int(freq))
  else:
    for index, line in enumerate(lines):
      word = line.strip('\n')
      vocab[index + 1] = word
      vocab_r[word] = (index + 1, 0)
      
  return vocab, vocab_r

vocab, vocab_r = load_vocab(FLAGS.vocab_path)

def _tokenize(caption):
  caption = caption.replace('<UNK>', '')
  caption = nltk.word_tokenize(caption.lower())
  return caption


def visualize(global_step, vis_examples):
  html = ''
  html += '<html>'
  html += '<table border=1>'

  html += '<tr>'
  html += '<th>Image ID</th>'
  html += '<th>Image Data</th>'
  #if 'proposed_boxes' in vis_examples[0]:
  #  html += '<th>Refined boxes</th>'
  if 'scores_topic' in vis_examples[0]:
    html += '<th>Topics</th>'
  if 'scores_symbol' in vis_examples[0]:
    html += '<th>Symbols</th>'
  html += '<th>Statements</th>'
  html += '</tr>'

  for example_index, example in enumerate(vis_examples):
    html += '<tr id="%s">' % (example['image_id'])
    if example_index % 10 == 0:
      tf.logging.info('Vis on image %d/%d.', example_index, len(vis_examples))

    # Anchor for the image_id.
    html += '<td><a href="#%s">%s</a></td>' % (example['image_id'], example['image_id'])

    # Image with proposal annotations.
    image = np.copy(example['image'])
    for obj_index, obj in enumerate(example['objects']):
      if 'proposed_scores' not in example or obj_index < example['proposed_scores'].shape[1]:
        y1, x1, y2, x2 = obj['ymin'], obj['xmin'], obj['ymax'], obj['xmax']
        vis.image_draw_bounding_box(image, [x1, y1, x2, y2])
        vis.image_draw_text(image, [x1, y1], 
            '%.3lf' % (1.0 if 'proposed_scores' not in example 
              else float(example['proposed_scores'][0, obj_index])),
            color=(0, 0, 0))
    html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
        vis.image_uint8_to_base64(image, convert_to_bgr=True))

    #if 'proposed_boxes' in example:
    #  image = cv2.resize(example['image'], (500, 500))
    #  for i in xrange(example['num_detections'][0]):
    #    if example['proposed_scores'][0, i] > 0.01:
    #      y1, x1, y2, x2 = example['proposed_boxes'][0, i].tolist()
    #      vis.image_draw_bounding_box(image, [x1, y1, x2, y2])
    #      vis.image_draw_text(image, [x1, y1],
    #          '%.3lf' % (example['proposed_scores'][0, i]),
    #          color=(0, 0, 0))
    #  html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
    #      vis.image_uint8_to_base64(image, convert_to_bgr=True))

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

    if 'scores_symbol' in example:
      html += '<td>'
      scores = example['scores_symbol']
      for symbol_id in scores.argsort()[:20]:
        if symbol_id in example['symbol_ids']:
          html += '<p style="background-color:Yellow">[%.4lf] %s</p>' % (
              scores[symbol_id], symbol_to_name[symbol_id])
        else:
          html += '<p>[%.4lf] %s</p>' % (
              scores[symbol_id], symbol_to_name[symbol_id])
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
    result_dict: a [num_captions] tensor denoting similarities between image and
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
        eval_placeholders['object_num']: example['num_detections'],
        eval_placeholders['object_features']: example['proposed_features'],
        eval_placeholders['caption_strings']: example['caption_strings'],
        eval_placeholders['caption_lengths']: example['caption_lengths'],
        })

    if example_id % 50 == 0:
      tf.logging.info('On image %d of %d.', example_id, len(eval_data))

    if len(vis_examples) < FLAGS.max_vis_examples:
      vis_example = example
      vis_example['objects'] = example['objects']
      vis_example['scores'] = result['scores']
      if 'proposed_scores' in result:
        vis_example['proposed_scores'] = result['proposed_scores']
      if 'scores_topic' in result:
        vis_example['scores_topic'] = result['scores_topic']
      if 'scores_symbol' in result:
        vis_example['scores_symbol'] = result['scores_symbol']
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
    result_dict: a [num_captions] tensor denoting similarities between image and
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

  The function delete model checkpoint if it is not a better model.

  Args:
    global_step: a integer denoting current global step.
    model_path: current model path.
    model_metric: a float number denoting performance of current model.

  Returns:
    global_step_best: global step of the best model.
    model_metric_best: performance of the best model.
  """
  # Read the record file to get the previous best model.
  global_step_best = None
  model_metric_best = None

  record_file = os.path.join(FLAGS.saved_ckpts_dir, 'model_record.txt')
  if tf.gfile.Exists(record_file):
    with open(record_file, 'r') as fp:
      global_step_best, model_metric_best = fp.readline().strip().split('\t')
    global_step_best = int(global_step_best)
    model_metric_best = float(model_metric_best)

  # Modify record file if the model beats the previous best one.
  if model_metric_best is None or model_metric > model_metric_best:
    tf.logging.info(
        'Current model[%.4lf] is better than the previous best one[%.4lf].',
        model_metric, 0.0 if model_metric_best is None else model_metric_best)
    global_step_best = global_step
    model_metric_best = model_metric

    with open(record_file, 'w') as fp:
      fp.write('%d\t%.8lf' % (global_step, model_metric))

  # Remove the backup files since the model is not better.
  else:
    tf.logging.info(
        'Current model[%.4lf] is not better than previous best one[%.4lf].',
        model_metric, model_metric_best)
    for file_path in tf.gfile.Glob(model_path + '*'):
      dest_path = os.path.join(FLAGS.saved_ckpts_dir, os.path.split(file_path)[1])
      tf.gfile.Remove(dest_path)
      tf.logging.info('Delete %s.', dest_path)

  return global_step_best, model_metric_best


def evaluation_loop(sess, saver, writer, global_step,
    result_dict, eval_placeholders, eval_data):
  global ckpt_path

  while True:
    start = time.time()
    try:
    #if True:
      model_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)

      if model_path and ckpt_path != model_path:
        ckpt_path = model_path

        # Restore model and verify all variables are fine.
        saver.restore(sess, model_path)
        names = sess.run(result_dict['invalid_tensor_names'])
        assert len(names) == 0

        tf.logging.info('*' * 128)
        tf.logging.info('Load checkpoint %s.', model_path)

        # Evaluate model.
        step = sess.run(global_step)
        if step < FLAGS.eval_min_global_steps:
          tf.logging.info('Global step=%s < %s.', step, FLAGS.eval_min_global_steps)
          continue
        tf.logging.info('Start to evaluate model at global step=%s.', step)

        # Backup checkpoint.
        tf.logging.info('Copying files...')
        tf.gfile.MakeDirs(FLAGS.saved_ckpts_dir)

        for file_path in tf.gfile.Glob(model_path + '*'):
          dest_path = os.path.join(FLAGS.saved_ckpts_dir, os.path.split(file_path)[1])
          tf.gfile.Copy(file_path, dest_path, overwrite=True)
          tf.logging.info('Copy %s to %s.', file_path, dest_path)

        model_metric, vis_examples = evaluate_once(
            sess, writer, step, result_dict, eval_placeholders, eval_data)

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

    except Exception as ex:
      pass

    eval_secs = time.time() - start
    if FLAGS.eval_interval_secs - eval_secs > 0:
      tf.logging.info('Now sleep for %s secs.', 
          FLAGS.eval_interval_secs - eval_secs)
      time.sleep(FLAGS.eval_interval_secs - eval_secs)

def _get_eval_data(raw_features, split, image_level_feature=False):
  api = ads_api.AdsApi(FLAGS.ads_config, 
      invalid_items=['densecap_annotations'])
  meta_list = api.get_meta_list(split)

  global topic_to_name
  topic_to_name = api.get_topic_to_name()

  global symbol_to_name
  symbol_to_name = api.get_symbol_to_name()

  examples = []
  for meta_index, meta in enumerate(meta_list):

    if meta_index % 100 == 0:
      tf.logging.info('Load %d/%d.', meta_index, len(meta_list))

    if 'statements' in meta and 'negative_statements' in meta:
      image_id = meta['image_id']

      # Topic of the image.
      topic_id = meta.get('topic_id', 0)
      topic_name = meta.get('topic_name', 'unclear')

      # Ads detection results.
      raw_feature = raw_features[image_id]
      if image_level_feature:
        num_detections = 1
        proposed_features = np.expand_dims(raw_feature['image_emb'], 0)
      else:
        proposed_features = np.concatenate([
            np.expand_dims(raw_feature['image_emb'], 0), 
            raw_feature['object_emb_list']],
            axis=0)[:-1]
        num_detections = proposed_features.shape[0]

      # Ads QA statements.
      captions = meta['statements'] + meta['negative_statements']
      caption_strings = np.zeros((len(captions), FLAGS.max_string_len), dtype=np.int64)
      caption_lengths = np.zeros((len(captions)))
      for c_index, caption in enumerate(captions):
        caption = [vocab_r.get(w, (0, 0))[0] for w in _tokenize(caption)][:FLAGS.max_string_len]
        caption_strings[c_index, :len(caption)] = caption
        caption_lengths[c_index] = len(caption)

      example = {
          'image_id': image_id,
          'num_detections': num_detections,
          'proposed_features': proposed_features,
          'objects': meta['objects'],
          'topic_id': topic_id,
          'topic': topic_name,
          'symbol_ids': meta.get('symbol_ids', []),
          'symbol_names': meta.get('symbol_names', []),
          'caption_lengths': caption_lengths,
          'caption_strings': caption_strings
      }

      # Load image data.
      if len(examples) < FLAGS.max_vis_examples:
        image = vis.image_load(meta['file_path'], True)
        height, width, _ = image.shape
        if height > FLAGS.max_image_size or width > FLAGS.max_image_size:
          image = cv2.resize(image, (FLAGS.max_image_size, FLAGS.max_image_size))
        example['image'] = image

      examples.append(example)

      if len(examples) >= FLAGS.max_val_examples:
        break

  tf.logging.info('Loaded %s %s examples.', len(examples), split)
  return examples


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  model_proto = ads_emb_model_pb2.AdsEmbModel()
  with open(FLAGS.model_config, 'r') as fp:
    text_format.Merge(fp.read(), model_proto)

  g = tf.Graph()
  with g.as_default():
    object_num_placeholder = tf.placeholder(
        dtype=tf.int64, shape=[])
    object_features_placeholder = tf.placeholder(dtype=tf.float32, 
        shape=[None, model_proto.examples_reader.feature_dimentions])

    caption_strings_placeholder = tf.placeholder(
        dtype=tf.int64, shape=[None, FLAGS.max_string_len])
    caption_lengths_placeholder = tf.placeholder(
        dtype=tf.int64, shape=[None])

    eval_placeholders = {
      'object_num': object_num_placeholder,
      'object_features': object_features_placeholder,
      'caption_strings': caption_strings_placeholder,
      'caption_lengths': caption_lengths_placeholder,
    }
    # Create ads embedding model.
    model = ads_emb_model.AdsEmbModel(model_proto)

    # Get image embedding vector.
    image_embs = model.build_image_model_from_feature(
        tf.expand_dims(object_num_placeholder, 0),
        tf.expand_dims(object_features_placeholder, 0),
        is_training=False)

    image_embs = tf.nn.l2_normalize(image_embs, 1)
    image_emb = image_embs[0, :]

    # Get caption embedding vectors.
    caption_embs, assign_fn_cap = model.build_caption_model(
        caption_lengths_placeholder,
        caption_strings_placeholder, 
        is_training=False)
    caption_embs = tf.nn.l2_normalize(caption_embs, 1)
    scores = ads_emb_model.distance_fn(image_emb, caption_embs)

    embedding_weights = model.caption_encoder.embedding_weights

    # Get topic embedding vectors.
    scores_topic = None
    if model.topic_encoder is not None:

      topic_embs = model.topic_encoder.build_weights(
          vocab_size=model_proto.topic_encoder.bow_encoder.vocab_size,
          embedding_size=model_proto.topic_encoder.bow_encoder.embedding_size) 
      topic_embs = tf.nn.l2_normalize(topic_embs, 1)
      scores_topic = ads_emb_model.distance_fn(image_emb, topic_embs)

    # Get symbol embedding vectors.
    scores_symbol = None
    if model.symbol_encoder is not None:

      symbol_embs = model.symbol_encoder.build_weights(
          vocab_size=model_proto.symbol_encoder.bow_encoder.vocab_size,
          embedding_size=model_proto.symbol_encoder.bow_encoder.embedding_size) 
      symbol_embs = tf.nn.l2_normalize(symbol_embs, 1)
      scores_symbol = ads_emb_model.distance_fn(image_emb, symbol_embs)

    # Get densecap word embedding vectors.
    scores_densecap = None
    if model.densecap_encoder is not None:
      pass

    global_step = slim.get_or_create_global_step()

    # Variables to restore, ignore variables in the pre-trained model.
    variables_to_restore = tf.global_variables()
    variables_to_restore = filter(
        lambda x: 'MobilenetV1' not in x.op.name, variables_to_restore)
    variables_to_restore = filter(
        lambda x: 'InceptionV4' not in x.op.name, variables_to_restore)
    variables_to_restore = filter(
        lambda x: 'BoxPredictor' not in x.op.name, variables_to_restore)
    #if not FLAGS.tune_caption_model:
    #  variables_to_train = filter(
    #      lambda x: 'caption_encoder' not in x.op.name, variables_to_restore)
    invalid_tensor_names = tf.report_uninitialized_variables()
    saver = tf.train.Saver(variables_to_restore)

  # Build result dict for both evaluation and visualization.
  result_dict = {
    'scores': scores,
    'image_emb': image_emb,
    'caption_embs': caption_embs,
    'embedding_weights': embedding_weights,
    'invalid_tensor_names': invalid_tensor_names,
  }

  if scores_topic is not None:
    result_dict['scores_topic'] = scores_topic
  if scores_symbol is not None:
    result_dict['scores_symbol'] = scores_symbol
  result_dict.update(model.tensors)

  def assign_fn(sess):
    assign_fn_cap(sess)

  # Load eval data.
  raw_features = np.load(FLAGS.feature_path).item()
  image_level_feature = model_proto.examples_reader.image_level_feature
  tf.logging.info("Use image level feature: %s", image_level_feature)
  if FLAGS.continuous_evaluation:
    eval_data = _get_eval_data(raw_features, 'valid', image_level_feature)
  else:
    eval_data = _get_eval_data(raw_features, 'test', image_level_feature)

  # Start session.
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

