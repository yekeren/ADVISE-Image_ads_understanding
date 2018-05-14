
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import json
import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util


flags = tf.app.flags

flags.DEFINE_string('model_proto', '', 
                    'Path to the pbtxt file describing the detection model.')

flags.DEFINE_string('label_map', '', 
                    'Path to the object detection label map file.')

flags.DEFINE_string('checkpoint', '', 
                    'Path to the detection checkpoint file.')

flags.DEFINE_string('action_reason_annot_path', '', 
                    'Path to the ads annotation file.')

flags.DEFINE_string('image_dir', '', 
                    'Path to the ads image directory.')

flags.DEFINE_string('output_json', '', 
                    'Path to the json output file.')

flags.DEFINE_integer('max_image_size', 600,
                     'The maximum image size for detection purpose.')

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


def load_model_proto(filename):
  """Loads object detection model proto.

  Args:
    filename: path to the model proto file.

  Returns:
    model_pb2.DetectionModel.
  """
  model_proto = pipeline_pb2.TrainEvalPipelineConfig()
  with open(filename, 'r') as fp:
    text_format.Merge(fp.read(), model_proto)
  return model_proto.model


def load_label_map(filename):
  """Loads label map file.

  Args:
    filename: path to the label map file.

  Reutrns:
    a dict mapping from id to text name.
  """
  label_map = label_map_util.load_labelmap(filename)
  id2name = {}
  for item_id, item in enumerate(label_map.item):
    if item.HasField('display_name'):
      id2name[item.id - 1] = item.display_name
    else:
      id2name[item.id - 1] = item.name
  return id2name


def load_ads_id_list(filename):
  """Loads ads list.

  Args:
    filename: path to the ads list file.

  Returns:
    ads_id_list: ad id.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())
  return data.keys()


def main(_):
  logging.set_verbosity(tf.logging.INFO)

  # Load object detection label map.
  id2name = load_label_map(FLAGS.label_map)

  g = tf.Graph()
  with g.as_default():
    image_placeholder = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
    images = tf.expand_dims(image_placeholder, 0)

    # Get detections from detection model.
    model_proto = load_model_proto(FLAGS.model_proto)
    model = model_builder.build(model_proto, is_training=False)
    predictions = model.predict(
        model.preprocess(tf.cast(images, tf.float32)))
    detections = model.postprocess(predictions)

    init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.checkpoint, tf.global_variables())
    uninitialized_variable_names = tf.report_uninitialized_variables()

  # Start the tf session.
  examples = {}
  with tf.Session(graph=g, config=default_session_config_proto()) as sess:
    init_fn(sess)

    assert len(sess.run(uninitialized_variable_names)) == 0

    ads_id_list = load_ads_id_list(FLAGS.action_reason_annot_path)

    for index, image_id in enumerate(ads_id_list):
      if index % 200 == 0:
        logging.info('On image %i/%i', index, len(ads_id_list))

      # Load image.
      filename = "{}/{}".format(FLAGS.image_dir, image_id)
      bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
      rgb = bgr[:, :, ::-1]

      height, width, _ = rgb.shape
      if height > FLAGS.max_image_size or width > FLAGS.max_image_size:
        rgb = cv2.resize(rgb, (FLAGS.max_image_size, FLAGS.max_image_size))
      
      # Dump detection results.
      results = sess.run(detections, feed_dict={image_placeholder: rgb})

      regions = []
      for i in xrange(results['num_detections'][0]):
        class_id = int(results['detection_classes'][0, i])
        class_name = id2name.get(class_id, 'unclear')
        class_score = round(float(results['detection_scores'][0, i]), 3)
        y1, x1, y2, x2 = [round(float(x), 3) for x in results['detection_boxes'][0, i]]
        regions.append({
            'id': class_id,
            'name': class_name,
            'score': class_score,
            'bbox': { 'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2 }
            })
      examples[image_id] = { 
        'image_id': image_id, 
        'regions': regions 
      }

  # Write to output json file.
  with open(FLAGS.output_json, 'w') as fp:
    fp.write(json.dumps(examples))

  logging.info('Done')

if __name__ == '__main__':
  app.run()
