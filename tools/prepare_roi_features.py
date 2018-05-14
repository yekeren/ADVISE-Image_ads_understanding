from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import json
import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from slim.nets import nets_factory

from utils.train_utils import default_session_config

flags.DEFINE_string('bounding_box_json', '', 
                    'Path to the json annotation file.')

flags.DEFINE_string('feature_extractor_name', 'inception_v4', 
                    'The name of the feature extractor.')

flags.DEFINE_string('feature_extractor_scope', 'InceptionV4', 
                    'The variable scope of the feature extractor.')

flags.DEFINE_string('feature_extractor_endpoint', 'PreLogitsFlatten', 
                    'The endpoint of the feature extractor.')

flags.DEFINE_string('feature_extractor_checkpoint', 'zoo/inception_v4.ckpt', 
                    'The path to the checkpoint file.')

flags.DEFINE_string('image_dir', 'data/train_images/', 
                    'Path to the ads image directory.')

flags.DEFINE_string('output_feature_path', 'output/roi_features.npy', 
                    'Path to the output npy file.')

flags.DEFINE_integer('max_number_of_regions', 10,
                     'Maximum number of regions.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _load_annots(filename):
  """Loads annotation file.

  Args:
    filename: path to the annotation json file.

  Returns:
    examples: a dict mapping from img_id to annotation.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())
  return data


def image_crop_and_resize(image, bbox, crop_size):
  """Crops roi from an image and resizes it.

  Args:
    image: the image.
    bbox: bounding box information.
    crop_size: the expected output size of the roi image.

  Returns:
    a [crop_size, crop_size, 3] roi image.
  """
  height, width, _ = image.shape

  x1, y1, x2, y2 = bbox
  x1 = int(x1 * width)
  y1 = int(y1 * height)
  x2 = int(x2 * width)
  y2 = int(y2 * height)

  return cv2.resize(image[y1: y2, x1: x2, :], crop_size)


def main(_):
  logging.set_verbosity(logging.INFO)

  examples = _load_annots(FLAGS.bounding_box_json)
  logging.info('Loaded %s examples.', len(examples))

  # Create computational graph.
  g = tf.Graph()
  with g.as_default():
    # Create model.
    net_fn = nets_factory.get_network_fn(
        name=FLAGS.feature_extractor_name, num_classes=1001)
    default_image_size = getattr(net_fn, 'default_image_size', 224)

    images = tf.placeholder(
        shape=(None, default_image_size, default_image_size, 3), 
        dtype=tf.float32)

    _, end_points = net_fn(images)
    output_tensor = end_points[FLAGS.feature_extractor_endpoint]

    init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.feature_extractor_checkpoint,
        slim.get_model_variables(FLAGS.feature_extractor_scope))
    uninitialized_variable_names = tf.report_uninitialized_variables()

  # Start session.
  results = {}
  with tf.Session(graph=g, config=default_session_config()) as sess:
    init_fn(sess)
    assert len(sess.run(uninitialized_variable_names)) == 0

    for index, (image_id, example) in enumerate(examples.iteritems()):
      if index % 50 == 0:
        logging.info('On image %i/%i', index, len(examples))

      # Load image, preprocess. TODO: now only works for inception family.
      filename = "{}/{}".format(FLAGS.image_dir, image_id)
      bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
      rgb = bgr[:, :, ::-1]
      rgb = rgb.astype(np.float32) * 2.0 / 255.0 - 1.0

      # Batch operation.
      regions_list = []
      for region in example['regions'][:FLAGS.max_number_of_regions]:
        roi = image_crop_and_resize(rgb, 
            bbox=(
              region['bbox']['xmin'], 
              region['bbox']['ymin'], 
              region['bbox']['xmax'], 
              region['bbox']['ymax']), 
            crop_size=(default_image_size, default_image_size))
        regions_list.append(roi)
      batch = np.stack(regions_list, axis=0)
      
      # Save results.
      features = sess.run(output_tensor, feed_dict={images: batch})
      results[image_id] = features

  # Write results.
  assert len(results) == len(examples)
  with open(FLAGS.output_feature_path, 'wb') as fp:
    np.save(fp, results)
  logging.info('Exported features for %i images.', len(results))

  logging.info('Done')

if __name__ == '__main__':
  app.run()
