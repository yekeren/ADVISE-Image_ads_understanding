from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

import cv2
import numpy as np
from nets import mobilenet_v1
from utils import vis
from utils.ads_dataset_api import AdsDatasetApi

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 64, 'Maximum number of examples in a batch.')
flags.DEFINE_string('model_name', 'MobilenetV1', 'Name scope of the model.')
flags.DEFINE_string('checkpoint_path', '', 'Path to the pre-trained ckpt file.')
flags.DEFINE_string('images_dir', '', 'Directory to the ads images.')
flags.DEFINE_string('entity_annot_file', '', 'Path to the entity annotations.')
flags.DEFINE_float('score_threshold', 0.5, 'Threshold for filtering out low confident scores.')
flags.DEFINE_string('output_path', '', 'Path to the output file.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _visualize(batch_data):
  for i, image in enumerate(batch_data):
    image = vis.image_float32_to_uint8(image)
    vis.image_save('%s.jpg' % (i), image, convert_to_bgr=True)


def main(_):
  logging.basicConfig(level=logging.DEBUG)

  batch_size = FLAGS.batch_size
  image_size = mobilenet_v1.mobilenet_v1.default_image_size

  # Create computational graph.
  g = tf.Graph()
  with g.as_default():
    # Create model.
    inputs = tf.placeholder(
        shape=(None, image_size, image_size, 3),
        dtype=tf.float32)

    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)
    with slim.arg_scope(arg_scope):
      logits, end_points = mobilenet_v1.mobilenet_v1(
          inputs, num_classes=1001, is_training=False)
    pre_logits_flatten = end_points['AvgPool_1a']
    pre_logits_flatten = tf.squeeze(pre_logits_flatten, [1, 2])

    # Init from checkpoint. 
    init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.checkpoint_path, slim.get_model_variables(FLAGS.model_name))
    invalid_tensor_names = tf.report_uninitialized_variables()

  # Start session.
  output = {}
  with tf.Session(graph=g) as sess:
    init_fn(sess)
    invalid_tensor_names = sess.run(invalid_tensor_names)
    if len(invalid_tensor_names) > 0:
      raise ValueError('There are uninitialized variables!')

    api = AdsDatasetApi()
    api.init(images_dir=FLAGS.images_dir,
        entity_annot_file_ex=FLAGS.entity_annot_file)
    meta_list = api.get_meta_list_with_entity_annots_ex(
        score_threshold=FLAGS.score_threshold)
    logging.info('Loaded %s examples.', len(meta_list))

    for meta_index, meta in enumerate(meta_list):
      image = vis.image_load(meta['filename'], convert_to_rgb=True)
      image = vis.image_uint8_to_float32(image)

      if meta_index % 10 == 0:
        logging.info('On image %d of %d', meta_index, len(meta_list))

      # Batch operation, the first image is the full image.
      roi_list = [cv2.resize(image, (image_size, image_size))]
      score_list = []
      for entity in meta['entities_ex']:
        roi = vis.image_crop_and_resize(image, 
            box=(entity['xmin'], entity['ymin'], entity['xmax'], entity['ymax']), 
            crop_size=(image_size, image_size))
        roi_list.append(roi)
        score_list.append(entity['score'])
      assert len(roi_list) == 11
      assert len(score_list) == 10

      roi_batch = np.stack(roi_list, axis=0)
      if meta_index == 0:
        _visualize(roi_batch)

      # Get feature from mobilenet model.
      embeddings = sess.run(pre_logits_flatten, feed_dict={inputs: roi_batch})
      output[meta['image_id']] = {
        'image_emb': embeddings[0],
        'entity_score_list': np.array(score_list),
        'entity_emb_list': embeddings[1:]
      }

  with open(FLAGS.output_path, 'wb') as fp:
    np.save(fp, output)

if __name__ == '__main__':
  tf.app.run()
