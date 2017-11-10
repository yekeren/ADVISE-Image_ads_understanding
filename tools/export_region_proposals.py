
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.builders import hyperparams_builder

from protos import region_proposal_networks_pb2
from region_proposal_networks import builder as rpn_builder
from utils import vis
from utils import ads_api

flags = tf.app.flags
flags.DEFINE_string('label_map', '../object_detection/configs/mscoco_label_map.pbtxt', 'Path to the object detection label map file.')
flags.DEFINE_string('model_proto', 'configs/coco_detection.pbtxt', 'Path to the pbtxt file describing the model.')
flags.DEFINE_string('checkpoint', '../object_detection/models/zoo/ssd_mobilenet_v1_coco_11_06_2017/model.ckpt', 'Path to the checkpoint file.')
flags.DEFINE_string('ads_config', 'configs/ads_api.empty.config', 'Path to the ads api config file.')
flags.DEFINE_string('output_json', 'coco_objects.json', 'Path to json output file.')

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

  label_map = label_map_util.load_labelmap(FLAGS.label_map)
  id2name = {}
  for item_id, item in enumerate(label_map.item):
    if item.HasField('display_name'):
      id2name[item.id - 1] = item.display_name
    else:
      id2name[item.id - 1] = item.name

  model_proto = region_proposal_networks_pb2.RegionProposalNetwork()
  with open(FLAGS.model_proto, 'r') as fp:
    text_format.Merge(fp.read(), model_proto)

  g = tf.Graph()
  with g.as_default():
    # Get detections from detection model.
    model = rpn_builder.build(model_proto)
    image_placeholder = tf.placeholder(
        shape=[None, None, 3], dtype=tf.uint8)
    images = tf.expand_dims(image_placeholder, 0)
    detections = model.predict(images, is_training=False)
    assign_fn = model.assign_from_checkpoint_fn(FLAGS.checkpoint)
    invalid_tensor_names = tf.report_uninitialized_variables()

  # Start the tf session.
  with tf.Session(graph=g, config=default_session_config_proto()) as sess:
    assign_fn(sess)

    invalid_tensor_names = sess.run(invalid_tensor_names)
    assert len(invalid_tensor_names) == 0

    examples = {}
    api = ads_api.AdsApi(FLAGS.ads_config)
    meta_list = api.get_meta_list()

    for meta_index, meta in enumerate(meta_list):
      if meta_index % 100 == 0:
        tf.logging.info('On image %d/%d.', meta_index, len(meta_list))
      image_data = vis.image_load(meta['file_path'], True)
      height, width, _ = image_data.shape
      if height > 300 or width > 300:
        image_data = cv2.resize(image_data, (300, 300))
      results = sess.run(detections, feed_dict={image_placeholder: image_data})

      # Pad the json objects.
      objects = []
      for i in xrange(results['num_detections'][0]):
        cid = int(results['detection_classes'][0, i])
        cname = id2name.get(cid, 'unclear')
        score = float(results['detection_scores'][0, i])
        y1, x1, y2, x2 = [float(x) for x in results['detection_boxes'][0, i]]

        objects.append({
            'object_id': cid,
            'object_name': cname,
            'score': score,
            'xmin': x1,
            'ymin': y1,
            'xmax': x2,
            'ymax': y2
            })
      examples[meta['image_id']] = objects

    with open(FLAGS.output_json, 'w') as fp:
      fp.write(json.dumps(examples))

if __name__ == '__main__':
  tf.app.run()
