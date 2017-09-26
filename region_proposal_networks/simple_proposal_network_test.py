
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import region_proposal_networks_pb2
from region_proposal_networks import builder
from region_proposal_networks import simple_proposal_network

from utils import vis

slim = tf.contrib.slim


class SimpleProposalNetworkTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      simple_proposal_network: {
      }
    """
    self.default_config = region_proposal_networks_pb2.RegionProposalNetwork()
    text_format.Merge(config_str, self.default_config)

  def test_predict(self):
    config = self.default_config

    image_data = vis.image_load('testdata/99790.jpg', convert_to_rgb=True)

    g = tf.Graph()
    with g.as_default():
      rpn = builder.build(config)
      self.assertIsInstance(rpn, simple_proposal_network.SimpleProposalNetwork)

      image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
      proposals = rpn.predict(tf.expand_dims(image, 0))

      self.assertEqual(
          proposals['num_detections'].get_shape().as_list(), [1])
      self.assertEqual(
          proposals['detection_scores'].get_shape().as_list(), [1, 1])
      self.assertEqual(
          proposals['detection_boxes'].get_shape().as_list(), [1, 1, 4])
      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      invalid_tensor_names = sess.run(invalid_tensor_names)
      self.assertListEqual(invalid_tensor_names.tolist(), [])

      proposals = sess.run(proposals, feed_dict={image: image_data})

      self.assertEqual(proposals['num_detections'][0], 1)
      self.assertNDArrayNear(
          proposals['detection_boxes'][0, 0],
          np.array([0.0, 0.0, 1.0, 1.0]),
          err=1e-8)

      y1, x1, y2, x2 = proposals['detection_boxes'][0, 0]
      vis_data = np.copy(image_data)
      vis.image_draw_bounding_box(vis_data, [x1, y1, x2, y2])
      vis.image_save('testdata/results/simple_proposal_network.jpg', 
          vis_data, convert_to_bgr=True)

if __name__ == '__main__':
    tf.test.main()
