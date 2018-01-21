
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import region_proposal_networks_pb2
from region_proposal_networks import builder
from region_proposal_networks import multi_objects_proposal_network

from utils import vis

slim = tf.contrib.slim


class MultiObjectsProposalNetworkTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      multi_objects_proposal_network: {
        detection_model {
          ssd {
            num_classes: 1
            box_coder {
              faster_rcnn_box_coder {
                y_scale: 10.0
                x_scale: 10.0
                height_scale: 5.0
                width_scale: 5.0
              }
            }
            matcher {
              argmax_matcher {
                matched_threshold: 0.5
                unmatched_threshold: 0.5
                ignore_thresholds: false
                negatives_lower_than_unmatched: true
                force_match_for_each_row: true
              }
            }
            similarity_calculator {
              iou_similarity {
              }
            }
            anchor_generator {
              ssd_anchor_generator {
                num_layers: 6
                min_scale: 0.2
                max_scale: 0.95
                aspect_ratios: 1.0
                aspect_ratios: 2.0
                aspect_ratios: 0.5
                aspect_ratios: 3.0
                aspect_ratios: 0.3333
              }
            }
            image_resizer {
              fixed_shape_resizer {
                height: 300
                width: 300
              }
            }
            box_predictor {
              convolutional_box_predictor {
                min_depth: 0
                max_depth: 0
                num_layers_before_predictor: 0
                use_dropout: true
                dropout_keep_probability: 0.5
                kernel_size: 1
                box_code_size: 4
                apply_sigmoid_to_scores: false
                conv_hyperparams {
                  activation: RELU_6,
                  regularizer {
                    l2_regularizer {
                      weight: 0.00004
                    }
                  }
                  initializer {
                    truncated_normal_initializer {
                      stddev: 0.03
                      mean: 0.0
                    }
                  }
                  batch_norm {
                    train: true,
                    scale: true,
                    center: true,
                    decay: 0.9997,
                    epsilon: 0.001,
                  }
                }
              }
            }
            feature_extractor {
              type: 'ssd_mobilenet_v1'
              min_depth: 16
              depth_multiplier: 1.0
              conv_hyperparams {
                activation: RELU_6,
                regularizer {
                  l2_regularizer {
                    weight: 0.00004
                  }
                }
                initializer {
                  truncated_normal_initializer {
                    stddev: 0.03
                    mean: 0.0
                  }
                }
                batch_norm {
                  train: true,
                  scale: true,
                  center: true,
                  decay: 0.9997,
                  epsilon: 0.001,
                }
              }
            }
            loss {
              classification_loss {
                weighted_sigmoid {
                  anchorwise_output: true
                }
              }
              localization_loss {
                weighted_smooth_l1 {
                  anchorwise_output: true
                }
              }
              hard_example_miner {
                num_hard_examples: 3000
                iou_threshold: 0.99
                loss_type: CLASSIFICATION
                max_negatives_per_positive: 3
                min_negatives_per_image: 0
              }
              classification_weight: 1.0
              localization_weight: 1.0
            }
            normalize_loss_by_num_matches: true
            post_processing {
              batch_non_max_suppression {
                score_threshold: 0.2
                iou_threshold: 0.5
                max_detections_per_class: 7
                max_total_detections: 7
              }
              score_converter: SIGMOID
            }
          }
        }
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
      self.assertIsInstance(rpn, multi_objects_proposal_network.MultiObjectsProposalNetwork)

      image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
      proposals = rpn.predict(tf.expand_dims(image, 0), is_training=False)
      assign_fn = rpn.assign_from_checkpoint_fn(
          "models/zoo/ssd_mobilenet_v1_ads_09_17_2017/model.ckpt-9316")

      self.assertEqual(
          proposals['num_detections'].get_shape().as_list(), [1])
      self.assertEqual(
          proposals['detection_scores'].get_shape().as_list(), [1, 7])
      self.assertEqual(
          proposals['detection_boxes'].get_shape().as_list(), [1, 7, 4])
      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      assign_fn(sess)

      invalid_tensor_names = sess.run(invalid_tensor_names)
      self.assertListEqual(invalid_tensor_names.tolist(), [])

      proposals = sess.run(proposals, feed_dict={image: image_data})

      self.assertEqual(proposals['num_detections'][0], 1)
      self.assertNDArrayNear(
          proposals['detection_boxes'][0, 0],
          np.array([0.04972243, 0.0, 0.3760559, 0.97612488]),
          err=1e-6)

      vis_data = np.copy(image_data)
      for i in xrange(int(proposals['num_detections'][0])):
        y1, x1, y2, x2 = proposals['detection_boxes'][0, i]
        vis.image_draw_bounding_box(vis_data, [x1, y1, x2, y2])
      vis.image_save('testdata/results/multi_objects_proposal_network.jpg', 
          vis_data, convert_to_bgr=True)

if __name__ == '__main__':
    tf.test.main()

