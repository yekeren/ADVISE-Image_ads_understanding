
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from protos import spatial_transformer_networks_pb2
from spatial_transformer_networks.spatial_transformer import SpatialTransformer

slim = tf.contrib.slim


class SimpleTransformer(SpatialTransformer):
  def __init__(self, model_proto):
    """Init SimpleTransformer.

    Args:
      model_proto: an instance of SimpleTransformer.
    """
    if not isinstance(model_proto, 
        spatial_transformer_networks_pb2.SimpleTransformer):
      raise ValueError(
          'model_proto has to be an instance of SimpleTransformer.')

  @property
  def scope(self):
    """Returns variable scope."""
    return None

  def predict_transformation(self, feature_map, 
      num_detections, detection_boxes, is_training=True):
    """Given the feature map and detection boxes, returns refined boxes.

    Args:
      feature_map: a [batch, height, width, depth] float32 tensor indicating a
        bunch of feature map.
      num_detections: a [batch] float32 tensor indicates number of detections.
      detection_boxes: a [batch, max_detections, 4] float32 tensor indicates
        coordinates (y1, x1, y2, x2) of boxes.
      is_training: if True, build training graph.

    Returns:
      theta: a [batch, max_detections, 6] tensors denoting transformations.
    """
    # Transform range of coordinates from [0, 1] to [-1, 1].
    detection_boxes = tf.multiply(tf.subtract(detection_boxes, 0.5), 2.0)

    # Get 2x3 transformation matrix.
    # / sx 0  tx \
    # \ 0  sy ty /
    batch_size, max_detections, _ = detection_boxes.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(detection_boxes)[0]
    y1, x1, y2, x2 = tf.split(detection_boxes, 4, 2)

    tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sx, sy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    p0 = tf.fill([batch_size, max_detections, 1], 0.0)

    theta = tf.concat([sx, p0, tx, p0, sy, ty], axis=2)
    return theta
