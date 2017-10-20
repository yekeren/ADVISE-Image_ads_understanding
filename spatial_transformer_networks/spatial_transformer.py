
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from transformer.spatial_transformer import batch_transformer

slim = tf.contrib.slim

class SpatialTransformer(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, config):
    """Initializes SpatialTransformer."""
    pass

  @abc.abstractproperty
  def scope(self):
    """Returns variable scope."""
    pass

  @abc.abstractmethod
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
    pass

  def transform_image(self, image, theta, out_size):
    """Transforms image given theta.

    Args:
      image: a [batch, height, width, depth] float32 tensor indicating images.
      theta: a [batch, max_detections, 6] tensors denoting transformations.
      out_size: a python list or tuple in the format of [out_height, out_width].

    Returns:
      transformed_image: a [batch_height, width, depth] float32 tensor
        indicating transformed images.
    """
    depth = image.get_shape()[-1].value
    batch_size, max_detections, _ = theta.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(theta)[0]

    transformed_image = batch_transformer(image, theta, out_size)
    transformed_image = tf.reshape(transformed_image, [batch_size,
        max_detections, out_size[0], out_size[1], depth])
    return transformed_image

  def decode_bounding_box(self, theta):
    """Transforms bounding box given theta.

    Args:
      theta: a [batch, max_detections, 6] tensors denoting transformations.

    Returns:
      detection_boxes: a [batch, max_detections, 4] float32 tensor indicates
        detection boxes.
    """
    batch_size, max_detections, _ = theta.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(theta)[0]

    ones = tf.fill([batch_size * max_detections, 1], 1.0)
    theta = tf.reshape(theta, [batch_size * max_detections, 2, 3])

    point1 = tf.fill([batch_size * max_detections, 2], -1.0)
    point1 = tf.expand_dims(tf.concat([point1, ones], axis=1), 2)
    point1 = tf.squeeze(tf.matmul(theta, point1), [2])
    x1, y1 = tf.split(point1, 2, 1)

    point2 = tf.fill([batch_size * max_detections, 2], 1.0)
    point2 = tf.expand_dims(tf.concat([point2, ones], axis=1), 2)
    point2 = tf.squeeze(tf.matmul(theta, point2), [2])
    x2, y2 = tf.split(point2, 2, 1)

    detection_boxes = tf.reshape(
        tf.concat([y1, x1, y2, x2], axis=1),
        [batch_size, max_detections, 4])
    detection_boxes = (detection_boxes + 1.0) * 0.5
    return detection_boxes
