
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from object_detection.builders import hyperparams_builder
from object_detection.protos import hyperparams_pb2

from protos import spatial_transformer_networks_pb2
from spatial_transformer_networks.spatial_transformer import SpatialTransformer

slim = tf.contrib.slim


class AffineTransformer(SpatialTransformer):
  def __init__(self, model_proto):
    """Init AffineTransformer.

    Args:
      model_proto: an instance of AffineTransformer.
    """
    if not isinstance(model_proto, 
        spatial_transformer_networks_pb2.AffineTransformer):
      raise ValueError(
          'model_proto has to be an instance of AffineTransformer.')
    self.model_proto = model_proto

  @property
  def scope(self):
    """Returns variable scope."""
    return self.model_proto.scope

  def _get_prior(self, detection_boxes):
    """Get 3x3 prior transformations.

    Args:
      detection_boxes: a [batch, max_detections, 4] float32 tensor indicates
        coordinates (y1, x1, y2, x2) of boxes.

    Returns:
      priors: a [batch, max_detections, 6] tensors denoting prior
        transformations.
    """
    # Transform range of coordinates from [0, 1] to [-1, 1].
    detection_boxes = tf.multiply(tf.subtract(detection_boxes, 0.5), 2.0)

    # Get 3x3 prior transformations.
    # / sx 0  tx \
    # | 0  sy ty |
    # \ 0  0  1  /
    batch_size, max_detections, _ = detection_boxes.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(detection_boxes)[0]
    y1, x1, y2, x2 = tf.split(detection_boxes, 4, 2)

    tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sx, sy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    p0 = tf.fill([batch_size, max_detections, 1], 0.0)
    p1 = tf.fill([batch_size, max_detections, 1], 1.0)

    priors = tf.concat([sx, p0, tx, p0, sy, ty, p0, p0, p1], axis=2)
    return tf.reshape(priors, [batch_size, max_detections, 3, 3])

  def _get_relative_transformation(self, 
      feature_map, detection_boxes, is_training=True):
    """Get relative transformations.

    Args:
      feature_map: a [batch, height, width, depth] float32 tensor.
      detection_boxes: a [batch, max_detections, 4] float32 tensor indicates
        coordinates (y1, x1, y2, x2) of boxes.
      is_training: if True, build training graph.

    Returns:
      delta: a [batch, max_detections, 6] tensors denoting relative
        transformations.
      score: a [batch, max_detections] tensors denoting confidence score.

    Raises:
      ValueError: if shape of input tensor is invalid.
    """
    if feature_map.get_shape().as_list()[1:] != [19, 19, 512]:
      raise ValueError('Shape of input tensor is invalid.')

    batch_size, max_detections, _ = detection_boxes.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(detection_boxes)[0]

    # Reshape tensors:
    #   detection_boxes: a [batch * max_detections, 4] float32 tensor.
    #   box_ind: a [batch * max_detections] int64 tensor.
    detection_boxes = tf.reshape(detection_boxes, [-1, 4])

    box_ind = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
    box_ind = tf.reshape(tf.tile(box_ind, [1, max_detections]), [-1])

    # Interpolate to get local feature representation.
    #   feature_map: [batch * max_detections, crop_height, crop_width, depth]
    crop_height, crop_width = feature_map.get_shape().as_list()[1:3]
    crop_height, crop_width = int((crop_height + 1) / 2), int((crop_width + 1) / 2)
    feature_map = tf.image.crop_and_resize(
        feature_map, detection_boxes, box_ind, 
        crop_size=(crop_height, crop_width))

    # Use ConvNet to extract transformation parameters [sx, sy, tx, ty].
    model_proto = self.model_proto
    conv_hyperparams = hyperparams_builder.build(
        model_proto.conv_hyperparams, is_training=is_training)

    tf.logging.info('*' * 128)
    with slim.arg_scope(conv_hyperparams):
      with tf.variable_scope(self.scope) as scope:
        # Net: [batch * max_detections, 10, 10, 512]
        net = feature_map
        net = slim.conv2d(net, 32, [1, 1], stride=1, scope='Conv2d_1')
        tf.logging.info('%s: %s', net.op.name, net.get_shape().as_list())

        # Net: [batch * max_detections, 10, 10, 32]
        net1 = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1,
            stride=2, scope='Conv2d_2/depthwise_3x3')
        net2 = slim.separable_conv2d(net, None, [5, 5], depth_multiplier=1,
            stride=2, scope='Conv2d_2/depthwise_5x5')
        net = tf.concat([net1, net2], axis=3)

        net = slim.conv2d(net, 32, [1, 1], stride=1, scope='Conv2d_2/pointwise')
        tf.logging.info('%s: %s', net.op.name, net.get_shape().as_list())

        # Net: [batch * max_detections, 5, 5, 32]
        net1 = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1,
            stride=2, scope='Conv2d_3/depthwise_3x3')
        net2 = slim.separable_conv2d(net, None, [5, 5], depth_multiplier=1,
            stride=2, scope='Conv2d_3/depthwise_5x5')
        net = tf.concat([net1, net2], axis=3)

        net = slim.conv2d(net, 32, [1, 1], stride=1, scope='Conv2d_3/pointwise')
        tf.logging.info('%s: %s', net.op.name, net.get_shape().as_list())

        # Net: [batch * max_detections, 3, 3, 32]
        net = slim.conv2d(net, 16, [3, 3],
            stride=1, padding='VALID', scope='Conv2d_4')
        tf.logging.info('%s: %s', net.op.name, net.get_shape().as_list())

        # Net: [batch * max_detections, 1, 1, 8]
        net = tf.contrib.layers.flatten(net)

        net = slim.fully_connected(net, 
            num_outputs=5,
            activation_fn=None,
            scope='project')
        tf.logging.info('%s: %s', net.op.name, net.get_shape().as_list())

    # Get 3x3 relative transformations.
    # / sx 0  tx \
    # | 0  sy ty |
    # \ 0  0  1  /
    sx, sy, tx, ty, score = tf.split(net, 5, 1)

    def _regularization_loss(val, limit, name):
      loss = tf.reduce_mean(tf.maximum(tf.abs(val) - limit, 0))
      tf.summary.scalar(name, loss)
      return loss

    tf.losses.add_loss(
        _regularization_loss(sx, model_proto.limit_scale, 'losses/sx'))
    tf.losses.add_loss(
        _regularization_loss(sy, model_proto.limit_scale, 'losses/sy'))
    tf.losses.add_loss(
        _regularization_loss(tx, model_proto.limit_translate, 'losses/tx'))
    tf.losses.add_loss(
        _regularization_loss(tx, model_proto.limit_translate, 'losses/ty'))

    # Location regularization loss.

    sx, sy = tf.exp(sx), tf.exp(sy)
    p0 = tf.fill([batch_size * max_detections, 1], 0.0)
    p1 = tf.fill([batch_size * max_detections, 1], 1.0)

    delta = tf.concat([sx, p0, tx, p0, sy, ty, p0, p0, p1], axis=1)
    delta = tf.reshape(delta, [batch_size, max_detections, 3, 3])
    score = tf.reshape(score, [batch_size, max_detections])

    return delta, score

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
    if feature_map is None:
      raise ValueError('feature map is invalid.')

    batch_size, max_detections, _ = detection_boxes.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(detection_boxes)[0]

    # Extract 3x3 prior transformations.
    priors = self._get_prior(detection_boxes)

    # Predict 3x3 relative transformations.
    delta, confidence_score = self._get_relative_transformation(
        feature_map, detection_boxes)

    # Final transformation is prior x delta.
    theta = tf.matmul(priors, delta)
    theta = tf.reshape(theta, [batch_size, max_detections, 9])[:, :, :6]
    return theta, confidence_score

