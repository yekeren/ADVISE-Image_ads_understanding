
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import mobilenet_v1

from feature_extractors.feature_extractor import FeatureExtractor

slim = tf.contrib.slim


class MobilenetV1Extractor(FeatureExtractor):

  @property
  def scope(self):
    """Returns variable scope."""
    return 'MobilenetV1'

  @property
  def default_image_size(self):
    """Returns default image size."""
    return mobilenet_v1.mobilenet_v1.default_image_size

  def extract_feature(self, image, is_training=True):
    """Extracts feature vectors using mobilenet v1 model.

    Args:
      image: a [batch, height, width, 3] float32 tensor denoting image.
      is_training: if True, update batch norm parameters.

    Returns:
      feature: a [batch, feature_dims] tensor denoting extracted feature.
    """
    image_size = self.default_image_size
    if image.get_shape().as_list()[1:3] != [image_size, image_size]:
      image = tf.image.resize_images(image, [image_size, image_size])

    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)
    with slim.arg_scope(arg_scope):
      logits, end_points = mobilenet_v1.mobilenet_v1(
          image, num_classes=1001, is_training=is_training)
    pre_logits_flatten = end_points['AvgPool_1a']
    pre_logits_flatten = tf.squeeze(pre_logits_flatten, [1, 2])

    return pre_logits_flatten
