
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import inception

from feature_extractors.feature_extractor import FeatureExtractor

slim = tf.contrib.slim


class InceptionV4Extractor(FeatureExtractor):

  @property
  def scope(self):
    """Returns variable scope."""
    return 'InceptionV4'

  @property
  def default_image_size(self):
    """Returns default image size."""
    return inception.inception_v4.default_image_size

  def extract_feature(self, image, is_training=True):
    """Extract feature vectors using inception v4 model.

    Args:
      image: a [batch, height, width, 3] float32 tensor denoting image.
      is_training: if True, update batch norm parameters.

    Returns:
      feature: a [batch, feature_dims] tensor denoting extracted feature.
    """
    image_size = self.default_image_size
    if image.get_shape().as_list()[1:3] != [image_size, image_size]:
      image = tf.image.resize_images(image, [image_size, image_size])

    arg_scope = inception.inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
      logits, end_points = inception.inception_v4(
          image, is_training=is_training)
    pre_logits_flatten = end_points['PreLogitsFlatten']

    return pre_logits_flatten
