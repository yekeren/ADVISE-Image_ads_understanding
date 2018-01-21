
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import vgg

from feature_extractors.feature_extractor import FeatureExtractor

slim = tf.contrib.slim


class VGG16Extractor(FeatureExtractor):

  @property
  def scope(self):
    """Returns variable scope."""
    return 'vgg_16'

  @property
  def default_image_size(self):
    """Returns default image size."""
    return vgg.vgg_16.default_image_size

  def extract_feature(self, image, is_training=True):
    """Extract feature vectors using vgg 16 model.

    Args:
      image: a [batch, height, width, 3] float32 tensor denoting image.
      is_training: if True, update batch norm parameters.

    Returns:
      feature: a [batch, feature_dims] tensor denoting extracted feature.
    """
    image_size = self.default_image_size
    if image.get_shape().as_list()[1:3] != [image_size, image_size]:
      image = tf.image.resize_images(image, [image_size, image_size])

    arg_scope = vgg.vgg_arg_scope()
    with slim.arg_scope(arg_scope):
      outputs, end_points = vgg.vgg_16(
          image, is_training=is_training)
    pre_logits_flatten = tf.squeeze(
        end_points['vgg_16/fc7'], [1, 2], name='vgg_16/fc7/squeezed')

    return pre_logits_flatten
