
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

slim = tf.contrib.slim


class FeatureExtractor(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def scope(self):
    """Returns variable scope."""
    pass

  @abc.abstractmethod
  def extract_feature(self, image, is_training=True):
    """Extracts feature vectors.

    Args:
      image: a [batch, height, width, 3] tensor denoting image.
      is_training: if True, update batch norm parameters.

    Returns:
      feature: a [batch, feature_dims] tensor denoting extracted feature.
    """
    pass

  def assign_from_checkpoint_fn(self, checkpoint_path):
    """Returns a function to load from checkpoint.

    Args:
      checkpoint_path: path to the checkpoint file.

    Returns:
      init_fn: a function that that load weights from checkpoint.
    """
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,
        slim.get_model_variables(self.scope))
    return init_fn
