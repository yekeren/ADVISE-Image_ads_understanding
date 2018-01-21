
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf


class RegionProposalNetwork(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def scope(self):
    """Returns variable scope."""
    pass

  @abc.abstractmethod
  def predict(self, images, is_training=True):
    """Predicts region proposals from images.

    Args:
      images: a [batch, height, width, 3] uint8 tensor.
      is_training: if True, build training graph.

    Returns:
      detections: a dictionary containing the following fields
        num_detections: a [batch] float32 tensor.
        detection_scores: a [batch, max_detections] float32 tensor.
        detection_boxes: a [batch, max_detections, 4] float32 tensor.
    """
    pass

  @abc.abstractmethod
  def assign_from_checkpoint_fn(self, checkpoint_path):
    """Returns a function to load from checkpoint.

    Args:
      checkpoint_path: path to the checkpoint file.

    Returns:
      assign_fn: a function that that load weights from checkpoint.
    """
    pass
