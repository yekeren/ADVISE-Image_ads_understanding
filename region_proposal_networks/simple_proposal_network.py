
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from protos import region_proposal_networks_pb2
from region_proposal_networks.region_proposal_network import RegionProposalNetwork


class SimpleProposalNetwork(RegionProposalNetwork):

  def __init__(self, model_proto):
    """Init proposal network.

    Args:
      model_proto: an instance of SimpleProposalNetwork.
    """
    if not isinstance(model_proto, 
        region_proposal_networks_pb2.SimpleProposalNetwork):
      raise ValueError(
          'model_proto has to be an instance of SimpleProposalNetwork.')

  @property
  def scope(self):
    """Returns variable scope."""
    return None

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
    batch_size = images.get_shape()[0].value
    detections = {
      'num_detections': tf.fill([batch_size], 1.0),
      'detection_scores': tf.fill([batch_size, 1], 1.0),
      'detection_boxes': tf.tile(
          tf.constant(np.array([[[0.0, 0.0, 1.0, 1.0]]], np.float32)),
          [batch_size, 1, 1])
    }
    return detections

  def assign_from_checkpoint_fn(self, checkpoint_path):
    """Returns a function to load from checkpoint.

    Args:
      checkpoint_path: path to the checkpoint file.

    Returns:
      assign_fn: a function that that load weights from checkpoint.
    """
    tf.logging.info('SimpleProposalNetwork::assign_from_checkpoint_fn.')
