
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from object_detection.builders import model_builder

from protos import region_proposal_networks_pb2
from region_proposal_networks.region_proposal_network import RegionProposalNetwork

slim = tf.contrib.slim


class MultiObjectsProposalNetwork(RegionProposalNetwork):

  def __init__(self, model_proto):
    """Init proposal network.

    Args:
      model_proto: an instance of MultiObjectsProposalNetwork.
    """
    if not isinstance(model_proto, 
        region_proposal_networks_pb2.MultiObjectsProposalNetwork):
      raise ValueError(
          'model_proto has to be an instance of MultiObjectsProposalNetwork.')
    self.model_proto = model_proto

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
    model = model_builder.build(
        self.model_proto.detection_model, 
        is_training=is_training)
    prediction_dict = model.predict(
        model.preprocess(tf.cast(images, tf.float32)))
    detections = model.postprocess(prediction_dict)
    detections['feature_map'] = prediction_dict['feature_maps'][0]
    return detections

  def assign_from_checkpoint_fn(self, checkpoint_path):
    """Returns a function to load from checkpoint.

    Args:
      checkpoint_path: path to the checkpoint file.

    Returns:
      assign_fn: a function that that load weights from checkpoint.
    """
    variables_to_restore = filter(
        lambda x: 'FeatureExtractor' in x.op.name or 'BoxPredictor' in x.op.name,
        tf.global_variables())
    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)
