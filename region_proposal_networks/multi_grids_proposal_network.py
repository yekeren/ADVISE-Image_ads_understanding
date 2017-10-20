
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from object_detection.builders import model_builder

from protos import region_proposal_networks_pb2
from region_proposal_networks.region_proposal_network import RegionProposalNetwork

slim = tf.contrib.slim


class MultiGridsProposalNetwork(RegionProposalNetwork):

  def __init__(self, model_proto):
    """Init proposal network.

    Args:
      model_proto: an instance of MultiGridsProposalNetwork.
    """
    if not isinstance(model_proto, 
        region_proposal_networks_pb2.MultiGridsProposalNetwork):
      raise ValueError(
          'model_proto has to be an instance of MultiGridsProposalNetwork.')
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
    batch_size = images.get_shape()[0].value
    model = model_builder.build(
        self.model_proto.detection_model, 
        is_training=is_training)
    prediction_dict = model.predict(
        model.preprocess(tf.cast(images, tf.float32)))

    # Generate anchors.
    anchors = np.zeros(shape=(16, 4), dtype=np.float32)
    grid_rows = grid_cols = 3
    grid_height = 1.0 / grid_rows
    grid_width = 1.0 / grid_cols

    index = 0
    for i in xrange(grid_rows):
      for j in xrange(grid_cols):
        anchors[index] = [i * grid_height, j * grid_width, (i + 1) * grid_height, (j + 1) * grid_width]
        index += 1
    for i in xrange(grid_rows):
      anchors[index] = [i * grid_height, 0, (i + 1) * grid_height, 1]
      index += 1
    for j in xrange(grid_cols):
      anchors[index] = [0, j * grid_width, 1, (j + 1) * grid_width]
      index += 1
    anchors[index] = [0, 0, 1, 1]

    detections = {
      'num_detections': tf.fill([batch_size], 16.0),
      'detection_scores': tf.fill([batch_size, 16], 1.0),
      'detection_boxes': tf.tile(
          tf.expand_dims(tf.constant(np.array(anchors), np.float32), 0),
          [batch_size, 1, 1])
    }
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
