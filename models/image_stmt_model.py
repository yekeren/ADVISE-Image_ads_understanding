
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import logging

from models import model
from models import utils
from losses import triplet_loss
from text_encoders import builder

slim = tf.contrib.slim


class Model(model.Model):
  """ImageStmtModel."""

  def __init__(self, model_proto, is_training=False):
    """Initializes ads model.

    Args:
      model_proto: shall not be checked here.
      is_training: if True, training graph would be built.
    """
    super(Model, self).__init__(model_proto, is_training)

  def build_loss(self, predictions, **kwargs):
    """Build tensorflow graph for computing loss.

    Args:
      predictions: a dict mapping from names to predicted tensors, involving:
        img_encoded: a [batch, common_dimensions] tf.float32 tensor.
        stmt_encoded: a [batch, common_dimensions] tf.float32 tensor.

    Returns:
      loss_dict: a dict mapping from names to loss tensors.
    """
    loss_dict = {}

    model_proto = self._model_proto
    is_training = self._is_training
    mining_fn = self._mining_fn

    image_id = predictions['image_id']
    img_encoded = predictions['img_encoded']
    stmt_encoded = predictions['stmt_encoded']

    # Compute the triplet loss.
    margin = model_proto.triplet_margin
    keep_prob = model_proto.joint_emb_dropout_keep_prob

    def distance_fn(x, y):
      """Distance function."""
      distance = slim.dropout(tf.multiply(x, y), keep_prob,
          is_training=is_training)
      distance = 1 - tf.reduce_sum(distance, 1)
      return distance

    def refine_fn(pos_indices, neg_indices):
      """Refine function."""
      pos_ids = tf.gather(image_id, pos_indices)
      neg_ids = tf.gather(image_id, neg_indices)

      masks = tf.not_equal(pos_ids, neg_ids)
      pos_indices = tf.boolean_mask(pos_indices, masks)
      neg_indices = tf.boolean_mask(neg_indices, masks)
      return pos_indices, neg_indices

    loss_img_stmt, summary = triplet_loss_wrap_func(
        img_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin,
        'img_stmt')
    loss_stmt_img, summary = triplet_loss_wrap_func(
        stmt_encoded, img_encoded, distance_fn, mining_fn, refine_fn, margin,
        'stmt_img')

    loss_dict = {
      'triplet_img_stmt': loss_img_stmt,
      'triplet_stmt_img': loss_stmt_img,
    }

    return loss_dict


def triplet_loss_wrap_func(
    anchors, positives, distance_fn, mining_fn, refine_fn, margin, tag=None):
  """Wrapper function for triplet loss.

  Args:
    anchors: a [batch, common_dimensions] tf.float32 tensor.
    positives: a [batch, common_dimensions] tf.float32 tensor.
    similarity_matrx: a [common_dimensions, common_dimensions] tf.float32 tensor.
    distance_fn: a callable that takes two batch of vectors as input.
    mining_fn: a callable that takes distance matrix as input.
    refine_fn: a callable that takes pos_indices and neg_indices as inputs.
    margin: margin alpha of the triplet loss.

  Returns:
    loss: the loss tensor.
  """
  distances = tf.multiply(
      tf.expand_dims(anchors, 1), tf.expand_dims(positives, 0))
  distances = 1 - tf.reduce_sum(distances, 2)

  pos_indices, neg_indices = mining_fn(distances)
  if not refine_fn is None:
    pos_indices, neg_indices = refine_fn(pos_indices, neg_indices)

  loss, summary = triplet_loss.compute_triplet_loss(
      anchors=tf.gather(anchors, pos_indices), 
      positives=tf.gather(positives, pos_indices), 
      negatives=tf.gather(positives, neg_indices),
      distance_fn=distance_fn,
      alpha=margin)

  if tag is not None:
    for k, v in summary.iteritems():
      tf.summary.scalar('triplet_train/{}_{}'.format(tag, k), v)

  return loss, summary
