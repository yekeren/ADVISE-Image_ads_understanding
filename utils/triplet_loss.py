
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def _safe_batch_size(tensor):
  """Safely get the batch size of tensor. 

  Args:
    tensor: a [batch, ...] tensor.

  Returns:
    batch_size: batch size of the tensor.
  """
  batch_size = tensor.get_shape()[0].value
  if batch_size is None:
    batch_size = tf.shape(tensor)[0]
  return batch_size

def mine_all_examples(distances):
  """Mine all examples.

  Mine all returns all True examples in the following matrix:

  / 0, 1, 1, 1 \
  | 1, 0, 1, 1 |
  | 1, 1, 0, 1 |
  \ 1, 1, 1, 0 /
    
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  batch_size = _safe_batch_size(distances)
  indices = tf.where(tf.less(tf.diag(tf.fill([batch_size], 1)), 1))
  return indices[:, 0], indices[:, 1]

def mine_hard_examples(distances, top_k):
  """Mine hard examples.

  Mine hard returns examples with smallest values in the following masked matrix:

  / 0, 1, 1, 1 \
  | 1, 0, 1, 1 |
  | 1, 1, 0, 1 |
  \ 1, 1, 1, 0 /
    
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.
    top_k: number of negative examples to choose per each row.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  batch_size = _safe_batch_size(distances)

  pos_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
  pos_indices = tf.tile(pos_indices, [1, 1 + top_k])

  _, neg_indices = tf.nn.top_k(-distances, k=1 + top_k)

  masks = tf.not_equal(pos_indices, neg_indices)
  pos_indices = tf.boolean_mask(pos_indices, masks)
  neg_indices = tf.boolean_mask(neg_indices, masks)

  return pos_indices, neg_indices


def mine_semi_hard_examples(distances):
  """Mine semi-hard examples.
    
  Mine semi-hard returns examples that have dist_fn(a, p) < dist_fn(a, n).

  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  pos_distances = tf.expand_dims(tf.diag_part(distances), 1)
  indices = tf.where(pos_distances < distances)
  return indices[:, 0], indices[:, 1]


def compute_triplet_loss(anchors, positives, negatives, distance_fn, alpha):
  """Compute triplet loss.

  Args:
    anchors: a [batch, embedding_size] tensor.
    positives: a [batch, embedding_size] tensor.
    negatives: a [batch, embedding_size] tensor.
    distance_fn: a function using to measure distance between two [batch,
      embedding_size] tensors
    alpha: a float value denoting the margin.

  Returns:
    loss_summary: a python dictionary mapping from name to loss tensors.
  """
  batch_size = _safe_batch_size(anchors)
  batch_size = tf.maximum(1e-12, tf.cast(batch_size, tf.float32))

  dist1 = distance_fn(anchors, positives)
  dist2 = distance_fn(anchors, negatives)

  losses = tf.maximum(dist1 - dist2 + alpha, 0)
  losses = tf.boolean_mask(losses, losses > 0)

  loss = tf.cond(tf.shape(losses)[0] > 0,
      lambda: tf.reduce_mean(losses),
      lambda: 0.0)

  # Gather statistics.
  good_ratio = tf.div(
      tf.count_nonzero(dist1 < dist2, dtype=tf.float32), 
      batch_size)
  loss_ratio = tf.div(
      tf.count_nonzero(dist1 + alpha >= dist2, dtype=tf.float32),
      batch_size)

  return {
    'losses/triplet_loss': loss,
    'triplet/batch_size': batch_size,
    'triplet/loss_ratio': loss_ratio,
    'triplet/good_ratio': good_ratio,
  }

