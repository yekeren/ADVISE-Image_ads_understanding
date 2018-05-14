
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from protos import triplet_mining_pb2


def _safe_batch_size(tensor):
  """Safely gets the batch size of tensor. 

  Args:
    tensor: a [batch, ...] tensor.

  Returns:
    batch_size: batch size of the tensor.
  """
  batch_size = tensor.get_shape()[0].value
  if batch_size is None:
    batch_size = tf.shape(tensor)[0]
  return batch_size


def _mine_all_examples(distances):
  """Mines all examples.

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


def _mine_random_examples(distances, negatives_per_anchor):
  """Mines random batch examples.

  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.
    negatives_per_anchor: number of negatives per each anchor.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  batch_size = _safe_batch_size(distances)

  pos_indices = tf.tile(tf.range(batch_size), [negatives_per_anchor])
  indices = tf.random_uniform(shape=tf.shape(pos_indices), 
      minval=1, maxval=batch_size, dtype=tf.int32)
  neg_indices = tf.mod(pos_indices + indices, batch_size)

  return pos_indices, neg_indices



def _mine_hard_examples(distances, top_k):
  """Mines hard examples.

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
  top_k = tf.minimum(top_k, batch_size - 1)

  pos_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
  pos_indices = tf.tile(pos_indices, [1, 1 + top_k])

  _, neg_indices = tf.nn.top_k(-distances, k=1 + top_k)

  masks = tf.not_equal(pos_indices, neg_indices)
  pos_indices = tf.boolean_mask(pos_indices, masks)
  neg_indices = tf.boolean_mask(neg_indices, masks)

  return pos_indices, neg_indices


def _mine_semi_hard_examples(distances):
  """Mines semi-hard examples.
    
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


def build_mining_func(config):
  """Builds triplet mining function based on config.

  Args:
    config: an instance of triplet_mining_pb2.TripletMining.

  Raises:
    ValueError if config is invalid.

  Returns:
    a callable that takes a distance matrix as input. 
  """
  triplet_mining = config.WhichOneof('triplet_mining')

  if 'mine_all' == triplet_mining:
    return _mine_all_examples

  if 'mine_semi_hard' == triplet_mining:
    return _mine_semi_hard_examples

  if 'mine_hard' == triplet_mining:
    top_k = config.mine_hard.top_k

    def _mine_hard_examples_wrap(distances):
      return _mine_hard_examples(distances, top_k)
    return _mine_hard_examples_wrap

  if 'mine_random' == triplet_mining:
    negatives_per_anchor = config.mine_random.negatives_per_anchor

    def _mine_random_examples_wrap(distances):
      return _mine_random_examples(distances, negatives_per_anchor)
    return _mine_random_examples_wrap

  raise ValueError('Invalid triplet_mining method {}.'.format(triplet_mining))


def compute_triplet_loss(anchors, positives, negatives, distance_fn, alpha):
  """Computes triplet loss.

  Args:
    anchors: a [batch, embedding_size] tensor.
    positives: a [batch, embedding_size] tensor.
    negatives: a [batch, embedding_size] tensor.
    distance_fn: a function using to measure distance between two [batch,
      embedding_size] tensors
    alpha: a float value denoting the margin.

  Returns:
    loss: the triplet loss tensor.
    summary: a dict mapping from summary names to summary tensors.
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
  loss_examples = tf.count_nonzero(dist1 + alpha >= dist2, dtype=tf.float32)
  return loss, { 'loss_examples': loss_examples}
