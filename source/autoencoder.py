
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

slim = tf.contrib.slim

def _compute_l1_loss(labels, predictions):
  return tf.reduce_mean(tf.abs(labels - predictions))


def _compute_l2_loss(labels, predictions):
  return tf.reduce_mean(tf.squared_difference(labels, predictions))


class AutoEncoder(object):
  def __init__(self, config):
    """Init parameters of the model.

    Args:
      config: a dictionary containing model configuations, including:
        n_hidden: number of hidden units.
        weight_decay: weight_decay for regularizer.
        keep_prob: a value or tensor indicating dropout keep probability.
        use_batch_norm: if True, use batch norm in stead of bias.
        reconstruction_loss: either l1_loss or l2_loss
    """
    self._n_hidden = config['n_hidden']
    self._weight_decay = config['weight_decay']
    self._keep_prob = config['keep_prob']
    self._use_batch_norm = config['use_batch_norm']
    self._scope = config.get('scope', 'autoencoder')
    self._reconstruction_loss = config['reconstruction_loss']

  def build(self, inputs, is_training=True):
    """Build autoencoder model.

    Args:
      inputs: a [batch, n_dims] tensor denoting inputs.
      is_training: whether or not the layer is in training mode.

    Returns:
      hidden: a [batch, n_hidden] tensor.
      reconstruction: a [batch, n_dims] tensor.

    Raises:
      ValueError: if input is invalid.
    """
    batch, n_input = inputs.get_shape().as_list()

    if is_training:
      inputs = tf.nn.dropout(inputs, self._keep_prob)

    if self._use_batch_norm:
      normalizer_fn = slim.batch_norm
      normalizer_params = {
        'decay': 0.999,
        'center': True,
        'scale': True,
        'epsilon': 0.001,
        'is_training': is_training,
      }
    else:
      normalizer_fn = None
      normalizer_params = None

    with slim.arg_scope([slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(self._weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params):

      with tf.variable_scope(self._scope):
        hidden = slim.fully_connected(inputs, 
            num_outputs=self._n_hidden,
            activation_fn=None,
            scope='encoder/hidden')
        reconstruction = slim.fully_connected(hidden,
            num_outputs=n_input,
            activation_fn=tf.nn.relu,
            scope='decoder/reconstruction')

    tf.summary.histogram('autoencoder/hidden', hidden)
    tf.summary.histogram('autoencoder/reconstruction', reconstruction)
    return hidden, reconstruction

  def build_loss(self, embeddings, reconstruction, weight=1.0):
    """Build autoencoder loss.

    Args:
      embeddings: a [batch, n_dims] tensor.
      reconstruction: a [batch, n_dims] tensor.

    Returns:
      losses: a dictionary containing all loss terms
    """
    reconstruction_l1_loss = _compute_l1_loss(embeddings, reconstruction)
    reconstruction_l2_loss = _compute_l2_loss(embeddings, reconstruction)
    if 'l2_loss' == self._reconstruction_loss:
      reconstruction_loss = reconstruction_l2_loss
    else:
      reconstruction_loss = reconstruction_l1_loss

    reconstruction_loss_weighted = weight * reconstruction_loss
    tf.losses.add_loss(reconstruction_loss_weighted)

    return {
      'losses/reconstruction_loss': reconstruction_loss,
      'losses/reconstruction_loss_weighted': reconstruction_loss_weighted,
      'losses/reconstruction_l1_loss': reconstruction_l1_loss,
      'losses/reconstruction_l2_loss': reconstruction_l2_loss,
    }
