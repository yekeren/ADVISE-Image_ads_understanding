
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import mobilenet_v1

from protos import ads_emb_model_pb2
from text_embedders.text_embedder import TextEmbedder

slim = tf.contrib.slim


class RNNEmbedder(TextEmbedder):

  def __init__(self, model_proto):
    """Initializes RNNEmbedder.

    Args:
      model_proto: an instance of RNNEmbedder proto.

    Raises:
      ValueError: if model_proto is invalid.
    """
    if not isinstance(model_proto, ads_emb_model_pb2.RNNEmbedder):
      raise ValueError('model_proto has to be an instance of RNNEmbedder.')

    if model_proto.cell_type != 'LSTM':
      raise ValueError('Only LSTM is supported.')

    self._model_proto = model_proto

  @property
  def scope(self):
    """Returns variable scope."""
    return 'RNN'

  def embed(self, text_lengths, text_strings, is_training=True):
    """Embeds texts into embedding vectors.

    Args:
      text_lengths: a [batch] tensor indicating lenghts of each text.
      text_strings: a [batch, max_text_len] tensor indicating multiple texts.
      is_training: if True, update batch norm parameters.

    Returns:
      embeddings: a [batch, embedding_size] tensor indicating embedding vectors
        of text_strings.
    """
    model_proto = self._model_proto

    # Embed the inputs.
    embedding_weights = self.build_weights(
        vocab_size=model_proto.vocab_size,
        embedding_size=model_proto.embedding_size,
        weight_decay=model_proto.weight_decay)
    embeddings = tf.nn.embedding_lookup(embedding_weights, text_strings)
    if is_training:
      embeddings = tf.nn.dropout(embeddings, model_proto.keep_prob)

    # Average embeddings by weights to get the representation of each string.
    batch_size = text_strings.get_shape()[0].value
    if batch_size is None:
      batch_size = tf.shape(text_strings)[0]

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=model_proto.rnn_num_units,
        state_is_tuple=True)

    if is_training:
      rnn_cell = tf.contrib.rnn.DropoutWrapper(
          rnn_cell,
          input_keep_prob=model_proto.rnn_input_keep_prob,
          output_keep_prob=model_proto.rnn_output_keep_prob,
          state_keep_prob=model_proto.rnn_state_keep_prob)

    # Build RNN.
    initializer = tf.random_uniform_initializer(-0.08, 0.08)
    with tf.variable_scope(self.scope, initializer=initializer) as rnn_scope:
      zero_state = rnn_cell.zero_state(
          batch_size=batch_size, dtype=tf.float32)

      outputs, encoded_state = tf.nn.dynamic_rnn(
          cell=rnn_cell,
          inputs=embeddings,
          sequence_length=text_lengths,
          initial_state=zero_state,
          dtype=tf.float32,
          scope=rnn_scope)

    encoded_state = tf.concat([encoded_state.c, encoded_state.h], axis=1)

    # Use fully connected layer to get the caption embedding vectors.
    normalizer_fn = slim.batch_norm
    normalizer_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'is_training': is_training,
    }
    with slim.arg_scope([slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(model_proto.weight_decay_fc),
        weights_initializer=slim.variance_scaling_initializer(),
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params):
      text_embs = slim.fully_connected(encoded_state, 
          num_outputs=model_proto.caption_embedding_size,
          activation_fn=None,
          scope=self.scope + '/project')
    return text_embs

