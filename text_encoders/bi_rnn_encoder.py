
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow import logging
from protos import text_encoders_pb2
from text_encoders.text_encoder import TextEncoder

slim = tf.contrib.slim


class BiRNNEncoder(TextEncoder):

  def __init__(self, model_proto, is_training):
    """Initializes BiRNNEncoder.

    Args:
      model_proto: an instance of BiRNNEncoder proto.

    Raises:
      ValueError: if model_proto is invalid.
    """
    super(BiRNNEncoder, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, text_encoders_pb2.BiRNNEncoder):
      raise ValueError('The model_proto has to be an instance of BiRNNEncoder.')

    if model_proto.cell_type != 'LSTM':
      raise ValueError('Only LSTM is supported.')

    def _rnn_cell():
      cell = tf.nn.rnn_cell.BasicLSTMCell( 
          num_units=model_proto.rnn_num_units, forget_bias=1.0)

      if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper( 
            cell,
            input_keep_prob=model_proto.rnn_input_keep_prob,
            output_keep_prob=model_proto.rnn_output_keep_prob,
            state_keep_prob=model_proto.rnn_state_keep_prob)
      return cell

    self._rnn_cell = tf.contrib.rnn.MultiRNNCell([
        _rnn_cell() for _ in xrange(model_proto.rnn_num_layers)])

  def _set_init_fn(self, embedding_weights, filename):
    """Sets the initialization function.

    Args:
      embedding_weights: a [vocab_size, embedding_size] tensor denoting
        embedding matrix.
      filename: the file path to initialize word embedding matrix from.
    """
    if not filename:
      def _default_init_fn(_):
        pass
      self._init_fn = _default_init_fn

    else:
      with open(filename, 'rb') as fp:
        word2vec = np.load(fp)
        init_assign_op, init_feed_dict = slim.assign_from_values({
            embedding_weights.op.name: word2vec} )

      def _init_fn(sess):
        sess.run(init_assign_op, init_feed_dict)
        logging.info('Initialize word embedding from %s.', filename)

      self._init_fn = _init_fn

  def encode(self, text_strings, text_lengths):
    """Encodes texts into embedding vectors.

    Args:
      text_strings: a [batch, max_text_len] tensor indicating multiple texts.
      text_lengths: a [batch] tensor indicating lenghts of each text.

    Returns:
      text_encoded: a [batch, embedding_size] tensor indicating final encoding.
      text_embedding: a [batch, max_text_len, embedding_size] tf.float tensor.
    """
    model_proto = self._model_proto
    is_training = self._is_training
    rnn_cell = self._rnn_cell

    # Build word embedding weights.
    init_width = model_proto.init_width
    weight_decay = model_proto.weight_decay

    initializer = tf.random_uniform_initializer(-init_width, init_width)

    embedding_weights = tf.get_variable(
        name='{}/weights'.format(model_proto.scope),
        shape=[model_proto.vocab_size, model_proto.embedding_size],
        trainable=model_proto.trainable,
        initializer=initializer,
        regularizer=slim.l2_regularizer(weight_decay))
    embeddings = tf.nn.embedding_lookup(embedding_weights, text_strings)

    norms = tf.norm(embedding_weights, axis=1)
    tf.summary.scalar(
        '{}/norm_max'.format(model_proto.scope), tf.reduce_max(norms))
    tf.summary.scalar(
        '{}/norm_min'.format(model_proto.scope), tf.reduce_min(norms))
    tf.summary.scalar(
        '{}/norm_avg'.format(model_proto.scope), tf.reduce_mean(norms))

    # Build RNN cell.
    batch = text_strings.get_shape()[0].value
    if batch is None:
      batch = tf.shape(text_strings)[0]

    # Build RNN.
    with tf.variable_scope(model_proto.scope, initializer=initializer) as rnn_scope:
      outputs, encoded_state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=rnn_cell,
          cell_bw=rnn_cell,
          inputs=embeddings,
          sequence_length=text_lengths,
          dtype=tf.float32,
          scope=rnn_scope)

    outputs_fw, outputs_bw = outputs
    states_fw, states_bw = encoded_state

    if model_proto.repr_method == text_encoders_pb2.BiRNNEncoder.USE_CONCAT:
      outputs = tf.concat([outputs_fw, outputs_bw], 2)
      states = tf.concat([states_fw[-1].h, states_bw[-1].h], 1)
    elif model_proto.repr_method == text_encoders_pb2.BiRNNEncoder.USE_AVERAGE:
      outputs = 0.5 * (outputs_fw + outputs_bw)
      states = 0.5 * (states_fw[-1].h + states_bw[-1].h)

    self._set_init_fn(embedding_weights, model_proto.init_emb_matrix_path)
    return states, outputs, embeddings
