
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow import logging
from protos import text_encoders_pb2
from text_encoders.text_encoder import TextEncoder

slim = tf.contrib.slim


class BOWEncoder(TextEncoder):

  def __init__(self, model_proto, is_training):
    """Initializes BOWEncoder.

    Args:
      model_proto: an instance of BOWEncoder proto.
    """
    super(BOWEncoder, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, text_encoders_pb2.BOWEncoder):
      raise ValueError('The model_proto has to be an instance of BOWEncoder.')

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

    # Build word embedding weights.
    init_width = model_proto.init_width
    weight_decay = model_proto.weight_decay

    embedding_weights = tf.get_variable(
        name='{}/weights'.format(model_proto.scope),
        shape=[model_proto.vocab_size, model_proto.embedding_size],
        trainable=model_proto.trainable,
        initializer=tf.random_uniform_initializer(-init_width, init_width),
        regularizer=slim.l2_regularizer(weight_decay))
    embeddings = tf.nn.embedding_lookup(embedding_weights, text_strings)
    embeddings = slim.dropout(embeddings, model_proto.dropout_keep_prob,
        is_training=is_training)
    self.embedding_weights = embedding_weights

    norms = tf.norm(embedding_weights, axis=1)
    tf.summary.scalar(
        '{}/norm_max'.format(model_proto.scope), tf.reduce_max(norms))
    tf.summary.scalar(
        '{}/norm_min'.format(model_proto.scope), tf.reduce_min(norms))
    tf.summary.scalar(
        '{}/norm_avg'.format(model_proto.scope), tf.reduce_mean(norms))

    # Average word embedding vectors.
    _, max_text_len = text_strings.get_shape().as_list()
    if max_text_len is None:
      max_text_len = tf.shape(text_strings, out_type=tf.int32)[1]

    boolean_masks = tf.less(
        tf.range(max_text_len, dtype=tf.int32), 
        tf.expand_dims(tf.cast(text_lengths, tf.int32), 1))

    weights = tf.cast(boolean_masks, tf.float32)
    if model_proto.repr_method == text_encoders_pb2.BOWEncoder.USE_OUTPUT_AVG:
      weights = tf.div(weights, 
          tf.maximum(1e-12, tf.tile( 
              tf.expand_dims(tf.cast(text_lengths, tf.float32), 1), 
              tf.stack([1, max_text_len]))))

    text_encoded = tf.squeeze(
        tf.matmul(tf.expand_dims(weights, 1), embeddings), [1])

    self._set_init_fn(embedding_weights, model_proto.init_emb_matrix_path)
    return text_encoded, embeddings, embeddings
