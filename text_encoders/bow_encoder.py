
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from protos import text_encoders_pb2
from text_encoders.text_encoder import TextEncoder

slim = tf.contrib.slim


class BOWEncoder(TextEncoder):

  def __init__(self, model_proto):
    """Initializes BOWEncoder.

    Args:
      model_proto: an instance of BOWEncoder proto.
    """
    if not isinstance(model_proto, text_encoders_pb2.BOWEncoder):
      raise ValueError('model_proto has to be an instance of BOWEncoder.')
    self._model_proto = model_proto

  @property
  def scope(self):
    """Returns variable scope."""
    return self._model_proto.scope

  def assign_from_checkpoint_fn(self, checkpoint_path):
    """Returns a function to load from checkpoint.

    Args:
      checkpoint_path: path to the checkpoint file.

    Returns:
      assign_fn: a function that that load weights from checkpoint.
    """
    if checkpoint_path:
      with open(checkpoint_path, 'rb') as fp:
        w2v = np.load(fp)

        # Pad for UNK_ID.
        pad = np.zeros([1, w2v.shape[1]], dtype=np.float32)
        w2v = np.concatenate([pad, w2v], 0)

        # Assign value for the variable.
        init_assign_op, init_feed_dict = slim.assign_from_values({
            self.embedding_weights.op.name: w2v} )

      def _assign_fn(sess):
        tf.logging.info('BOWEncoder::assign_fn is called, checkpoint_path=%s.',
            checkpoint_path)
        sess.run(init_assign_op, init_feed_dict)

      return _assign_fn

    def _empty_assign_fn(sess):
      tf.logging.info('BOWEncoder::empty_assign_fn is called.')

    return _empty_assign_fn

  def encode(self, text_lengths, text_strings, is_training=True):
    """Encodes texts into embedding vectors.

    Args:
      text_lengths: a [batch] tensor indicating lenghts of each text.
      text_strings: a [batch, max_text_len] tensor indicating multiple texts.
      is_training: if True, update batch norm parameters.

    Returns:
      embeddings: a [batch, embedding_size] tensor indicating embedding vectors
        of text_strings.
    """
    model_proto = self._model_proto

    embedding_weights = self.build_weights(
        vocab_size=model_proto.vocab_size,
        embedding_size=model_proto.embedding_size,
        init_width=model_proto.init_width,
        weight_decay=model_proto.weight_decay)

    embeddings = tf.nn.embedding_lookup(embedding_weights, text_strings)
    if text_encoders_pb2.BOWEncoder.RELU_6 == model_proto.activation_fn:
      if not is_training:
        embeddings = tf.nn.relu6(embeddings)
      else:
        # leaky relu.
        alpha = model_proto.leaky_relu_alpha
        embeddings = tf.maximum(alpha * embeddings, embeddings)
    elif text_encoders_pb2.BOWEncoder.SIGMOID == model_proto.activation_fn:
      embeddings = tf.sigmoid(embeddings)

    if is_training:
      embeddings = tf.nn.dropout(embeddings, model_proto.keep_prob)

    # Average embeddings by weights to get the representation of each string.
    batch_size, max_text_len = text_strings.get_shape().as_list()

    boolean_masks = tf.less(
        tf.range(max_text_len, dtype=tf.int64),
        tf.expand_dims(text_lengths, 1))
    weights = tf.cast(boolean_masks, tf.float32)

    if model_proto.average_method == text_encoders_pb2.BOWEncoder.AVG:
      # Average word embedding vectors.
      weights = tf.div(weights, 
          tf.maximum(1e-12, tf.tile( 
              tf.expand_dims(tf.cast(text_lengths, tf.float32), 1), 
              [1, max_text_len])))

    elif model_proto.average_method == text_encoders_pb2.BOWEncoder.SUM:
      # Sum word embedding vectors.
      pass

    else:
      raise ValueError('Invalid average method %s.' % (model_proto.average_method))

    weights = tf.expand_dims(weights, axis=1)
    embeddings_averaged = tf.squeeze(tf.matmul(weights, embeddings), [1])

    return embeddings_averaged
