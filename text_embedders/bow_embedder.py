
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import mobilenet_v1

from text_embedders.text_embedder import TextEmbedder

slim = tf.contrib.slim


class BOWEmbedder(TextEmbedder):

  def __init__(self, model_proto):
    """Initializes BOWEmbedder.

    Args:
      model_proto: an instance of BOWEmbedder proto.
    """
    self._model_proto = model_proto

  @property
  def scope(self):
    """Returns variable scope."""
    return 'BOW'

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

    embedding_weights = self.build_weights(
        vocab_size=model_proto.vocab_size,
        embedding_size=model_proto.embedding_size,
        weight_decay=model_proto.weight_decay)
    embeddings = tf.nn.embedding_lookup(embedding_weights, text_strings)
    if is_training:
      embeddings = tf.nn.dropout(embeddings, model_proto.keep_prob)

    # Average embeddings by weights to get the representation of each string.
    batch_size, max_text_len = text_strings.get_shape().as_list()

    boolean_masks = tf.less(
        tf.range(max_text_len, dtype=tf.int64),
        tf.expand_dims(text_lengths, 1))
    weights = tf.cast(boolean_masks, tf.float32)
    weights = tf.div(
        weights, 
        1e-12 + tf.tile(
          tf.expand_dims(tf.cast(text_lengths, tf.float32), 1), 
          [1, max_text_len])
        )
    weights = tf.expand_dims(weights, axis=1)
    embeddings_averaged = tf.squeeze(tf.matmul(weights, embeddings), [1])

    return embeddings_averaged
