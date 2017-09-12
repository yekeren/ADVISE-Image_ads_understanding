
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


class BOW(object):
  def __init__(self, config):
    """Init paramegers of the model.

    Args:
      config: a dictionary containing model configuations, including:
        vocab_size: size of vocabulary.
        embedding_size: size of embedding vector.
        weight_decay: weight_decay for embedding regularizer.
        keep_prob: a value or tensor indicating dropout keep probability.
    """
    self._vocab_size = config['vocab_size']
    self._embedding_size = config['embedding_size']
    self._weight_decay = config['weight_decay']
    self._keep_prob = config['keep_prob']

  def build_weights(self):
    """Build embedding matrix."""
    init_width = 0.08
    initializer = tf.random_uniform_initializer(-init_width, init_width)
    tf.logging.info('init_width of embedding_weights: %.3lf', init_width)

    with tf.variable_scope('BOW'):
      embedding_weights = tf.get_variable(
          name='weights', 
          shape=[self._vocab_size, self._embedding_size], 
          initializer=initializer,
          regularizer=slim.l2_regularizer(self._weight_decay))
    return embedding_weights

  def build(self, caption_lengths, caption_strings, is_training=True):
    """Build bag of word model.

    Args:
      caption_lengths: a [batch] tensor indicating lenghts of each caption.
      caption_strings: a [batch, max_caption_len] tensor indicating multiple captions.
      is_training: whether or not the layer is in training mode.

    Returns:
      embeddings_averaged: a [batch, embedding_size] tensor indicating feature representations.
      embeddings: a [batch, max_caption_len, embedding_size] tensor indicating words' embeddings.
      embedding_weights: a [vocab_size, embedding_size] tensor indicating word embedding matrix.
    """
    batch_size, max_caption_len = caption_strings.get_shape().as_list()

    # init_width = 0.08
    # initializer = tf.random_uniform_initializer(-init_width, init_width)
    # tf.logging.info('init_width of embedding_weights: %.3lf', init_width)
    #
    # Embed captions.
    # with tf.variable_scope('BOW'):
    #   embedding_weights = tf.get_variable(
    #       name='weights', 
    #       shape=[self._vocab_size, self._embedding_size], 
    #       initializer=initializer,
    #       regularizer=slim.l2_regularizer(self._weight_decay))

    embedding_weights = self.build_weights()
    embeddings = tf.nn.embedding_lookup(embedding_weights, caption_strings)
    if is_training:
      embeddings = tf.nn.dropout(embeddings, self._keep_prob)

    # Average embeddings by weights to get the representation of each caption.
    boolean_masks = tf.less(
        tf.range(max_caption_len, dtype=tf.int64),
        tf.expand_dims(caption_lengths, 1))
    weights = tf.cast(boolean_masks, tf.float32)
    weights = tf.div(
        weights, 
        1e-12 + tf.tile(
          tf.expand_dims(tf.cast(caption_lengths, tf.float32), 1), 
          [1, max_caption_len])
        )
    weights = tf.expand_dims(weights, axis=1)
    embeddings_averaged = tf.squeeze(tf.matmul(weights, embeddings), [1])
    return embeddings_averaged, embeddings, embedding_weights
