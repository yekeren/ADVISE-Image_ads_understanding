
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

slim = tf.contrib.slim


class TextEmbedder(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def scope(self):
    """Returns variable scope."""
    pass

  @abc.abstractmethod
  def embed(self, text_lengths, text_strings, is_training=True):
    """Embeds texts into embedding vectors.

    Args:
      text_lengths: a [batch] tensor indicating lenghts of each text.
      text_strings: a [batch, max_text_len] tensor indicating multiple texts.
      is_training: if True, update batch norm parameters.
    """
    pass

  def build_weights(self, vocab_size, embedding_size, weight_decay=0.0):
    """Builds word embedding matrix.

    Args:
      vocab_size: size of vocabulary.
      embedding_size: dimensions of embedding vector.
      weight_decay: weight decay of word embedding weights.

    Returns:
      embedding_weights: a [vocab_size, embedding_size] tensor.
    """
    initializer = tf.random_uniform_initializer(-0.08, 0.08)

    with tf.variable_scope(self.scope):
      embedding_weights = tf.get_variable(
          name='weights', 
          shape=[vocab_size, embedding_size], 
          initializer=initializer,
          regularizer=slim.l2_regularizer(weight_decay))
    return embedding_weights
