
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import triplet_loss

slim = tf.contrib.slim


class TripletLossTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

  def test_mine_all_examples(self):
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss.mine_all_examples(distances)

    with self.test_session(graph=g) as sess:
      # Case 1.
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.ones([4, 4])})
      self.assertAllEqual(pos, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]))

      # Case 2.
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.ones([5, 5])})
      self.assertAllEqual(pos, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3,
            3, 3, 3, 4, 4, 4, 4]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0,
            1, 2, 4, 0, 1, 2, 3]))

  def test_mine_hard_examples(self):
    # Case 1.
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss.mine_hard_examples(distances, 1)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.array([
            [0, 1, 2, 3],
            [2, 0, 0, 3],
            [3, 1, 0, 0],
            [1, 3, 2, 0],
            ])})
      self.assertAllEqual(pos, np.array([0, 1, 2, 3]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 0]))

    # Case 2.
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss.mine_hard_examples(distances, 2)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.array([
            [0, 1, 2, 3],
            [2, 0, 0, 3],
            [3, 1, 0, 0],
            [1, 3, 2, 0],
            ])})
      self.assertAllEqual(pos, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
      self.assertAllEqual(neg, np.array([1, 2, 2, 0, 3, 1, 0, 2]))

    # Case 3.
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss.mine_hard_examples(distances, 3)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.array([
            [0, 1, 2, 3],
            [2, 0, 0, 3],
            [3, 1, 0, 0],
            [1, 3, 2, 0],
            ])})
      self.assertAllEqual(pos, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 2, 0, 3, 3, 1, 0, 0, 2, 1]))

if __name__ == '__main__':
  tf.test.main()

