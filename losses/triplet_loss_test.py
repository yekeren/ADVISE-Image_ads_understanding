
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from losses import triplet_loss

slim = tf.contrib.slim


class TripletLossTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

  def test1(self):
    g = tf.Graph()
    with g.as_default():
      labels = tf.placeholder(shape=[4], dtype=tf.int32)
      positive_masks = tf.sparse_to_dense(
          tf.stack([tf.range(4), labels], 1),
          output_shape=tf.stack([4, 10]),
          sparse_values=True, 
          default_value=False, 
          validate_indices=True)

    with self.test_session(graph=g) as sess:
      mat = sess.run([positive_masks], feed_dict={
          labels: np.array([0, 1, 2, 3])})
      print(mat)


  def test_mine_random_examples(self):
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss._mine_random_examples(distances, 4)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.ones([4, 4])})
      self.assertEqual(pos.shape, (16,))
      self.assertEqual(neg.shape, (16,))
      for i in xrange(16):
        self.assertNotEqual(pos[i], neg[i])

  def test_mine_all_examples(self):
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss._mine_all_examples(distances)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.ones([4, 4])})
      self.assertAllEqual(pos, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]))

      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.ones([5, 5])})
      self.assertAllEqual(pos, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3,
            3, 3, 3, 4, 4, 4, 4]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0,
            1, 2, 4, 0, 1, 2, 3]))

  def test_mine_semi_hard_examples(self):
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss._mine_semi_hard_examples(distances)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={distances: np.array([
            [0, 1, 2, 3],
            [2, 0, 0, 3],
            [3, 1, 0, 0],
            [1, 3, 2, 0],
            ])})
      self.assertAllEqual(pos, np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 0, 3, 0, 1, 0, 1, 2]))

  def test_mine_hard_examples(self):
    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss._mine_hard_examples(distances, 1)

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

    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss._mine_hard_examples(distances, 2)

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

    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss._mine_hard_examples(distances, 3)

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

    g = tf.Graph()
    with g.as_default():
      distances = tf.placeholder(shape=[None, None], dtype=tf.float32)
      pos_indices, neg_indices = triplet_loss._mine_hard_examples(distances, 10)

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
