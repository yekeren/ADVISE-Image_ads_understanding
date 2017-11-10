import numpy as np
import tensorflow as tf

class NMSProcessor(object):
  """Helper class that process non maximum suppression on single image."""

  def __init__(self, max_output_size, iou_threshold):
    """Init.

    Args:
      max_output_size: maximum number of boxes to maintain.
      iou_threshold: threhold for intersection over union.
    """
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    self._sess = tf.Session(config=config)
    self._boxes = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    self._scores= tf.placeholder(dtype=tf.float32, shape=[None])

    self._selected = tf.image.non_max_suppression(
        self._boxes, self._scores, max_output_size, iou_threshold)
    self._selected_boxes = tf.gather(self._boxes, self._selected)
    self._selected_scores = tf.gather(self._scores, self._selected)

  def process(self, boxes, scores):
    """Process non maximum suppression.

    Args:
      boxes: a [num_boxes, 4] np array.
      scores: a [num_boxes] np array.

    Returns:
      selected_boxes: a [num_selected_boxes, 4] np array.
    """
    return self._sess.run([
        self._selected, self._selected_boxes, self._selected_scores],
        feed_dict={self._boxes: boxes, self._scores: scores})
