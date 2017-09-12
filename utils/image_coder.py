import numpy as np
import tensorflow as tf

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    self._sess = tf.Session()

    self._encoded_data = tf.placeholder(dtype=tf.string)
    self._image_data = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    self._decode_png = tf.image.decode_png(self._encoded_data, channels=3)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_data, channels=3)

    self._encode_jpeg = tf.image.encode_jpeg(
        self._image_data, format='rgb', quality=100)

  def encode_jpeg(self, image):
    """Encode image data into jpeg encoded string.

    Args:
      image: a [None, None, 3] np arrray representing an image.

    Returns:
      encoded_data: encoded jpeg string.
    """
    return self._sess.run(self._encode_jpeg, 
        feed_dict={self._image_data: image})

  def decode_png(self, encoded_data):
    """Decode png encoded string into image data.

    Args:
      encoded_data: encoded png string.

    Returns:
      image: a [None, None, 3] np array representing an image.
    """
    image = self._sess.run(self._decode_png, 
        feed_dict={self._encoded_data: encoded_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_jpeg(self, encoded_data):
    """Decode jpeg encoded string into image data.

    Args:
      encoded_data: encoded jpeg string.

    Returns:
      image: a [None, None, 3] np array representing an image.
    """
    image = self._sess.run(self._decode_jpeg, 
        feed_dict={self._encoded_data: encoded_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image
