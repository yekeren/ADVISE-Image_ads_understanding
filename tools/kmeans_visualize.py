from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import logging
import numpy as np

from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner

from utils.ads_dataset_api import is_training_example
from utils.ads_dataset_api import is_validation_example

from utils import ads_dataset_api
from utils import vis

flags = tf.app.flags
flags.DEFINE_string('images_dir', '', 'Path to ads dataset')
flags.DEFINE_string('entity_annot_path', '', 'File path to entity annotations.')
flags.DEFINE_float('score_threshold', 0.5, 'Threshold for filtering out proposals')

flags.DEFINE_string('data_path', '', 'Path to the feature data file.')
flags.DEFINE_string('model_path', '', 'Path to the model dir.')
flags.DEFINE_string('output_path', '', 'Path to the output html file.')
flags.DEFINE_integer('num_patches_per_class', 20, 'Number of patches visualized in a row.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _get_data(raw_data):
  examples = []
  patches = []
  for k, v in raw_data.iteritems():
    if is_validation_example(k):
      examples.append(v['entity_emb_list'])
      for i in xrange(len(v['entity_emb_list'])):
        patches.append((k, i))

  logging.info('Loaded %s images.', len(examples))
  examples = np.concatenate(examples, axis=0)
  logging.info('Loaded %s patches.', len(examples))

  return examples, patches

def main(_):
  logging.basicConfig(level=logging.DEBUG)

  # Read data file.
  raw_data = np.load(FLAGS.data_path).item()
  examples, patches = _get_data(raw_data)

  assert len(examples) == len(patches)

  kmeans = joblib.load(FLAGS.model_path)
  labels = kmeans.predict(examples)

  # Load ads data.
  api = ads_dataset_api.AdsDatasetApi()
  api.init(images_dir=FLAGS.images_dir,
      entity_annot_file_ex=FLAGS.entity_annot_path)

  # Create visualization dict.
  vis_dict = {}
  for patch, label in zip(patches, labels):
    vis_dict.setdefault(label, []).append(patch)

  # Visualize visual words.
  html = ''
  html += '<table border=1>'
  for vis_id, patches in vis_dict.iteritems():
    logging.info('On visual word %d.', vis_id)
    patches = patches[:FLAGS.num_patches_per_class]

    html += '<tr id="%s">' % (vis_id)
    html += '<td><a href="#%d">%d</a></td>' % (vis_id, vis_id)
    for image_id, entity_id in patches:
      meta = api.get_meta_list_by_ids(image_ids=[image_id])[0]
      image = vis.image_load(meta['filename'])
      box = meta['entities_ex'][entity_id]
      box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
      patch = vis.image_crop_and_resize(image, box, crop_size=(160, 160))

      html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
          vis.image_uint8_to_base64(patch))
    html += '</tr>'

  html += '</table>'
  with open(FLAGS.output_path, 'w') as fp:
    fp.write(html)

if __name__ == '__main__':
  tf.app.run()
