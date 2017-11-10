import os
import sys
import json

import cv2
import numpy as np
import tensorflow as tf

from utils import ads_api
from utils import vis

flags = tf.app.flags
flags.DEFINE_string('ads_config', 'configs/ads_api.empty.config', 'Path to ads dataset config file.')
flags.DEFINE_string('object_proposals', 'raw_data/ads/annotations/coco_objects.json', 'Path to object proposal json file.')
flags.DEFINE_string('output_html', 'output/ads_proposals.html', 'Path to output html file.')
flags.DEFINE_integer('max_vis_examples', 100, 'Number of examples to visualize.')
flags.DEFINE_integer('max_vis_patches', 10, 'Number of patches per image to visualize.')
flags.DEFINE_float('score_threshold', 0.0, 'Threshold for filtering out low confident proposals.')
flags.DEFINE_integer('vis_image_size', 300, 'Image size.')
flags.DEFINE_integer('vis_patch_size', 150, 'Patch size.')

FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Load objects annotations.
  with open(FLAGS.object_proposals, 'r') as fp:
    proposal_data = json.loads(fp.read())

  # Read and visualize ads images.
  api = ads_api.AdsApi(FLAGS.ads_config)
  meta_list = api.get_meta_list()[:FLAGS.max_vis_examples]

  image_size = FLAGS.vis_image_size
  patch_size = FLAGS.vis_patch_size

  scale_fn = lambda x: int(x * image_size)

  html = ''
  html += '<table border=1>'
  html += '<tr>'
  html += '<th>INPUT IMAGE</th>'
  html += '<th>IMAGE WITH BOUNDING BOX</th>'
  for i in xrange(FLAGS.max_vis_patches):
    html += '<th>PATCH %d</th>' % (i)
  html += '</tr>'

  for meta in meta_list:
    image_data = vis.image_load(meta['file_path'])
    height, width, _ = image_data.shape

    #if not meta['image_id'] in proposal_data:
    #  continue

    html += '<tr>'

    # Images.
    image_resized = cv2.resize(image_data, (image_size, image_size))
    image_resized_with_annot = np.copy(image_resized)

    html += '<td><img src="data:image/jpg;base64,%s"></td>' % (
        vis.image_uint8_to_base64(
          image_resized, 
          disp_size=(image_size, image_size)))

    # Draw bounding boxes.
    objects = proposal_data[meta['image_id']][:FLAGS.max_vis_patches]

    for obj in objects:
      x1, y1, x2, y2 = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
      vis.image_draw_bounding_box(
          image_resized_with_annot, 
          (x1, y1, x2, y2),
          color=(0, 255, 0))
      vis.image_draw_text(
          image_resized_with_annot, 
          (x1, y1 + 0.01), 
          '%.2lf' % (obj['score']),
          color=(0, 0, 0), 
          bkg_color=(0, 255, 0))

    html += '<td><img src="data:image/jpg;base64,%s"></td>' % ( 
        vis.image_uint8_to_base64(
          image_resized_with_annot, 
          disp_size=(image_size, image_size)))

    # Draw cropped patches.
    count = 0
    for obj in objects:
      count += 1
      x1, y1, x2, y2 = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
      caption = obj.get('caption', '')
      if not caption:
        caption = obj.get('object_name', '')
      caption = caption.replace('<UNK>', 'UNK')

      roi_image = vis.image_crop_and_resize(
          image_data,
          (x1, y1, x2, y2),
          (patch_size, patch_size))
      html += '<td><img src="data:image/jpg;base64,%s"></br>%.2lf</br>%s</td>' % (
          vis.image_uint8_to_base64(roi_image),
          obj['score'], 
          caption)

    for i in xrange(count, FLAGS.max_vis_patches):
      html += '<td></td>'

    html += '</tr>'
  html += '</table>'

  with open(FLAGS.output_html, 'w') as fp:
    fp.write(html)

if __name__ == '__main__':
  tf.app.run()
