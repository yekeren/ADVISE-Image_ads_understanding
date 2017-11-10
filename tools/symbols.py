
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
import numpy as np
import tensorflow as tf

from utils.ads_api import AdsApi
import random

from utils import vis

flags = tf.app.flags
flags.DEFINE_string('ads_config', 
    'configs/ads_api.config.0', 'Directory to ads dataset config file.')

FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Read data file.
  api = AdsApi(FLAGS.ads_config, invalid_items=['densecap_annotations'])

  # Create vocabulary.
  meta_list = api.get_meta_list()

  results = {}
  for meta in meta_list:
    if 'symbol_names' in meta and 'statements' in meta:
      for statement in meta['statements']:
        for name in meta['symbol_names']:
          results.setdefault(name, []).append(
              (meta['image_id'], meta['symbol_names'], statement))

  for symbol, statements in results.iteritems():
    random.seed(286)
    random.shuffle(statements)
    statements = statements[:200]
    print(symbol)
    with open('ads_symbols/statements/%s.txt' % (symbol), 'w') as fp:
      for image_id, symbols, statement in statements:
        fp.write('%s\t%s\t%s\n' % (image_id, ' '.join(symbols), statement))

    with open('ads_symbols/statements/%s.html' % (symbol), 'w') as fp:
      fp.write('<table border=1>')
      for image_id, symbols, statement in statements[:20]:
        fp.write('<tr>')
        image = vis.image_load('raw_data/ads/images/%s' % (image_id))
        image_str = vis.image_uint8_to_base64(image, disp_size=(200, 200))
        fp.write('<td><img src="data:image/jpg;base64,%s"></td>' % (image_str))
        fp.write('<td>%s</td>' % (' '.join(symbols)))
        fp.write('<td>%s</td>' % (statement))
        fp.write('</tr>')
      fp.write('</table>')

if __name__ == '__main__':
  tf.app.run()
