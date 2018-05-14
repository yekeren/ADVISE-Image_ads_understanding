
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import sqlite3
import io
import json
import os.path
import numpy as np
import random
import cv2

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from utils import image_coder
from utils.nms_processor import NMSProcessor

flags = tf.app.flags
flags.DEFINE_string('symbol_annot_path', '', 'Path to the symbol annotation file.')
flags.DEFINE_string('img_dir', '', 'Directory to store images.')
flags.DEFINE_string('vis_dir', '', 'Directory to store visualization results.')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('train_output_path', '', 'Directory to output TFRecord')
flags.DEFINE_string('valid_output_path', '', 'Directory to output TFRecord')
flags.DEFINE_string('test_output_path', '', 'Directory to output TFRecord')
flags.DEFINE_integer('max_image_size', 500, 'Maximum height/widht of the image.')
flags.DEFINE_integer('number_of_val_images', 2000, 'Number of validation images.')

FLAGS = flags.FLAGS
coder = image_coder.ImageCoder()


def _read_examples_list():
  """Read list of training or validation examples.

  Returns:
    examples: list of data examples.
  """
  with open(FLAGS.symbol_annot_path) as fp:
    annots = json.loads(fp.read())

  examples = []
  nms = NMSProcessor(max_output_size=10, iou_threshold=0.5)

  for image_id, regions in annots.iteritems():
    # Preprocess annotations.
    boxes, scores = [], []
    for i, entity in enumerate(regions):
      x1, y1, x2, y2 = [x / 500.0 for x in entity[:4]]
      if x1 > x2: x1, x2 = x2, x1
      if y1 > y2: y1, y2 = y2, y1
      boxes.append([y1, x1, y2, x2])
      scores.append(1.0)

    # Sort by area and process nms.
    area_func = lambda x: (x[2] - x[0]) * (x[3] - x[1])
    boxes = sorted(boxes, lambda x, y: cmp(area_func(x), area_func(y)))
    _, selected_boxes, _ = nms.process(np.array(boxes), np.array(scores))

    regions = []
    for box in selected_boxes:
      ymin, xmin, ymax, xmax = box.tolist()
      regions.append({
          'name': 'region',
          'bndbox': { 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax },
          })
    examples.append({
        'image_id': image_id,
        'filename': os.path.join(FLAGS.img_dir, image_id),
        'object': regions
        })

  return examples


def _visualize(examples_list, vis_dir):
  """Visualize examples.

  Args:
    examples_list: a list of examples returned by _read_examples_list.
    vis_dir: directory to store output images.
  """
  for i, example in enumerate(examples_list):
    image = cv2.imread(example['filename'])
    height, width, _ = image.shape
    for obj in example['object']:
      cv2.rectangle(image, 
          (int(obj['bndbox']['xmin'] * width), int(obj['bndbox']['ymin'] * height)),
          (int(obj['bndbox']['xmax'] * width), int(obj['bndbox']['ymax'] * height)),
          (0, 255, 0), thickness=2)
    filename = os.path.join(vis_dir, '%d.jpg' % (i))
    cv2.imwrite(filename, image)


def dict_to_tf_example(data, label_map_dict):
  """Convert dictionary to tf.Example proto.

  Args:
    data: dict holding image and bounding box information.
    label_map_dict: a dict mapping from string label names to integer ids.
    data_dir: directory to caltech webfaces dataset.

  Returns:
    example: converted tf.Example

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = data['filename']
  encoded_data = tf.gfile.FastGFile(full_path, 'r').read()
  try:
    if full_path[-4:].lower() == '.png':
      image = coder.decode_png(encoded_data)
    else:
      image = coder.decode_jpeg(encoded_data)
  except Exception as ex:
    tf.logging.info('failed to decode %s.', full_path)
    return None

  height, width, channels = image.shape

  # Resize image if necessary.
  if height > FLAGS.max_image_size or width > FLAGS.max_image_size:
    ratio = min(
        float(FLAGS.max_image_size) / height,
        float(FLAGS.max_image_size) / width)
    image = cv2.resize(image, (int(width * ratio), int(height * ratio)))

  encoded_data = coder.encode_jpeg(image)
  key = hashlib.sha256(encoded_data).hexdigest()

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  for obj in data['object']:
    xmin.append(obj['bndbox']['xmin'])
    ymin.append(obj['bndbox']['ymin'])
    xmax.append(obj['bndbox']['xmax'])
    ymax.append(obj['bndbox']['ymax'])
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])

  xmin = [min(max(e, 0), 1) for e in xmin]
  xmax = [min(max(e, 0), 1) for e in xmax]
  ymin = [min(max(e, 0), 1) for e in ymin]
  ymax = [min(max(e, 0), 1) for e in ymax]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_data),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example


def create_tf_record(output_filename, label_map_dict, examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 200 == 0:
      tf.logging.info('On image %d of %d', idx, len(examples))
    tf_example = dict_to_tf_example(example, label_map_dict)
    if tf_example is not None:
      writer.write(tf_example.SerializeToString())
  writer.close()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  examples_list = _read_examples_list()

  tf.logging.info('Load %s records.', len(examples_list))
  _visualize(examples_list[:20], FLAGS.vis_dir)

  # Split dataset into train/val.
  if not FLAGS.test_output_path:
    random.seed(286)
    random.shuffle(examples_list)
    train_examples = examples_list[FLAGS.number_of_val_images:]
    val_examples = examples_list[:FLAGS.number_of_val_images]
    tf.logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    train_output_path = FLAGS.train_output_path
    val_output_path = FLAGS.valid_output_path
    create_tf_record(train_output_path, label_map_dict, train_examples)
    create_tf_record(val_output_path, label_map_dict, val_examples)

  else:
    tf.logging.info('%d test examples.', len(examples_list))
    test_output_path = FLAGS.test_output_path
    create_tf_record(test_output_path, label_map_dict, examples_list)

  tf.logging.info('Done')

if __name__ == '__main__':
  tf.app.run()
