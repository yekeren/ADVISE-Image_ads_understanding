import os
import json
import argparse

def _enum_image_ids(image_path):
  """Enumerates image ids.

  Args:
    image_path: path to the image directory.

  Returns:
    image_ids: a list of image ids.
  """
  image_ids = []
  for dir_name, sub_dir_list, file_list in os.walk(image_path):
    for filename in file_list:
      if filename[-4:].lower() in ['.jpg', '.png']:
        image_id = os.path.join(dir_name, filename)[len(image_path):]
        image_ids.append(image_id)
  return sorted(image_ids)


def _filter(image_ids, split_id):
  """Filter to get train / valid / test set.

  Args:
    image_ids: a list of image ids.
    split_id: a integer split id. 

  Returns:
    train_ids: a list of image ids for training.
    valid_ids: a list of image ids for validation.
    test_ids: a list of image ids for testing.
  """
  train_ids = [x for x in image_ids if (hash(x) + split_id) % 10 in xrange(6)]
  valid_ids = [x for x in image_ids if (hash(x) + split_id) % 10 in xrange(6, 8)]
  test_ids = [x for x in image_ids if (hash(x) + split_id) % 10 in xrange(8, 10)]
  return train_ids, valid_ids, test_ids


def main(args):
  """Main."""
  output_path = args.output_path
  image_path = args.image_path
  if image_path[-1] != '/':
    image_path += '/'

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # Enumerate image ids.
  image_ids = _enum_image_ids(image_path)

  # Split data.
  for split_id in xrange(10):
    train_ids, valid_ids, test_ids = _filter(image_ids, split_id)
    assert len(set(train_ids) & set(valid_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(valid_ids) & set(test_ids)) == 0
    assert len(train_ids) + len(valid_ids) + len(test_ids) == len(image_ids)

    with open(os.path.join(output_path, 'train_%d.txt' % (split_id)), 'w') as fp:
      fp.write('\n'.join(train_ids))
    with open(os.path.join(output_path, 'valid_%d.txt' % (split_id)), 'w') as fp:
      fp.write('\n'.join(valid_ids))
    with open(os.path.join(output_path, 'test_%d.txt' % (split_id)), 'w') as fp:
      fp.write('\n'.join(test_ids))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--image_path', default='raw_data/ads/images', 
      help='Path to the image directory.')
  parser.add_argument('--output_path', default='raw_data/ads/split/', 
      help='Path to store output split files.')

  args = parser.parse_args()
  print(json.dumps(vars(args), indent=2))

  main(args)
