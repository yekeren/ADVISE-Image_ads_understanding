import os
import sys
import re
import json
import string
import random

import numpy as np
#import tensorflow as tf

from nms_processor import NMSProcessor

def is_training_example(image_id):
  """Returns true if image_id belongs to training examples.

  Args:
    image_id: id of the image

  Returns:
    is_training_example: true if image_id belongs to training examples.
  """
  hash_id = hash(image_id) % 20
  if hash_id in range(12):
    return True
  return False


def is_validation_example(image_id):
  """Returns true if image_id belongs to validation examples.

  Args:
    image_id: id of the image

  Returns:
    is_validation_example: true if image_id belongs to validation examples.
  """
  hash_id = hash(image_id) % 20
  if hash_id in range(12, 16):
    return True
  return False


def is_testing_example(image_id):
  """Returns true if image_id belongs to validation examples.

  Args:
    image_id: id of the image

  Returns:
    is_validation_example: true if image_id belongs to validation examples.
  """
  hash_id = hash(image_id) % 20
  if hash_id in range(16, 20):
    return True
  return False


class AdsDatasetApi(object):
  def __init__(self):
    self._images_dict = None

    # Topics.
    self._name_to_topic = None
    self._topic_to_name = None
    self._topic_to_images = None
    self._topic_annotations = None

    # Regions annotated by mturkers.
    self._entity_annotations = None

    # Regions annotated by program.
    self._entity_annotations_ex = None

    # Densecap labels annotated by program.
    self._densecap_annotations = None

    # Q-A actions and reasons.
    self._action_reason_annotations = None

  @property
  def topic_to_name(self):
    return self._topic_to_name

  def init(self, images_dir=None, 
      topic_list_file=None, 
      topic_annot_file=None,
      entity_annot_file=None,
      entity_annot_file_ex=None,
      densecap_annot_file=None,
      qa_action_annot_file=None,
      qa_reason_annot_file=None,
      qa_action_reason_annot_file=None,
      qa_action_reason_padding=None):
    """Initialize ads dataset api.

    Args:
      images_dir: path containing 0-10 subdirs.
      topic_list_file: a file containing a list of topics.
      topic_annot_file: a file containing topics annotations.
      entity_annot_file: a file containing entity annotations.
      entity_annot_file_ex: a file containing entity annotations.
      qa_action_annot_file: a file containing qa action annotations.
      qa_reason_annot_file: a file containing qa reason annotations.
      qa_action_reason_annot_file: a file containing qa action-reason annotations.
    """
    self._index_images(images_dir)

    if topic_list_file is not None and topic_annot_file is not None:
      self._index_topics(topic_list_file, topic_annot_file)

    if entity_annot_file is not None:
      self._index_entities(entity_annot_file)

    if entity_annot_file_ex is not None:
      self._index_entities_ex(entity_annot_file_ex)

    if densecap_annot_file is not None:
      self._index_densecap(densecap_annot_file)

    if qa_action_annot_file is not None and qa_reason_annot_file is not None and qa_action_reason_annot_file is not None:
      self._index_action_reason_pairs(
          qa_action_annot_file, 
          qa_reason_annot_file,
          qa_action_reason_annot_file,
          qa_action_reason_padding)

  def get_meta_list(self):
    """Returns all meta info.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if image_id is invalid.
    """
    if self._images_dict is None:
      raise ValueError('images_dict is None.')
    return self._images_dict.values()

  def get_meta_list_by_ids(self, image_ids=[]):
    """Returns meta info based on image id.

    Args:
      image_ids: a list of image ids.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if image_id is invalid.
    """
    if self._images_dict is None:
      raise ValueError('images_dict is None.')

    meta_list = []
    for image_id in set(image_ids):
      if image_id not in self._images_dict:
        raise ValueError('image_id %s is invalid.' % (image_id))
      meta_list.append(self._images_dict[image_id])
    return meta_list

  def get_topic_names(self):
    """Returns all topic names.
      
    Returns:
      topic_names: a list of topic names.
    """
    if self._name_to_topic is None:
      raise ValueError('name_to_topic is None.')
    return sorted(self._name_to_topic.keys())

  def get_meta_list_by_topics(self, topic_name_list=[], min_votes=1):
    """Returns meta info list based on topics.

    Args:
      topic_name_list: topic names.
      min_votes: minimum votes for filtering out minorities.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if topic_name is invalid.
    """
    if topic_name_list == []:
      topic_name_list = [k for k in self._name_to_topic]

    # Extracts image ids.
    image_ids = set()
    for topic_name in topic_name_list:
      if topic_name not in self._name_to_topic:
        raise ValueError('Invalid topic name %s.' % (topic_name))
      topic_id = self._name_to_topic[topic_name]
      if topic_id not in self._topic_to_images:
        raise ValueError('Invalid topic id %s.' % (topic_id))

      image_ids.update(self._topic_to_images[topic_id])

    # Filters out minorities.
    meta_list = self.get_meta_list_by_ids(image_ids=image_ids)
    if min_votes > 1:
      meta_list = filter(lambda x: x['topic_votes'] >= min_votes, meta_list)
    return meta_list

  def get_meta_list_with_entity_annots(self):
    """Returns meta info list that contains entity annotations.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if entity_annotations is invalid.
    """
    if self._entity_annotations is None:
      raise ValueError('entity_annotations is None')
    return self._entity_annotations

  def get_meta_list_with_entity_annots_ex(self, score_threshold=0.0):
    """Returns meta info list that contains entity annotations.

    Args:
      score_threshold: the score threshold to filter meta info.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if entity_annotations is invalid.
    """
    if self._entity_annotations_ex is None:
      raise ValueError('entity_annotations_ex is None')
    meta_list = []
    for meta in self._entity_annotations_ex:
      meta['entities_ex'] = filter(
          lambda x: x['score'] >= score_threshold, meta['entities_ex'])
      meta_list.append(meta)
    return meta_list

  def get_meta_list_with_action_reason_annots(self):
    """Returns meta info list that contains action reason annotations.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if action_reason_annotations is invalid.
    """
    if self._action_reason_annotations is None:
      raise ValueError('action_reason_annotations is None')
    return self._action_reason_annotations

  def _index_images(self, images_dir=None):
    """Index images in the images_dir.

    Args:
      images_dir: root directory to be scanned.

    Returns:
      images_dict: a dictionary mapping id to image info.

    Raises:
      ValueError: if images_dir is invalid.
    """
    if images_dir is None:
      raise ValueError('images_dir is None.')

    if images_dir[-1] != '/':
      images_dir += '/'

    self._images_dict = {}
    for dir_name, sub_dir_list, file_list in os.walk(images_dir):
      for filename in file_list:
        if filename[-4:].lower() in ['.jpg', '.png']:
          filename = os.path.join(dir_name, filename)
          image_id = filename[len(images_dir):]
          self._images_dict.setdefault(image_id, {})
          self._images_dict[image_id].setdefault('image_id', image_id)
          self._images_dict[image_id].setdefault('filename', filename)

    return self._images_dict

  def _majority_vote(self, elems):
    """Process majority votes.

    Args:
      elems: a list of elems.

    Returns:
      elem: the element who gets the most votes.
      num_votes: number of votes.
    """
    votes = {}
    for e in elems:
      votes[e] = votes.get(e, 0) + 1
    votes = sorted(votes.iteritems(), lambda x, y: -cmp(x[1], y[1]))
    return votes[0]

  def _index_topics(self, topic_list_file, topic_annot_file):
    """Index topics for the images.

    Args:
      topic_list_file: a file containing a list of topics.
      topic_annot_file: a file containing topics annotations.
    
    Returns:
      name_to_topic: a dictionary mapping from topic_name to topic_id.
      topic_to_name: a dictionary mapping from topic_id to topic_name.
      topic_to_images: a dictionary mapping from topic_id to image_id_list

    Raises:
      ValueError: if either topic_list_file or topic_annot_file is invalid.
    """
    if topic_list_file is None:
      raise ValueError('topic_list_file is None.')

    if topic_annot_file is None:
      raise ValueError('topic_annot_file is None.')

    if self._images_dict is None:
      raise ValueError('images_dict is None.')

    def _revise_topic_id(topic_id):
      if not topic_id.isdigit():
        return None
      topic_id = int(topic_id)
      if topic_id == 39: topic_id = 0
      return topic_id

    def _revise_topic_name(name):
      matches = re.findall(r"\"(.*?)\"", name)
      if len(matches) > 1:
        return matches[1].lower()
      return matches[0].lower()

    self._topic_to_name = {}
    self._name_to_topic = {}
    with open(topic_list_file, 'r') as fp:
      for line in fp.readlines():
        topic_id, topic_name = line.strip('\n').split('\t') 

        topic_id = _revise_topic_id(topic_id)
        topic_name = _revise_topic_name(topic_name)
        assert topic_id is not None

        self._topic_to_name[topic_id] = topic_name
        self._name_to_topic[topic_name] = topic_id

    self._topic_to_images = {}
    with open(topic_annot_file, 'r') as fp:
      annots = json.loads(fp.read())
      for image_id, topic_id_list in annots.iteritems():
        meta = self._images_dict.get(image_id, None)
        if meta is None:
          raise ValueError('cannot find image with image_id %s' % (image_id))
        topic_id_list = [
          _revise_topic_id(tid) for tid in topic_id_list if _revise_topic_id(tid) in self._topic_to_name]

        if len(topic_id_list) > 0:
          topic_id, num_votes = self._majority_vote(topic_id_list)

          meta['topic_id'], meta['topic_votes'] = topic_id, num_votes
          meta['topic_name'] = self._topic_to_name[topic_id]
          image_ids = self._topic_to_images.setdefault(topic_id, [])
          image_ids.append(image_id)

    return self._name_to_topic, self._topic_to_name, self._topic_to_images

  def _index_entities(self, entity_annot_file):
    """Index entities for images.

    Args:
      entity_annot_file: a file containing entity annotations.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if input is invalid.
    """
    if entity_annot_file is None:
      raise ValueError('entity_annot_file is None.')

    if self._images_dict is None:
      raise ValueError('images_dict is None.')

    self._entity_annotations = []

    # Uses non maximum suppression to preprocess annotations.
    nms = NMSProcessor(max_output_size=10, iou_threshold=0.5)

    with open(entity_annot_file, 'r') as fp:
      annots = json.loads(fp.read())
      for image_id, entity_list in annots.iteritems():
        meta = self._images_dict.get(image_id, None)
        if meta is None:
          raise ValueError('cannot find image with image_id %s' % (image_id))

        # Preprocess annotations.
        boxes = []
        scores = []
        for i, entity in enumerate(entity_list):
          x1, y1, x2, y2 = [e / 500.0 for e in entity[:4]]
          if x1 > x2:
            x1, x2 = x2, x1
          if y1 > y2:
            y1, y2 = y2, y1
          boxes.append([y1, x1, y2, x2])
          scores.append(1.0)

        # Sort by area and process nms.
        area_func = lambda x: (x[2] - x[0]) * (x[3] - x[1])
        boxes = sorted(boxes, lambda x, y: cmp(area_func(x), area_func(y)))
        selected_boxes, _ = nms.process(np.array(boxes), np.array(scores))

        entities = []
        for box in selected_boxes:
          ymin, xmin, ymax, xmax = box.tolist()
          entities.append({
              'xmin': xmin,
              'ymin': ymin,
              'xmax': xmax,
              'ymax': ymax
              })
        meta['entities'] = entities
        self._entity_annotations.append(meta)
    return self._entity_annotations

  def _index_entities_ex(self, entity_annot_file):
    """Index entities for images.

    Args:
      entity_annot_file: a file containing entity annotations.

    Returns:
      meta_list: a list of meta info.

    Raises:
      ValueError: if input is invalid.
    """
    if entity_annot_file is None:
      raise ValueError('entity_annot_file is None.')

    if self._images_dict is None:
      raise ValueError('images_dict is None.')

    self._entity_annotations_ex = []
    with open(entity_annot_file, 'r') as fp:
      annots = json.loads(fp.read())

      for image_id, annot in annots.iteritems():
        meta = self._images_dict.get(image_id, None)
        if meta is None:
          raise ValueError('cannot find image with image_id %s' % (image_id))
        #entities = annot
        #for box, score in zip(annot['boxes'], annot['scores']):
        #  xmin, ymin, xmax, ymax = box
        #  entities.append({
        #      'score': score,
        #      'xmin': xmin,
        #      'ymin': ymin,
        #      'xmax': xmax,
        #      'ymax': ymax
        #      })
        meta['entities_ex'] = annot
        self._entity_annotations_ex.append(meta)
    return self._entity_annotations_ex

  def _index_densecap(self, densecap_annot_file):
    """Index densecap labels.

    Args:
      densecap_annot_file: path to the densecap annotation file.

    Raises:
      ValueError: if annotations file is invalid.
    """
    if densecap_annot_file is None:
      raise ValueError('densecap_annot_file is None.')

    if self._images_dict is None:
      raise ValueError('images_dict is None.')

    self._densecap_annotations = []
    with open(densecap_annot_file, 'r') as fp:
      annots = json.loads(fp.read())
      for image_id, annot in annots.iteritems():
        meta = self._images_dict.get(image_id, None)
        if meta is None:
          raise ValueError('cannot find image with image_id %s' % (image_id))
        meta['densecap_entities'] = annot['object']
        self._densecap_annotations.append(meta)
    return self._densecap_annotations

  def _index_action_reason_pairs(self, qa_action_annot_file,
      qa_reason_annot_file, qa_action_reason_annot_file,
      qa_action_reason_padding):
    """Index action reason pairs.

    Args:
      qa_action_annot_file: a file containing qa action annotations.
      qa_reason_annot_file: a file containing qa reason annotations.
      qa_action_reason_annot_file: a file containing qa action-reason annotations.

    Raises:
      ValueError: if annotations file is invalid.

    """
    _action_reason_annotations = []

    printable = set(string.printable)
    convert_to_printable = lambda caption: filter(lambda x: x in printable, caption)

    def _pad_and_add(meta, captions):
      captions = captions[:qa_action_reason_padding]
      if qa_action_reason_padding is None or len(captions) == qa_action_reason_padding:
        meta['action_reason_captions'] = captions

    def _process_combined_annots(qa_action_reason_annot_file):
      """Process combined annotations.

      Args:
        qa_action_reason_annot_file: a file containing qa action-reason annotations

      Returns:
        meta_list: a list containing meta info.

      Raises:
        ValueError: if annotations file is invalid.
      """
      # Read file.
      with open(qa_action_reason_annot_file, 'r') as fp:
        annots = json.loads(fp.read())

      # Parse annotations.
      meta_list = []
      for image_id, captions in annots.iteritems():
        meta = self._images_dict.get(image_id, None)
        if meta is None:
          raise ValueError('cannot find image with image_id %s' % (image_id))
        captions = [convert_to_printable(caption) for caption in captions]
        _pad_and_add(meta, captions)
        if 'action_reason_captions' in meta:
          meta_list.append(meta)
      print >> sys.stderr, 'Load %d combined QA annotations.' % (len(meta_list))
      return meta_list

    def _revise_action_reason(action, reason):
      if action[-1] == '.':
        action = action[:-1]
      if reason[:len('Because')] == 'Because':
        reason = 'because' + reason[len('Because'):]

      caption = convert_to_printable(action + ' ' + reason)
      return caption

    def _process_seperate_annots(qa_action_annot_file, qa_reason_annot_file):
      """Process seperate annotations.

      Args:
        qa_action_annot_file: a file containing qa action annotations.
        qa_reason_annot_file: a file containing qa reason annotations.

      Returns:
        meta_list: a list containing meta info.

      Raises:
        ValueError: if annotations file is invalid.
      """
      # Read files.
      with open(qa_action_annot_file, 'r') as fp:
        actions = sorted(json.loads(fp.read()).iteritems(), 
            lambda x, y: cmp(x[0], y[0]))
      with open(qa_reason_annot_file, 'r') as fp:
        reasons = sorted(json.loads(fp.read()).iteritems(),
            lambda x, y: cmp(x[0], y[0]))
      if len(actions) != len(reasons):
        raise ValueError('Action annots do not match to the reason annots.')

      # Prase annotations.
      meta_list = []
      for (action_id, action_annots), (reason_id, reason_annots) in zip(actions, reasons):
        if action_id != reason_id or len(action_annots) != len(reason_annots):
          raise ValueError('Action annots do not match to the reason annots.')
        image_id = action_id
        meta = self._images_dict.get(image_id, None)
        if meta is None:
          raise ValueError('cannot find image with image_id %s' % (image_id))

        # Process captions.
        captions = []
        for action, reason in zip(action_annots, reason_annots):
          action, reason = action.strip(), reason.strip()
          if action and reason:
            captions.append(_revise_action_reason(action, reason))
        _pad_and_add(meta, captions)
        if 'action_reason_captions' in meta:
          meta_list.append(meta)

      print >> sys.stderr, 'Load %d seperated QA annotations.' % (len(meta_list))
      return meta_list

    meta_list = []
    meta_list.extend(_process_combined_annots(qa_action_reason_annot_file))
    meta_list.extend(_process_seperate_annots(qa_action_annot_file, qa_reason_annot_file))
    print >> sys.stderr, 'Load %d QA annotations.' % (len(meta_list))
    self._action_reason_annotations = meta_list
    return meta_list

  def sample_negative_action_reason_captions(self, negative_examples_per_image=10):
    """Randomly sample negative action reason captions.

    Args:
      negative_examples_per_image: number of negative examples.

    Raises:
      ValueError: if action_reason_annotations is invalid.
    """
    if self._action_reason_annotations is None:
      raise ValueError('action_reason_annotations is None.')

    def _sample_negatives(meta_list, negative_examples_per_image):
      """Sample negative examples.

      Args:
        meta_list: a list containing meta info.
        negative_examples_per_image: number of negative examples.
      """
      random.seed(42)
      for i, meta in enumerate(meta_list):
        captions = []
        for rep in xrange(negative_examples_per_image):
          index = random.randint(1, len(meta_list) - 1)
          caption_list = meta_list[index]['action_reason_captions']
          index = random.randint(0, len(caption_list) - 1)
          captions.append(caption_list[index])
        meta['action_reason_captions_neg'] = captions

    for split_fn in [is_training_example, is_validation_example, is_testing_example]:
      meta_list = filter(lambda meta: split_fn(meta['image_id']), self._action_reason_annotations)
      _sample_negatives(meta_list, negative_examples_per_image)
