import os
import sys
import re
import json
import string
import random

import numpy as np

printable = set(string.printable)
convert_to_printable = lambda caption: filter(lambda x: x in printable, caption)

class AdsApi(object):
  def __init__(self, conf, invalid_items=[]):
    """Initialize api from config file.

    Args:
      config: a config file in json format.
    """
    with open(conf, 'r') as fp:
      config = json.loads(fp.read())

    for item in invalid_items:
      print >> sys.stderr, 'Ignore config item: %s.' % (item)
      del config[item]

    train_ids, valid_ids, test_ids = self._get_splits(
        config['train_ids'], config['valid_ids'], config['test_ids'])

    # Initialize meta info.
    assert len(set(train_ids) & set(valid_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(valid_ids) & set(test_ids)) == 0

    self._meta = {}
    self._meta.update([(x, {'split': 'train'}) for x in train_ids])
    self._meta.update([(x, {'split': 'valid'}) for x in valid_ids])
    self._meta.update([(x, {'split': 'test'}) for x in test_ids])
    for image_id in self._meta:
      self._meta[image_id].update({
        'image_id': image_id,
        'file_path': os.path.join(config['image_path'], image_id),
      })
    print >> sys.stderr, 'Load %d examples.' % (len(self._meta))

    if 'topic_list' in config and 'topic_annotations' in config:
      self._process_topic(config['topic_list'], config['topic_annotations'])

    if 'symbol_list' in config and 'symbol_annotations' in config:
      self._process_symbol(config['symbol_list'], config['symbol_annotations'])

    if 'entity_annotations' in config:
      self._process_entity(config['entity_annotations'])

    if 'detection_annotations' in config:
      self._process_detection(config['detection_annotations'])

    if 'qa_action_annotations' in config and 'qa_reason_annotations' in config:
      self._process_seperate_action_reason(
          config['qa_action_annotations'],
          config['qa_reason_annotations'],
          config.get('num_positive_statements', None))

    if 'qa_action_reason_annotations' in config:
      self._process_combined_action_reason(
          config['qa_action_reason_annotations'],
          config.get('num_positive_statements', None))

    if 'num_negative_statements' in config:
      self._sample_negative_statements(
          config['num_negative_statements'])
      self._sample_hard_negative_statements(
          config['num_negative_statements'])

    if 'slogan_annotations' in config:
      self._process_slogan(
          config['slogan_annotations'],
          config.get('num_positive_slogans', None))

    if 'num_negative_slogans' in config:
      self._sample_negative_slogans(
          config['num_negative_slogans'])

    if 'densecap_annotations' in config:
      self._process_densecap(config['densecap_annotations'])

    self._summerize()

  def get_meta_by_id(self, image_id):
    """Returns meta info based on image_id.

    Args:
      image_id: string image_id.

    Returns:
      meta: meta info.
    """
    return self._meta[image_id]

  def get_meta_list(self, split=None):
    """Get meta list.

    Args:
      split: could be one of 'train', 'valid', 'test'.

    Returns:
      meta_list: meta list for the specific split.
    """
    meta_list = self._meta.values()
    
    if split is not None:
      assert split in ['train', 'valid', 'test']
      meta_list = [x for x in meta_list if x['split'] == split]

    return meta_list

  def get_topic_to_name(self):
    """Get the mapping from topic to name.

    Returns:
      topic_to_name: a mapping from topic to name.
    """
    return self._topic_to_name

  def get_symbol_to_name(self):
    """Get the mapping from symbol to name.

    Returns:
      symbol_to_name: a mapping from symbol to name.
    """
    return self._symbol_to_name

  def _summerize(self):
    """Print statistics to the standard error.
    """
    meta_list = self.get_meta_list()
    num_topic_annots = len([x for x in meta_list if 'topic_id' in x])
    print >> sys.stderr, '%d examples associate with topics.' % (num_topic_annots)

    num_slogan_annots = len([x for x in meta_list if 'slogans' in x])
    print >> sys.stderr, '%d examples associate with slogans.' % (num_slogan_annots)

    num_neg_slogan_annots = len([x for x in meta_list if 'negative_slogans' in x])
    print >> sys.stderr, '%d examples associate with negative slogans.' % (num_neg_slogan_annots)

    num_stmt_annots = len([x for x in meta_list if 'statements' in x])
    print >> sys.stderr, '%d examples associate with statements.' % (num_stmt_annots)

    num_neg_stmt_annots = len([x for x in meta_list if 'negative_statements' in x])
    print >> sys.stderr, '%d examples associate with negative statements.' % (num_neg_stmt_annots)

    num_hard_neg_stmt_annots = len([x for x in meta_list if 'hard_negative_statements' in x])
    print >> sys.stderr, '%d examples associate with hard negative statements.' % (num_hard_neg_stmt_annots)

    num_qa_annots = len([x for x in meta_list if 'questions' in x])
    print >> sys.stderr, '%d examples associate with question-answer pairs.' % (num_qa_annots)

    num_densecap_annots = len([x for x in meta_list if 'densecap_objects' in x])
    print >> sys.stderr, '%d examples associate with densecap annotations.' % (num_densecap_annots)

    num_symb_annots = len([x for x in meta_list if 'symbol_ids' in x])
    print >> sys.stderr, '%d examples associate with symbols.' % (num_symb_annots)

  def _get_splits(self, train_ids_file, valid_ids_file, test_ids_file):
    """Get splits from pre-partitioned file.

    Args:
      train_ids_file: file containing train ids.
      valid_ids_file: file containing valid ids.
      test_ids_file: file containing test ids.
    """
    with open(train_ids_file, 'r') as fp:
      train_ids = [x.strip() for x in fp.readlines()]
    with open(valid_ids_file, 'r') as fp:
      valid_ids = [x.strip() for x in fp.readlines()]
    with open(test_ids_file, 'r') as fp:
      test_ids = [x.strip() for x in fp.readlines()]

    print >> sys.stderr, 'Load %d train examples.' % (len(train_ids))
    print >> sys.stderr, 'Load %d valid examples.' % (len(valid_ids))
    print >> sys.stderr, 'Load %d test examples.' % (len(test_ids))
    return train_ids, valid_ids, test_ids

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

  def _process_detection(self, detection_annotations):
    """Process detection annotations.

    Modify 'objects' data field.

    Args:
      detection_annotations: path to the detection json file.
    """
    with open(detection_annotations, 'r') as fp:
      annots = json.loads(fp.read())
      
      for image_id, annot in annots.iteritems():
        meta = self._meta[image_id]
        meta['objects'] = annot

  def _process_entity(self, entity_annotations):
    """Process entity annotations.

    Modify 'entities' data field.

    Args:
      entity_annotations: path to the symbol json file.
    """
    with open(entity_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    for image_id, entity_list in annots.iteritems():
      # Preprocess annotations.
      meta = self._meta[image_id]
      boxes = []
      scores = []
      for i, entity in enumerate(entity_list):
        x1, y1, x2, y2 = [x / 500.0 for x in entity[:4]]
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
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
            'score': 1.0,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
            })
      meta['entities'] = entities

    # Uses non maximum suppression to preprocess annotations.
    from utils.nms_processor import NMSProcessor
    nms = NMSProcessor(max_output_size=10, iou_threshold=0.5)

    for image_id, entity_list in annots.iteritems():
      # Preprocess annotations.
      meta = self._meta[image_id]
      boxes = []
      scores = []
      for i, entity in enumerate(entity_list):
        x1, y1, x2, y2 = [x / 500.0 for x in entity[:4]]
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
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
            'score': 1.0,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
            })
      meta['entities'] = entities

  def _process_topic(self, topic_list, topic_annotations):
    """Process topic annotations.

    Modifies 'topic_id' data field in the meta data.
    Modifies 'topic_name' data field in the meta data.

    Args:
      topic_list: path to the file storing topic list.
      topic_annotations: path to the topic json file.
    """

    def _revise_topic_id(topic_id):
      """Revises topic id.
      Args:
        topic_id: topic id in string format.
      Returns:
        topic_id: topic id in number format.
      """
      if not topic_id.isdigit():
        return None
      topic_id = int(topic_id)
      if topic_id == 39: topic_id = 0
      return topic_id

    def _revise_topic_name(name):
      """Revises topic name.
      Args:
        topic_name: topic name in long description format.
      Returns:
        topic_name: short topic name.
      """
      matches = re.findall(r"\"(.*?)\"", name)
      if len(matches) > 1:
        return matches[1].lower()
      return matches[0].lower()

    self._topic_to_name = {}
    with open(topic_list, 'r') as fp:
      for line in fp.readlines():
        topic_id, topic_name = line.strip('\n').split('\t') 

        topic_id = _revise_topic_id(topic_id)
        topic_name = _revise_topic_name(topic_name)
        if topic_id is None:
          raise ValueError('Invalid topic id %s.' % (topic_id))
        self._topic_to_name[topic_id] = topic_name

    with open(topic_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    for image_id, topic_id_list in annots.iteritems():
      meta = self._meta[image_id]
      topic_id_list = [_revise_topic_id(tid) for tid in topic_id_list]
      topic_id_list = [tid for tid in topic_id_list if tid is not None]

      if len(topic_id_list) > 0:
        topic_id, num_votes = self._majority_vote(topic_id_list)

        meta['topic_id'], meta['topic_votes'] = topic_id, num_votes
        meta['topic_name'] = self._topic_to_name[topic_id]

  def _process_symbol(self, symbol_list, symbol_annotations):
    """Process symbol annotations.

    Modifies 'symbol_ids' data field in the meta data.
    Modifies 'symbol_names' data field in the meta data.

    Args:
      symbol_list: path to the file storing symbol list.
      symbol_annotations: path to the symbol json file.
    """
    with open(symbol_list, 'r') as fp:
      data = json.loads(fp.read())

    self._symbol_to_name = {0: 'unclear'}
    symbol_to_id = {}
    for cluster in data['data']:
      self._symbol_to_name[cluster['cluster_id']] = cluster['cluster_name']
      for symbol in cluster['symbols']:
        symbol_to_id[symbol] = cluster['cluster_id']

    with open(symbol_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    for image_id, objects in annots.iteritems():
      meta = self._meta[image_id]
      symbol_list = []
      for obj in objects:
        symbols = [s.strip() for s in obj[4].lower().split('/') if len(s.strip()) > 0]
        symbols = [symbol_to_id[s] for s in symbols if s in symbol_to_id]
        symbol_list.extend(symbols)

      if len(symbol_list) > 0:
        symbol_list = list(set(symbol_list))
        meta['symbol_ids'] = symbol_list
        meta['symbol_names'] = [self._symbol_to_name[s] for s in symbol_list]

  def _revise_action_reason_to_statement(self, action, reason):
    """Revises action and reason to statement.

    Args:
      action: a string similar to "I should buy a car".
      reason: a string similar to "Because it is stable and strong.".

    Returns:
      statement: a string similar to "I should buy a car because it is stable
      and strong".
    """
    action, reason = action.strip().strip('.'), reason.strip().strip('.')
    if not action or not reason:
      return None

    if reason[:len('Because')].lower() == 'because':
      reason = 'because' + reason[len('Because'):]

    return action + ' ' + reason

  def _revise_action_to_question(self, action):
    """Revises action to question.

    Args:
      action: a string similar to "I should buy a car".

    Returns:
      question: a string similar to "Why should you buy a car".
    """
    action = action.strip().strip('.')
    if not action:
      return None

    if action[:len('I should')] == 'I should':
      action = action[len('I should'):]
    return "Why should you" + action

  def _revise_reason_to_answer(self, reason):
    """Revises reason to answer.

    Args:
      reason: a string similar to "Because it is cheap."

    Returns:
      answer: a string similar to "Because it is cheap."
    """
    reason = reason.strip().strip('.')
    if not reason:
      return None
    return reason

  def _revise_statement_to_question_answer(self, statement):
    """Revises statement to question and answer.

    Args:
      statement: a string similar to "I should buy a car because it is stable
      and strong".

    Returns:
      question: a string similar to "Why should you buy a car".
      answer: a string similar to "Because it is cheap."
    """
    if 'because' not in statement:
      return None, None

    pos = statement.find('because')
    action, reason = statement[:pos], statement[pos:]

    question = self._revise_action_to_question(action)
    answer = self._revise_reason_to_answer('Because' + reason[len('because'):])
    return question, answer

  def _process_slogan(self, slogan_annotations, num_positive_slogans):
    """Process slogan annotations.

    Modifies 'slogans' data field in the meta data.

    Args:
      slogan_annotations: a file containing slogan annotations.
      num_positive_slogans: number of positive slogans needed.
    """
    with open(slogan_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    for image_id, slogans in annots.iteritems():
      meta = self._meta[image_id]

      # Process slogans.
      slogans = [convert_to_printable(slogan) for slogan in slogans]

      if len(slogans) > 0:
        if num_positive_slogans is None:
          meta['slogans'] = slogans
        elif len(slogans) >= num_positive_slogans:
          meta['slogans'] = slogans[:num_positive_slogans]

  def _process_combined_action_reason(self,
      qa_action_reason_annotations, num_positive_statements):
    """Processes action reason annotations.

    Modifies 'statements' data field in the meta data.

    Args:
      qa_action_reason_annotations: a file containing qa action-reason
      annotations.
      num_positive_statements: number of positive statements needed.

    Raises:
      ValueError: if annotations file is invalid.
    """
    # Read file.
    with open(qa_action_reason_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    # Parse annotations.
    for image_id, statements in annots.iteritems():
      meta = self._meta[image_id]

      # Process statements.
      statements = [convert_to_printable(statement) for statement in statements]

      if len(statements) > 0:
        if num_positive_statements is None:
          meta['statements'] = statements
        elif len(statements) >= num_positive_statements:
          meta['statements'] = statements[:num_positive_statements]

      # Process questions and answers.
      questions, answers = [], []
      for statement in statements:
        question, answer = self._revise_statement_to_question_answer(statement)
        if question and answer:
          questions.append(question)
          answers.append(answer)

      assert len(questions) == len(answers)
      if len(questions) > 0:
        meta['questions'] = questions
        meta['answers'] = answers

  def _process_seperate_action_reason(self, 
      qa_action_annotations, qa_reason_annotations,
      num_positive_statements):
    """Process action reason annotations.

    Modifies 'questions' and 'answers' data field in the meta data.

    Args:
      qa_action_annotations: a file containing qa action annotations.
      qa_reason_annotations: a file containing qa reason annotations.
      num_positive_statements: number of positive statements needed.
    """
    # Read files.
    with open(qa_action_annotations, 'r') as fp:
      actions = sorted(json.loads(fp.read()).iteritems(), 
          lambda x, y: cmp(x[0], y[0]))
    with open(qa_reason_annotations, 'r') as fp:
      reasons = sorted(json.loads(fp.read()).iteritems(),
          lambda x, y: cmp(x[0], y[0]))

    if len(actions) != len(reasons):
      raise ValueError('Annotation do not match.')

    # Process qa annotations.
    for (action_id, action_annots), (reason_id, reason_annots) in zip(actions, reasons):
      if action_id != reason_id:
        raise ValueError('Annotation do not match.')
      if len(action_annots) != len(reason_annots):
        raise ValueError('Annotation do not match.')

      meta = self._meta[action_id]

      statements = []
      questions, answers = [], []

      for action, reason in zip(action_annots, reason_annots):
        action = convert_to_printable(action)
        reason = convert_to_printable(reason)

        # Process statements.
        statement = self._revise_action_reason_to_statement(action, reason)
        if statement: statements.append(statement)

        # Process question answers.
        question = self._revise_action_to_question(action)
        answer = self._revise_reason_to_answer(reason)
        if question and answer:
          questions.append(question)
          answers.append(answer)

      if len(statements) > 0:
        if num_positive_statements is None:
          meta['statements'] = statements
        elif len(statements) >= num_positive_statements:
          meta['statements'] = statements[:num_positive_statements]

      assert len(questions) == len(answers)
      if len(questions) > 0:
        meta['questions'] = questions
        meta['answers'] = answers

  def _sample_negative_statements(self, negative_examples_per_image):
    """Randomly sample negative statements, only for evaluation purpose.

    Modify 'negative_statements' data field.

    Args:
      negative_examples_per_image: number of negative examples of each image.
    """
    random.seed(286)
    #meta_list = [meta for meta in self.get_meta_list() \
    #            if 'statements' in meta and meta['split'] != 'train']
    meta_list = [meta for meta in self.get_meta_list() \
                if 'statements' in meta]

    for i, meta in enumerate(meta_list):
      neg_stmts = []
      for _ in xrange(negative_examples_per_image):
        index = random.randint(1, len(meta_list) - 1)
        index = (i + index) % len(meta_list)
        assert index != i
        stmts = meta_list[index]['statements']
        index = random.randint(0, len(stmts) - 1)
        neg_stmts.append(stmts[index])
      meta['negative_statements'] = neg_stmts

  def _sample_hard_negative_statements(self, negative_examples_per_image):
    """Randomly sample negative statements, only for evaluation purpose.

    Modify 'negative_statements' data field.

    Args:
      negative_examples_per_image: number of negative examples of each image.
    """
    random.seed(286)
    meta_list = [meta for meta in self.get_meta_list() \
                if 'statements' in meta and meta['split'] != 'train']

    # Build a reverse index: topic_id -> meta list.
    rindex = {}
    for meta in meta_list:
      rindex.setdefault(meta.get('topic_id', 0), []).append(meta)

    for topic_id, meta_list in rindex.iteritems():
      for i, meta in enumerate(meta_list):
        neg_stmts = []
        for _ in xrange(negative_examples_per_image):
          index = random.randint(1, len(meta_list) - 1)
          index = (i + index) % len(meta_list)
          assert index != i
          stmts = meta_list[index]['statements']
          index = random.randint(0, len(stmts) - 1)
          neg_stmts.append(stmts[index])
        meta['hard_negative_statements'] = neg_stmts

  def _sample_negative_slogans(self, negative_examples_per_image):
    """Randomly sample negative slogans, only for evaluation purpose.

    Modify 'negative_slogans' data field.

    Args:
      negative_examples_per_image: number of negative examples of each image.
    """
    random.seed(286)
    meta_list = [meta for meta in self.get_meta_list() \
                if 'slogans' in meta and meta['split'] != 'train']

    for i, meta in enumerate(meta_list):
      neg_slogans = []
      for _ in xrange(negative_examples_per_image):
        index = random.randint(1, len(meta_list) - 1)
        index = (i + index) % len(meta_list)
        assert index != i
        slogans = meta_list[index]['slogans']
        index = random.randint(0, len(slogans) - 1)
        neg_slogans.append(slogans[index])
      meta['negative_slogans'] = neg_slogans

  def _process_densecap(self, densecap_annotations):
    """Process densecap annotations.

    Modifies 'densecap_objects' data field in the meta data.

    Args:
      densecap_annotations: a file containing densecap annotations.
    """
    with open(densecap_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    for i, (image_id, annot) in enumerate(annots.iteritems()):
      #if i % 1000 == 0:
      #  print >> sys.stderr, 'Densecap: %d/%d' % (i, len(annots))
      meta = self._meta[image_id]
      meta['densecap_objects'] = annot

  def _process_attributes(self):
    """Transform statement annotations into attributes.

    Modifies 'attributes' data field in the meta data.

    Raises:
      ValueError: if annotations file is invalid.
    """
    for i, (image_id, meta) in enumerate(self._meta.iteritems()):
      if i % 100 == 0:
        print >> sys.stderr, 'On image %s/%s' % (i, len(self._meta))
      if 'statements' in meta:
        for statement in meta['statements']:
          action, topic, reason = [], [], []
          flag = False
          for (word, postag) in pos_tag(word_tokenize(statement.lower())):
            if 'because' == word:
              flag = True
            elif not flag:
              if 'V' in postag: 
                action.append(word)
              elif 'NN' in postag and word != 'i': 
                topic.append(word)
            else:
              if 'JJ' in postag or 'NN' in postag and word != 'i': 
                reason.append(word)
          if action and topic and reason:
            meta.setdefault('attributes', []).append({
                'action': action,
                'topic': topic,
                'reason': reason
                })

if __name__ == '__main__':
  api = AdsApi('configs/ads_api.config.0', invalid_items=[])
