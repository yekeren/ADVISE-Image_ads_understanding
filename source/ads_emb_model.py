
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import ads_emb_model_pb2
from object_detection.builders import model_builder
from feature_extractors import builder as feature_extractor_builder
from text_embedders import builder as text_embedder_builder

slim = tf.contrib.slim


def preprocess(image):
  assert image.dtype == tf.uint8
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def unit_norm(x):
  """Compute unit norm for tensor x.

  Args:
    x: a [batch, embedding_size] tensor.

  Returns:
    x_unit: a [batch, embedding_size] tensor that is normalized.
  """
  embedding_size = x.get_shape()[1].value
  x_norm = tf.tile(
      #tf.norm(x, axis=1, keep_dims=True), 
      tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keep_dims=True) + 1e-12),
      [1, embedding_size])
  return x / (x_norm + 1e-12)


def compute_triplet_loss(anchors, positives, negatives, alpha=0.3):
  """Compute triplet loss.

  Args:
    anchors: a [batch, embedding_size] tensor.
    positives: a [batch, embedding_size] tensor.
    negatives: a [batch, embedding_size] tensor.

  Returns:
    triplet_loss: a scalar tensor.
  """
  batch_size = anchors.get_shape()[0].value
  if batch_size is None:
    batch_size = tf.shape(anchors)[0]
  batch_size = 1e-12 + tf.cast(batch_size, tf.float32)

  cosine_distance_fn = lambda x, y: 1 - tf.reduce_sum(tf.multiply(x, y), axis=1)

  dist1 = cosine_distance_fn(anchors, positives)
  dist2 = cosine_distance_fn(anchors, negatives)

  losses = tf.maximum(dist1 - dist2 + alpha, 0)
  losses = tf.boolean_mask(losses, losses > 0)

  loss = tf.cond(
      tf.shape(losses)[0] > 0,
      lambda: tf.reduce_mean(losses),
      lambda: 0.0)
  #loss = tf.cond(tf.is_nan(loss),
  #    lambda: 0.0, lambda: loss)

  #losses = tf.concat([losses, tf.constant([1.0], dtype=tf.float32)], 0)
  #loss = tf.reduce_mean(losses)

  # Gather statistics.
  loss_ratio = tf.count_nonzero(dist1 + alpha >= dist2, dtype=tf.float32) / batch_size
  good_ratio = tf.count_nonzero(dist1 < dist2, dtype=tf.float32) / batch_size
  bad_ratio = 1 - good_ratio

  return {
    'losses/triplet_loss': loss,
    'triplet/num_triplets': batch_size,
    'triplet/good_ratio': good_ratio,
    'triplet/bad_ratio': bad_ratio,
    'triplet/loss_ratio': loss_ratio,
  }

def mine_semi_hard_examples(distances):
  """Mine semi hard examples.
    
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      cosine distance between i-th image and j-th caption.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  # pos_distances is the distance between matched image-caption pairs.
  pos_distances = tf.expand_dims(tf.diag_part(distances), 1)
  indices = tf.where(pos_distances < distances)
  return indices[:, 0], indices[:, 1]


def mine_hard_examples(categories):
  """Mine hard examples.

  Args:
    categories: a [batch] int tensor, in which categories[i] denotes the
      category of i-th topic, value 0 indicates 'unclear' category.
  """
  batch_size = categories.get_shape()[0].value
  if batch_size is None:
    batch_size = tf.shape(categories)[0]

  categories = tf.expand_dims(categories, 0)
  boolean_masks = tf.not_equal(categories, tf.transpose(categories))

  unclear_masks = tf.tile(tf.equal(categories, 0), [batch_size, 1])
  unclear_masks = tf.logical_or(unclear_masks, tf.transpose(unclear_masks))
  boolean_masks = tf.logical_and(
      boolean_masks, tf.logical_not(unclear_masks))

  indices = tf.where(boolean_masks)
  return indices[:, 0], indices[:, 1]


def _mine_random_negatives(positives):
  """Mine negative examples from positive examples.

  Args:
    positives: a [batch, embedding_size] tensor indicating positive examples.

  Returns:
    negatives: a [batch, embedding_size] tensor indicating negative examples.
  """
  batch_size = positives.get_shape()[0].value
  indices = tf.add(
      tf.range(batch_size, dtype=tf.int64),
      tf.random_uniform(
        shape=[batch_size], minval=1, maxval=batch_size, dtype=tf.int64))
  indices = tf.mod(indices, batch_size)
  return tf.gather(positives, indices)

 
def _stack_embedding_vectors(embs, num_replicas=1, is_negative=False):
  """Stack embedding vectors.

  Args:
    embs: a [batch, embedding_size] tensor.
    num_replicas: number of replicas in the batch.
    is_negative: is True, stack random negatives instead of original examples.
  """
  stacked_embs = []
  for _ in xrange(num_replicas):
    if not is_negative:
      stacked_embs.append(embs)
    else:
      stacked_embs.append(_mine_random_negatives(embs))
  return tf.concat(stacked_embs, 0)


class AdsEmbModel(object):
  """Ads embedding model."""

  def __init__(self, model_proto):
    """Initializes AdsEmbModel.

    Args:
      model_proto: an instance of AdsEmbModel proto.
    """
    self._model_proto = model_proto
    self._tensors = {}

    # Region proposal network.
    self._detection_model = None
    if model_proto.HasField('detection_model'):
      self._detection_model = model_builder.build(
          model_proto.detection_model, is_training=False)

    # Feature extractor.
    self._feature_extractor = feature_extractor_builder.build(
        model_proto.feature_extractor)

    # Caption embedder.
    self._caption_embedder = text_embedder_builder.build(
        model_proto.caption_embedder)

    # Topic embedder.
    self._topic_embedder = None
    if model_proto.HasField('topic_embedder'):
      self._topic_embedder = text_embedder_builder.build(
          model_proto.topic_embedder)

    # Densecap embedder.
    self._densecap_embedder = None
    if model_proto.HasField('densecap_embedder'):
      self._densecap_embedder = text_embedder_builder.build(
          model_proto.densecap_embedder)

  @property
  def model_proto(self):
    """Returns model_proto.

    Returns:
      model_proto: an instance of AdsEmbModel proto.
    """
    return self._model_proto

  def add_tensor(self, name, tensor):
    """Add a mapping from name to tensor.

    Args:
      name: name of the tensor.
      tensor: tf tensor.
    """
    self._tensors[name] = tensor

  @property
  def tensors(self):
    """Returns tensors.

    Returns:
      tensors: a dictionary mapping from name to tensor.
    """
    return self._tensors

  @property
  def detection_model(self):
    """Returns detection_model.

    Returns:
      detection_model: an instance of DetectionModel, or None.
    """
    return self._detection_model

  @property
  def feature_extractor(self):
    """Returns feature_extractor.

    Returns:
      feature_extractor: an instance of FeatureExtractor.
    """
    return self._feature_extractor

  @property
  def caption_embedder(self):
    """Returns caption_embedder.

    Returns:
      caption_embedder: an instance of TextEmbedder.
    """
    return self._caption_embedder

  @property
  def topic_embedder(self):
    """Returns topic_embedder.

    Returns:
      topic_embedder: an instance of TextEmbedder.
    """
    return self._topic_embedder

  @property
  def densecap_embedder(self):
    """Returns densecap_embedder.

    Returns:
      densecap_embedder: an instance of TextEmbedder.
    """
    return self._densecap_embedder

  def _build_region_proposals(self, images):
    """Builds region proposal network and infer proposals from images.

    See https://github.com/tensorflow/models/blob/master/object_detection/meta_architectures/ssd_meta_arch.py

    Args:
      images: a [batch, height, width, 3] uint8 tensor.

    Returns:
      detections: a dictionary containing the following fields
        num_detections: a [batch] float32 tensor.
        detection_scores: a [batch, max_detections] float32 tensor.
        detection_boxes: a [batch, max_detections, 4] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    # Generate region proposals using object_detection model.
    if self.detection_model is not None:
      model = self.detection_model
      prediction_dict = model.predict(
          model.preprocess(tf.cast(images, tf.float32)))
      detections = model.postprocess(prediction_dict)

      # Get variables of the detection model.
      if not self.model_proto.detection_checkpoint:
        raise ValueError('Detection checkpoint is invalid.')

      variables_to_restore = filter(
          lambda x: 'FeatureExtractor' in x.op.name or 'BoxPredictor' in x.op.name,
          tf.global_variables())
      assign_fn = slim.assign_from_checkpoint_fn(
          self.model_proto.detection_checkpoint, 
          variables_to_restore)
      return detections, assign_fn

    # Use whole image as proposal.
    def _assign_fn(sess):
      tf.logging.info('Empty assign_fn is called.')

    batch_size = images.get_shape()[0].value
    detections = {
      'num_detections': tf.fill([batch_size], 1.0),
      'detection_scores': tf.fill([batch_size, 1], 1.0),
      'detection_boxes': tf.tile(
          tf.constant(np.array([[[0.0, 0.0, 1.0, 1.0]]], np.float32)),
          [batch_size, 1, 1]
          )
    }
    return detections, _assign_fn

  def _crop_and_resize_region_proposals(self, 
      images, num_detections, detection_boxes, crop_size):
    """Extract region proposals and resize them.

    Given the bounding box info, extract roi image.

    Args:
      images: a [batch, height, width, 3] uint8 tensor.
      num_detections: a [batch] float32 tensor.
      detection_boxes: a [batch, max_detections, 4] float32 tensor.
      crop_size: [crop_height, crop_width] python list or tuple.

    Returns:
      boolean_masks: a [batch, max_detections] boolean tensor.
      proposed_images: a [proposal_batch, image_size, image_size, 3] float32
        tensor.
    """
    batch_size, max_detections, _ = detection_boxes.get_shape().as_list()
    boolean_masks = tf.less(
      tf.range(max_detections, dtype=tf.int64),
      tf.expand_dims(tf.cast(num_detections, dtype=tf.int64), 1))

    # boxes is a [proposal_batch, 4] float32 tensor.
    boxes = tf.boolean_mask(detection_boxes, boolean_masks)

    # box_ind is a [proposal_batch] int32 tensor, value of box_ind[i] specifies
    # the image that the i-th box refers to.
    box_ind = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
    box_ind = tf.boolean_mask(
        tf.tile(box_ind, [1, max_detections]),
        boolean_masks)

    proposed_images = tf.image.crop_and_resize(
        images, boxes, box_ind, crop_size)
    return boolean_masks, proposed_images

  def _embed_feature(self, feature_vectors, 
      embedding_size, weight_decay, is_training=True):
    """Use fully connections to get embedding vectors.

    Args:
      feature_vectors: a [proposal_batch, feature_dims] float32 tensor.
      embedding_size: dimensions of the embedding vector.
      weight_decay: weight decay for the fully connected layers.
      is_training: if True, use mean and variance within the batch.

    Returns:
      embeddings: a [proposal_batch, embedding_size] float32 tensor.
    """
    normalizer_fn = slim.batch_norm
    normalizer_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'is_training': is_training,
    }
    with slim.arg_scope([slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params):
      embs = slim.fully_connected(feature_vectors, 
          num_outputs=embedding_size,
          activation_fn=tf.nn.relu6,
          scope='image/embedding')
    return embs

  def _average_proposed_embs(self, proposed_embs, boolean_masks):
    """Average proposed embedding vectors to get embedding vector of an image.
  
    Args:
      proposed_embs: a [batch, max_detections, embedding_size] float32 tensor.
      boolean_masks: a [batch, max_detections] boolean tensor.
  
    Returns:
      embeddings_averaged: a [batch, embedding_size] tensor storing averaged patch embeddings for each image.
    """
    max_detections = proposed_embs.get_shape()[1].value

    weights = tf.cast(boolean_masks, tf.float32)
    num_detections = tf.reduce_sum(weights, axis=1)
    weights = tf.expand_dims(tf.div(
        weights, 
        1e-12 + tf.tile(tf.expand_dims(num_detections, 1), [1, max_detections])
        ), 1)
    embeddings_averaged = tf.squeeze(tf.matmul(weights, proposed_embs), [1])
    return embeddings_averaged

  def build_image_model_from_feature(self, 
      num_detections, proposed_features, is_training=True):
    """Get image embedding vectors.

    Args:
      num_detections: a [batch] int64 tensor.
      proposed_features: a [batch, max_detections, feature_size] float32 tensor.

    Returns:
      image_embs: a [batch, embedding_size] tensor.
    """
    model_proto = self.model_proto
    tf.summary.scalar('detection/num_detections', 
        tf.reduce_mean(tf.cast(num_detections, tf.float32)))

    batch_size, max_detections, _ = proposed_features.get_shape().as_list()
    boolean_masks = tf.less(
      tf.range(max_detections, dtype=tf.int64),
      tf.expand_dims(tf.cast(num_detections, dtype=tf.int64), 1))

    self.add_tensor('proposed_features', proposed_features)
    proposed_features = tf.boolean_mask(proposed_features, boolean_masks)

    if is_training:
      proposed_features = tf.nn.dropout(proposed_features,
          model_proto.dropout_keep_prob)

    proposed_embs = self._embed_feature(
        proposed_features, 
        embedding_size=model_proto.embedding_size,
        weight_decay=model_proto.weight_decay_fc,
        is_training=is_training)

    # TODO(yek@): write unittest.
    """
      Reshape proposed_embs, and change it:
        from a sparse [proposal_batch, embedding_size] float32 tensor
        to a dense [batch, max_detections, embedding_size] float32 tensor.
    """
    sparse_indices = tf.where(boolean_masks)
    lookup = tf.sparse_to_dense(sparse_indices, 
        output_shape=[batch_size, max_detections], 
        sparse_values=tf.range(tf.shape(proposed_embs)[0]))

    proposed_embs = tf.nn.embedding_lookup(proposed_embs, lookup)
    image_embs = self._average_proposed_embs(proposed_embs, boolean_masks)

    self.add_tensor('proposed_embs', proposed_embs)

    return image_embs

  def build_image_model(self, images, is_training):
    """Get image embedding vectors.

    Args:
      images: a [batch, height, width, 3] uint8 tensor.
      is_training: if True, build a model for training.

    Returns:
      image_embs: a [batch, embedding_size] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    model_proto = self.model_proto

    # Get region proposals.
    region_proposals, assign_fn_rpn = self._build_region_proposals(images)
    self.add_tensor('num_detections', region_proposals['num_detections'])
    self.add_tensor('detection_scores', region_proposals['detection_scores'])
    self.add_tensor('detection_boxes', region_proposals['detection_boxes'])

    tf.summary.scalar('detection/num_detections', 
        tf.reduce_mean(tf.cast(region_proposals['num_detections'], tf.float32)))

    # Extract features.
    crop_size = self.feature_extractor.default_image_size
    boolean_masks, proposed_images = self._crop_and_resize_region_proposals(
        images,
        num_detections=region_proposals['num_detections'],
        detection_boxes=region_proposals['detection_boxes'],
        crop_size=(crop_size, crop_size))
    proposed_images = tf.cast(proposed_images, tf.uint8)
    self.add_tensor('proposed_images', proposed_images)

    proposed_features = self.feature_extractor.extract_feature(
        preprocess(proposed_images), is_training=False)
    self.add_tensor('proposed_features', proposed_features)

    if not model_proto.feature_extractor_checkpoint:
      raise ValueError('Feature extractor checkpoint is missing.')
    assign_fn_extractor = self.feature_extractor.assign_from_checkpoint_fn(
        model_proto.feature_extractor_checkpoint)

    # Embed features into embedding vectors.
    # proposed_embs is a [proposal_batch, embedding_size] float32 tensor.
    if is_training:
      proposed_features = tf.nn.dropout(proposed_features,
          model_proto.dropout_keep_prob)

    proposed_embs = self._embed_feature(
        proposed_features, 
        embedding_size=model_proto.embedding_size,
        weight_decay=model_proto.weight_decay_fc,
        is_training=is_training)

    # TODO(yek@): write unittest.
    """
      Reshape proposed_embs, and change it:
        from a sparse [proposal_batch, embedding_size] float32 tensor
        to a dense [batch, max_detections, embedding_size] float32 tensor.
    """
    batch_size, max_detections, _ = region_proposals['detection_boxes'].get_shape().as_list()

    sparse_indices = tf.where(boolean_masks)
    lookup = tf.sparse_to_dense(sparse_indices, 
        output_shape=[batch_size, max_detections], 
        sparse_values=tf.range(tf.shape(proposed_embs)[0]))

    proposed_embs = tf.nn.embedding_lookup(proposed_embs, lookup)
    image_embs = self._average_proposed_embs(proposed_embs, boolean_masks)

    self.add_tensor('proposed_embs', proposed_embs)

    def _assign_fn(sess):
      assign_fn_rpn(sess)
      assign_fn_extractor(sess)

    return image_embs, _assign_fn

  def build_caption_model(self, caption_lengths, caption_strings, is_training):
    """Get caption embedding vectors.

    Args:
      caption_lengths: a [batch] tensor indicating lenghts of each caption.
      caption_strings: a [batch, max_caption_len] tensor indicating multiple
        captions.
      is_training: if True, update batch norm parameters.

    Returns:
      caption_embs: a [batch, embedding_size] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    caption_embs = self.caption_embedder.embed(
        caption_lengths, caption_strings, is_training)

    self.add_tensor('caption_lengths', caption_lengths)
    self.add_tensor('caption_strings', caption_strings)
    self.add_tensor('caption_embs', caption_embs)

    def _assign_fn(sess):
      tf.logging.info('Empty caption assign_fn is called.')

    return caption_embs, _assign_fn

  def build_densecap_caption_model(self, caption_lengths, caption_strings, is_training):
    """Get caption embedding vectors.

    Args:
      caption_lengths: a [batch] tensor indicating lenghts of each caption.
      caption_strings: a [batch, max_caption_len] tensor indicating multiple
        captions.
      is_training: if True, update batch norm parameters.

    Returns:
      caption_embs: a [batch, embedding_size] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    caption_embs = self.densecap_embedder.embed(
        caption_lengths, caption_strings, is_training)

    self.add_tensor('densecap_caption_lengths', caption_lengths)
    self.add_tensor('densecap_caption_strings', caption_strings)
    self.add_tensor('densecap_caption_embs', caption_embs)

    def _assign_fn(sess):
      tf.logging.info('Empty caption assign_fn is called.')

    return caption_embs, _assign_fn

  def build_topic_model(self, topics, is_training):
    """Get topic embedding vectors.

    Args:
      topics: a [batch] int64 tensor indicating topics.
      is_training: if True, update batch norm parameters.

    Returns:
      topic_embs: a [a batch, embedding_size] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.

    Raises:
      ValueError: if topic_embedder is disabled.
    """
    if self.topic_embedder is None:
      raise ValueError('topic_embedder is disabled.')

    topic_lengths = tf.ones(shape=topics.get_shape(), dtype=tf.int64)
    topic_strings = tf.expand_dims(topics, 1)

    topic_embs = self.topic_embedder.embed(
        topic_lengths, topic_strings, is_training=is_training)

    def _assign_fn(sess):
      tf.logging.info('Empty topic assign_fn is called.')

    return topic_embs, _assign_fn

  def _mine_related_captions(self, num_captions):
    """Returns random selected indices.
  
    Args:
      num_captions: a [batch] tensor indicating number of captions for each image.
  
    Returns:
      caption_indices: a [batch, 2] indicating selected caption.
    """
    batch_size = num_captions.get_shape()[0].value
    main_indices = tf.range(batch_size, dtype=tf.int64)
    caption_indices = tf.mod(
        tf.random_uniform([batch_size], maxval=10000, dtype=tf.int64),
        num_captions)
    caption_indices = tf.stack([main_indices, caption_indices], axis=1)
    return caption_indices

  def build(self, images, 
      num_captions, caption_lengths, caption_strings, 
      num_detections, proposed_features, 
      topics,
      densecap_num_captions, densecap_caption_lengths, densecap_caption_strings,
      is_training=True):
    """Builds ads embedding model.

    Args:
      images: a [batch, height, width, 3] uint8 tensor.
      num_captions: a [batch] int64 tensor.
      caption_lengths: a [batch, max_num_captions] int64 tensor.
      caption_strings: a [batch, max_num_captions, max_caption_len] int64 tensor.
      num_detections: a [batch] int64 tensor.
      proposed_features: a [batch, max_detections, feature_size] float tensor.
      topics: a [batch] int64 tensor.
      densecap_num_captions: a [batch] int64 tensor.
      densecap_caption_lengths: a [batch, densecap_max_num_captions] int64 tensor.
      densecap_caption_strings: a [batch, densecap_max_num_captions, densecap_max_caption_len] int64 tensor.
      is_training: if True, build a model for training.

    Returns:
      loss_summaries: a dictionary mapping loss name to loss tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    model_proto = self.model_proto

    # Image model.
    assign_fn_img = None
    if model_proto.from_feature:
      image_embs = self.build_image_model_from_feature(
          num_detections, proposed_features, is_training=is_training)
    else:
      image_embs, assign_fn_img = self.build_image_model(
          images, is_training=is_training) 
    self.add_tensor('image_embs', image_embs)
    image_embs = unit_norm(image_embs)

    # Caption model.
    caption_indices = self._mine_related_captions(num_captions)
    caption_lengths = tf.gather_nd(caption_lengths, caption_indices)
    caption_strings = tf.gather_nd(caption_strings, caption_indices)

    caption_embs, assign_fn_txt = self.build_caption_model(
        caption_lengths, caption_strings, is_training=is_training)
    caption_embs = unit_norm(caption_embs) 

    losses = []
    triplet_loss_summaries = {}

    # Negative mining: DEFAULT.
    method = model_proto.triplet_mining_method
    assert method == ads_emb_model_pb2.AdsEmbModel.DEFAULT
    if method == ads_emb_model_pb2.AdsEmbModel.DEFAULT:
      stacked_image_embs = _stack_embedding_vectors(
          embs=image_embs,
          num_replicas=model_proto.triplet_negatives_multiplier, 
          is_negative=False)
      stacked_caption_embs = _stack_embedding_vectors(
          embs=caption_embs,
          num_replicas=model_proto.triplet_negatives_multiplier, 
          is_negative=False)
      stacked_image_embs_unrelated = _stack_embedding_vectors(
          embs=image_embs,
          num_replicas=model_proto.triplet_negatives_multiplier, 
          is_negative=True)
      stacked_caption_embs_unrelated = _stack_embedding_vectors(
          embs=caption_embs,
          num_replicas=model_proto.triplet_negatives_multiplier, 
          is_negative=True)

      loss_summaries = compute_triplet_loss(
          anchors=stacked_image_embs, 
          positives=stacked_caption_embs, 
          negatives=stacked_caption_embs_unrelated,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_img_cap'] = v

      loss_summaries = compute_triplet_loss(
          anchors=stacked_caption_embs, 
          positives=stacked_image_embs,
          negatives=stacked_image_embs_unrelated,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_cap_img'] = v

    # Negative mining: SEMI_HARD.
    elif method == ads_emb_model_pb2.AdsEmbModel.SEMI_HARD:
      # distances: a [batch, batch] tensor,
      # distances[i, j] is the cosine distance between i-th image and j-th caption.
      distances = 1 - tf.matmul(image_embs, caption_embs, transpose_b=True)

      # Use image as anchor.
      pos_indices, neg_indices = mine_semi_hard_examples(distances)
      loss_summaries = compute_triplet_loss(
          anchors=tf.gather(image_embs, pos_indices), 
          positives=tf.gather(caption_embs, pos_indices), 
          negatives=tf.gather(caption_embs, neg_indices),
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_img'] = v

      # Use caption as anchor.
      pos_indices, neg_indices = mine_semi_hard_examples(tf.transpose(distances, [1, 0]))
      loss_summaries = compute_triplet_loss(
          anchors=tf.gather(caption_embs, pos_indices), 
          positives=tf.gather(image_embs, pos_indices), 
          negatives=tf.gather(image_embs, neg_indices),
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_cap'] = v

    tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_img_cap'])
    tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_cap_img'])

    # Topic model.
    if self.topic_embedder is not None:
      topic_embs, _ = self.build_topic_model(topics, is_training=is_training)
      topic_embs = unit_norm(topic_embs) 

      stacked_topic_embs = _stack_embedding_vectors(
          embs=topic_embs,
          num_replicas=model_proto.triplet_negatives_multiplier, 
          is_negative=False)
      stacked_topic_embs_unrelated = _stack_embedding_vectors(
          embs=topic_embs,
          num_replicas=model_proto.triplet_negatives_multiplier, 
          is_negative=True)

      loss_summaries = compute_triplet_loss(
          anchors=stacked_image_embs, 
          positives=stacked_topic_embs, 
          negatives=stacked_topic_embs_unrelated,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_img_topic'] = v

      loss_summaries = compute_triplet_loss(
          anchors=stacked_topic_embs, 
          positives=stacked_image_embs,
          negatives=stacked_image_embs_unrelated,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_topic_img'] = v

      loss_summaries = compute_triplet_loss(
          anchors=stacked_caption_embs, 
          positives=stacked_topic_embs, 
          negatives=stacked_topic_embs_unrelated,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_cap_topic'] = v

      loss_summaries = compute_triplet_loss(
          anchors=stacked_topic_embs, 
          positives=stacked_caption_embs,
          negatives=stacked_caption_embs_unrelated,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_topic_cap'] = v

      #pos_indices, neg_indices = mine_hard_examples(topics)

      ## Topic-image pairs.
      #loss_summaries = compute_triplet_loss(
      #    anchors=tf.gather(topic_embs, pos_indices), 
      #    positives=tf.gather(image_embs, pos_indices), 
      #    negatives=tf.gather(image_embs, neg_indices),
      #    alpha=model_proto.triplet_alpha)
      #for k, v in loss_summaries.iteritems():
      #  triplet_loss_summaries[k + '_topic_img'] = v

      ## Topic-caption pairs.
      #loss_summaries = compute_triplet_loss(
      #    anchors=tf.gather(topic_embs, pos_indices), 
      #    positives=tf.gather(caption_embs, pos_indices), 
      #    negatives=tf.gather(caption_embs, neg_indices),
      #    alpha=model_proto.triplet_alpha)
      #for k, v in loss_summaries.iteritems():
      #  triplet_loss_summaries[k + '_topic_cap'] = v

      tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_topic_img'])
      tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_topic_cap'])
      tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_img_topic'])
      tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_cap_topic'])

    # Densecap model.
    if self.densecap_embedder is not None:
      densecap_caption_indices = self._mine_related_captions(densecap_num_captions)
      densecap_caption_lengths = tf.gather_nd(
          densecap_caption_lengths, densecap_caption_indices)
      densecap_caption_strings = tf.gather_nd(
          densecap_caption_strings, densecap_caption_indices)

      densecap_caption_embs, _ = self.build_densecap_caption_model(
          densecap_caption_lengths, densecap_caption_strings, is_training=is_training)
      densecap_caption_embs = unit_norm(densecap_caption_embs) 

      method = model_proto.triplet_mining_method
      assert method == ads_emb_model_pb2.AdsEmbModel.DEFAULT

      # Negative mining: DEFAULT.
      if method == ads_emb_model_pb2.AdsEmbModel.DEFAULT:
        stacked_densecap_caption_embs = _stack_embedding_vectors(
            embs=densecap_caption_embs,
            num_replicas=model_proto.triplet_negatives_multiplier, 
            is_negative=False)
        stacked_densecap_caption_embs_unrelated = _stack_embedding_vectors(
            embs=densecap_caption_embs,
            num_replicas=model_proto.triplet_negatives_multiplier, 
            is_negative=True)

        loss_summaries = compute_triplet_loss(
            anchors=stacked_image_embs, 
            positives=stacked_densecap_caption_embs, 
            negatives=stacked_densecap_caption_embs_unrelated,
            alpha=model_proto.triplet_alpha)
        for k, v in loss_summaries.iteritems():
          triplet_loss_summaries[k + '_img_densecap'] = v

        loss_summaries = compute_triplet_loss(
            anchors=stacked_densecap_caption_embs, 
            positives=stacked_image_embs,
            negatives=stacked_image_embs_unrelated,
            alpha=model_proto.triplet_alpha)
        for k, v in loss_summaries.iteritems():
          triplet_loss_summaries[k + '_densecap_img'] = v

        tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_img_densecap'])
        tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_densecap_img'])

      ## Negative mining: SEMI_HARD.
      #elif method == ads_emb_model_pb2.AdsEmbModel.SEMI_HARD:
      ##if True:
      #  # distances: a [batch, batch] tensor,
      #  # distances[i, j] is the cosine distance between i-th image and j-th caption.
      #  distances = 1 - tf.matmul(image_embs, densecap_caption_embs, transpose_b=True)

      #  # Use image as anchor.
      #  pos_indices, neg_indices = mine_semi_hard_examples(distances)
      #  loss_summaries = compute_triplet_loss(
      #      anchors=tf.gather(image_embs, pos_indices), 
      #      positives=tf.gather(densecap_caption_embs, pos_indices), 
      #      negatives=tf.gather(densecap_caption_embs, neg_indices),
      #      alpha=model_proto.triplet_alpha)
      #  for k, v in loss_summaries.iteritems():
      #    triplet_loss_summaries[k + '_img_densecap'] = v

      #  # Use caption as anchor.
      #  pos_indices, neg_indices = mine_semi_hard_examples(tf.transpose(distances, [1, 0]))
      #  loss_summaries = compute_triplet_loss(
      #      anchors=tf.gather(densecap_caption_embs, pos_indices), 
      #      positives=tf.gather(image_embs, pos_indices), 
      #      negatives=tf.gather(image_embs, neg_indices),
      #      alpha=model_proto.triplet_alpha)
      #  for k, v in loss_summaries.iteritems():
      #    triplet_loss_summaries[k + '_densecap_img'] = v

      #  self.add_tensor('densecap_loss_ratio', loss_summaries['triplet/loss_ratio'])

      # losses.append(triplet_loss_summaries['losses/triplet_loss_densecap_img'])
      # losses.append(triplet_loss_summaries['losses/triplet_loss_img_densecap'])

      #tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_densecap_img'])
      #tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_img_densecap'])

      #self.add_tensor('cap_img',
      #    triplet_loss_summaries['losses/triplet_loss_cap_img'])
      #self.add_tensor('img_cap',
      #    triplet_loss_summaries['losses/triplet_loss_img_cap'])
      #self.add_tensor('densecap_img',
      #    triplet_loss_summaries['losses/triplet_loss_densecap_img'])
      #self.add_tensor('img_densecap',
      #    triplet_loss_summaries['losses/triplet_loss_img_densecap'])

    #tf.losses.add_loss(tf.add_n(losses) / (1.0 * len(losses)))

    def _assign_fn(sess):
      if assign_fn_img is not None:
        assign_fn_img(sess)
      assign_fn_txt(sess)

    return triplet_loss_summaries, _assign_fn
