
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder

from utils import triplet_loss
from protos import ads_emb_model_pb2
from object_detection.builders import model_builder
from region_proposal_networks import builder as rpn_builder
from spatial_transformer_networks import builder as spatial_transformer_builder
from feature_extractors import builder as feature_extractor_builder
from text_encoders import builder as text_encoder_builder

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
  return tf.nn.l2_normalize(x, 1)


def distance_fn(x, y):
  return 1 - tf.reduce_sum(tf.multiply(x, y), axis=1)
  #return tf.reduce_sum(tf.square(x - y), axis=1)


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

def mine_all_examples(distances):
  """Mine semi hard examples.
    
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      cosine distance between i-th image and j-th caption.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  # pos_distances is the distance between matched image-caption pairs.
  batch_size = distances.get_shape()[0].value

  indices = tf.where(
      tf.less(
        tf.diag(tf.fill([batch_size], 1)), 
        1))
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
    self.detection_model = None
    if model_proto.HasField('region_proposal_network'):
      self.detection_model = rpn_builder.build(
          model_proto.region_proposal_network)

    # Spatial transformer network.
    self.spatial_transformer = None
    if model_proto.HasField('spatial_transformer'):
      self.spatial_transformer = spatial_transformer_builder.build(
          model_proto.spatial_transformer)

    # Feature extractor.
    self.feature_extractor = feature_extractor_builder.build(
        model_proto.feature_extractor)

    # Image encoder.
    self.image_encoder = feature_extractor_builder.build(
        model_proto.image_encoder)

    # Confidence predictor.
    self.confidence_predictor = None
    if model_proto.HasField('confidence_predictor'):
      self.confidence_predictor = feature_extractor_builder.build(
          model_proto.confidence_predictor)

    # Caption encoder.
    self.caption_encoder = text_encoder_builder.build(
        model_proto.caption_encoder)

    # Topic encoder.
    self.topic_encoder = None
    if model_proto.HasField('topic_encoder'):
      self.topic_encoder = text_encoder_builder.build(
          model_proto.topic_encoder)

    # Densecap encoder.
    self.densecap_encoder = None
    if model_proto.HasField('densecap_encoder'):
      self.densecap_encoder = text_encoder_builder.build(
          model_proto.densecap_encoder)

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
    if self.detection_model is None:
      raise ValueError('detection model cannot be None.')

    model = self.detection_model
    detections = model.predict(images, is_training=False)

    # Get variables of the detection model.
    if not self.model_proto.region_proposal_network_checkpoint:
      tf.logging.warning("Detection checkpoint is invalid !!!")

    assign_fn = model.assign_from_checkpoint_fn(
        self.model_proto.region_proposal_network_checkpoint)
    return detections, assign_fn

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

#  def embed_feature(self, feature_vectors, is_training=True):
#    """Use fully connections to get embedding vectors.
#
#    Args:
#      feature_vectors: a [proposal_batch, feature_dims] float32 tensor.
#      is_training: if True, use mean and variance within the batch.
#
#    Returns:
#      embeddings: a [proposal_batch, embedding_size] float32 tensor.
#    """
#
#    model_proto = self.model_proto
#
#    fc_hyperparams = hyperparams_builder.build(
#        model_proto.fc_hyperparams,
#        is_training=is_training)
#    pred_hyperparams = hyperparams_builder.build(
#        model_proto.pred_hyperparams,
#        is_training=is_training)
#
#    tf.logging.info('*' * 128)
#    with slim.arg_scope(fc_hyperparams):
#      node = feature_vectors
#      for i in xrange(model_proto.fc_hidden_layers):
#        node = slim.fully_connected(node, 
#            num_outputs=model_proto.fc_hidden_units,
#            scope='image_encoder/hidden_%d' % (i))
#        if is_training:
#          node = tf.nn.dropout(node, model_proto.fc_dropout_keep_prob)
#        tf.logging.info('%s: %s', node.op.name, node.get_shape().as_list())
#
#    with slim.arg_scope(pred_hyperparams):
#      node = slim.fully_connected(node, 
#          num_outputs=model_proto.embedding_size,
#          scope='image_encoder/project')
#      if model_proto.pred_activation_fn == ads_emb_model_pb2.AdsEmbModel.SIGMOID:
#        node = tf.sigmoid(node)
#
#      tf.logging.info('%s: %s', node.op.name, node.get_shape().as_list())
#
#    return node

  def _average_proposed_embs(self, proposed_embs, boolean_masks,
      proposed_scores=None):
    """Average proposed embedding vectors to get embedding vector of an image.
  
    Args:
      proposed_embs: a [batch, max_detections, embedding_size] float32 tensor.
      boolean_masks: a [batch, max_detections] boolean tensor.
  
    Returns:
      embeddings_averaged: a [batch, embedding_size] tensor storing averaged patch embeddings for each image.
    """
    # max_detections = proposed_embs.get_shape()[1].value

    # weights = tf.cast(boolean_masks, tf.float32)
    # if proposed_scores is not None:
    #   weights = tf.multiply(weights, proposed_scores)
    # num_detections = tf.reduce_sum(weights, axis=1)
    # weights = tf.expand_dims(tf.div(
    #     weights, 
    #     1e-12 + tf.tile(tf.expand_dims(num_detections, 1), [1, max_detections])
    #     ), 1)
    # embeddings_averaged = tf.squeeze(tf.matmul(weights, proposed_embs), [1])
    # return embeddings_averaged

    # Change by yek@.
    max_detections = proposed_embs.get_shape()[1].value

    weights = tf.cast(boolean_masks, tf.float32)
    if proposed_scores is not None:
      weights = tf.multiply(weights, proposed_scores)
    num_detections = tf.reduce_sum(weights, axis=1)
    if self.model_proto.average_method == ads_emb_model_pb2.AdsEmbModel.AVG:
      weights = tf.div(weights, 
          tf.maximum(1e-12, 
            tf.tile(tf.expand_dims(num_detections, 1), [1, max_detections])))
    weights = tf.expand_dims(weights, 1)
    embeddings_averaged = tf.squeeze(tf.matmul(weights, proposed_embs), [1])
    return embeddings_averaged

  def build_image_model_from_feature(self, 
      num_detections, proposed_features,
      is_training=True):
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

    # Get features from region proposals.
    batch_size, max_detections, _ = proposed_features.get_shape().as_list()
    boolean_masks = tf.less(
      tf.range(max_detections, dtype=tf.int64),
      tf.expand_dims(tf.cast(num_detections, dtype=tf.int64), 1))

    self.add_tensor('proposed_features', proposed_features)
    proposed_features = tf.boolean_mask(proposed_features, boolean_masks)

    # Extract image embedding vectors using FC layers.
    proposed_embs = self.image_encoder.extract_feature(
        proposed_features, is_training=is_training)

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

    # Use confidence scores.
    proposed_scores = None
    if self.confidence_predictor is not None:
      proposed_scores = self.confidence_predictor.extract_feature(
          proposed_features, is_training=is_training)
      proposed_scores = tf.nn.embedding_lookup(proposed_scores, lookup)
      proposed_scores = tf.nn.softmax(tf.squeeze(proposed_scores, [2]))

    image_embs = self._average_proposed_embs(proposed_embs, boolean_masks,
        proposed_scores)

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

    num_detections = region_proposals['num_detections']
    detection_scores = region_proposals['detection_scores']
    detection_boxes = region_proposals['detection_boxes']

    self.add_tensor('num_detections', num_detections)
    self.add_tensor('detection_scores', detection_scores)
    self.add_tensor('detection_boxes', detection_boxes)

    tf.summary.scalar('detection/num_detections', 
        tf.reduce_mean(tf.cast(num_detections, tf.float32)))

    # Spatial transformer networks.
    theta = self.spatial_transformer.predict_transformation(
        feature_map=region_proposals.get('feature_map', None),
        num_detections=num_detections,
        detection_boxes=detection_boxes, 
        is_training=is_training)

    crop_size = self.feature_extractor.default_image_size
    proposed_images = self.spatial_transformer.transform_image(
        tf.cast(images, tf.float32), theta, out_size=(crop_size, crop_size))
    proposed_boxes = self.spatial_transformer.decode_bounding_box(theta)
    self.add_tensor('proposed_boxes', proposed_boxes)

    batch_size, max_detections, _ = detection_boxes.get_shape().as_list()
    boolean_masks = tf.less(
      tf.range(max_detections, dtype=tf.int64),
      tf.expand_dims(tf.cast(num_detections, dtype=tf.int64), 1))

    proposed_images = tf.boolean_mask(proposed_images, boolean_masks)

    # Extract features from proposals.
    proposed_images = tf.cast(proposed_images, tf.uint8)
    self.add_tensor('proposed_images', proposed_images)

    tf.summary.image("images", images)
    tf.summary.image("proposed_images", proposed_images)

    proposed_features = self.feature_extractor.extract_feature(
        preprocess(proposed_images), is_training=False)
    self.add_tensor('proposed_features', proposed_features)

    if not model_proto.feature_extractor_checkpoint:
      raise ValueError('Feature extractor checkpoint is missing.')
    assign_fn_extractor = self.feature_extractor.assign_from_checkpoint_fn(
        model_proto.feature_extractor_checkpoint)

    # Embed features into embedding vectors.
    # proposed_embs is a [proposal_batch, embedding_size] float32 tensor.

    proposed_embs = self.image_encoder.extract_feature(
        proposed_features, is_training=is_training)

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

    # Use confidence scores.
    proposed_scores = None
    if self.confidence_predictor is not None:
      proposed_scores = self.confidence_predictor.extract_feature(
          proposed_features, is_training=is_training)
      proposed_scores = tf.nn.embedding_lookup(proposed_scores, lookup)
      proposed_scores = tf.nn.softmax(tf.squeeze(proposed_scores, [2]))
      self.add_tensor('proposed_scores', proposed_scores)

    image_embs = self._average_proposed_embs(
        proposed_embs, boolean_masks, proposed_scores)

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
    caption_embs = self.caption_encoder.encode(
        caption_lengths, caption_strings, is_training)

    self.add_tensor('caption_lengths', caption_lengths)
    self.add_tensor('caption_strings', caption_strings)
    # self.add_tensor('caption_embs', caption_embs)

    return caption_embs, self.caption_encoder.assign_from_checkpoint_fn(
        self.model_proto.caption_encoder_checkpoint)

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
    caption_embs = self.densecap_encoder.encode(
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
      ValueError: if topic_encoder is disabled.
    """
    if self.topic_encoder is None:
      raise ValueError('topic_encoder is disabled.')

    topic_lengths = tf.ones(shape=topics.get_shape(), dtype=tf.int64)
    topic_strings = tf.expand_dims(topics, 1)

    topic_embs = self.topic_encoder.encode(
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
          num_detections, proposed_features, 
          is_training=is_training)
    else:
      image_embs, assign_fn_img = self.build_image_model(
          images, is_training=is_training) 
    self.add_tensor('image_embs', image_embs)

    # Caption model.
    caption_indices = self._mine_related_captions(num_captions)
    caption_lengths = tf.gather_nd(caption_lengths, caption_indices)
    caption_strings = tf.gather_nd(caption_strings, caption_indices)

    caption_embs, assign_fn_txt = self.build_caption_model(
        caption_lengths, caption_strings, 
        is_training=is_training)

    #if model_proto.extra_relu6_during_training: 
    #  image_embs = tf.nn.relu6(image_embs)

    if model_proto.normalize_image_embedding:
      image_embs = unit_norm(image_embs)
    caption_embs = unit_norm(caption_embs)

    tf.summary.histogram('embedding/image', image_embs)
    tf.summary.histogram('embedding/caption', caption_embs)

    # Process triplets mining.
    triplet_mining_fn = None
    method = model_proto.triplet_mining_method

    if method == ads_emb_model_pb2.AdsEmbModel.ALL:
      triplet_mining_fn = triplet_loss.mine_all_examples

    elif method == ads_emb_model_pb2.AdsEmbModel.HARD:
      def _mine_hard_examples(distances):
        return triplet_loss.mine_hard_examples(distances,
            model_proto.num_negative_examples)
      triplet_mining_fn = _mine_hard_examples

    assert triplet_mining_fn is not None

    distances = tf.reduce_sum(
        tf.squared_difference(
          tf.expand_dims(image_embs, 1), 
          tf.expand_dims(caption_embs, 0), 
          ), 2)

    # Get the distance measuring function.
    def _distance_fn_with_dropout(x, y):
      joint_emb = tf.multiply(x, y)
      if is_training:
        joint_emb = tf.nn.dropout(joint_emb,
            model_proto.joint_emb_dropout_keep_prob)
      return 1 - tf.reduce_sum(joint_emb, axis=1)

    triplet_loss_summaries = {}

    # Use image as anchor.
    pos_indices, neg_indices = triplet_mining_fn(distances)
    loss_summaries = triplet_loss.compute_triplet_loss(
        anchors=tf.gather(image_embs, pos_indices), 
        positives=tf.gather(caption_embs, pos_indices), 
        negatives=tf.gather(caption_embs, neg_indices),
        distance_fn=_distance_fn_with_dropout,
        alpha=model_proto.triplet_alpha)
    for k, v in loss_summaries.iteritems():
      triplet_loss_summaries[k + '_img_cap'] = v

    # Use caption as anchor.
    pos_indices, neg_indices = triplet_mining_fn(
        tf.transpose(distances, [1, 0]))
    loss_summaries = triplet_loss.compute_triplet_loss(
        anchors=tf.gather(caption_embs, pos_indices), 
        positives=tf.gather(image_embs, pos_indices), 
        negatives=tf.gather(image_embs, neg_indices),
        distance_fn=_distance_fn_with_dropout,
        alpha=model_proto.triplet_alpha)
    for k, v in loss_summaries.iteritems():
      triplet_loss_summaries[k + '_cap_img'] = v

    tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_img_cap'])
    tf.losses.add_loss(triplet_loss_summaries['losses/triplet_loss_cap_img'])

    # Densecap model.
    if self.densecap_encoder is not None:
      densecap_caption_indices = self._mine_related_captions(densecap_num_captions)
      densecap_caption_lengths = tf.gather_nd(
          densecap_caption_lengths, densecap_caption_indices)
      densecap_caption_strings = tf.gather_nd(
          densecap_caption_strings, densecap_caption_indices)

      densecap_caption_embs, _ = self.build_densecap_caption_model(
          densecap_caption_lengths, densecap_caption_strings, is_training=is_training)
      densecap_caption_embs = unit_norm(densecap_caption_embs)

      distances = tf.reduce_sum(
          tf.squared_difference(
            tf.expand_dims(image_embs, 1), 
            tf.expand_dims(densecap_caption_embs, 0), 
            ), 2)

      # Use image as anchor.
      pos_indices, neg_indices = triplet_mining_fn(distances)
      loss_summaries = triplet_loss.compute_triplet_loss(
          anchors=tf.gather(image_embs, pos_indices), 
          positives=tf.gather(densecap_caption_embs, pos_indices), 
          negatives=tf.gather(densecap_caption_embs, neg_indices),
          distance_fn=_distance_fn_with_dropout,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_img_densecap'] = v

      # Use caption as anchor.
      pos_indices, neg_indices = triplet_mining_fn(
          tf.transpose(distances, [1, 0]))
      loss_summaries = triplet_loss.compute_triplet_loss(
          anchors=tf.gather(densecap_caption_embs, pos_indices), 
          positives=tf.gather(image_embs, pos_indices), 
          negatives=tf.gather(image_embs, neg_indices),
          distance_fn=_distance_fn_with_dropout,
          alpha=model_proto.triplet_alpha)
      for k, v in loss_summaries.iteritems():
        triplet_loss_summaries[k + '_densecap_img'] = v

      tf.losses.add_loss(model_proto.loss_weight_densecap * triplet_loss_summaries['losses/triplet_loss_img_densecap'])
      tf.losses.add_loss(model_proto.loss_weight_densecap * triplet_loss_summaries['losses/triplet_loss_densecap_img'])

    def _assign_fn(sess):
      if assign_fn_img is not None:
        assign_fn_img(sess)
      assign_fn_txt(sess)

    return triplet_loss_summaries, _assign_fn
