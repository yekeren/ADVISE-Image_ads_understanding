
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

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
  x_norm = tf.tile(tf.norm(x, axis=1, keep_dims=True), [1, embedding_size])
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
  cosine_distance_fn = lambda x, y: 1 - tf.reduce_sum(tf.multiply(x, y), axis=1)

  dist1 = cosine_distance_fn(anchors, positives)
  dist2 = cosine_distance_fn(anchors, negatives)

  losses = tf.maximum(dist1 - dist2 + alpha, 0)
  losses = tf.boolean_mask(losses, losses > 0)

  loss = tf.cond(
      tf.shape(losses)[0] > 0,
      lambda: tf.reduce_mean(losses),
      lambda: 0.0)

  # Gather statistics.
  loss_ratio = tf.count_nonzero(dist1 + alpha >= dist2, dtype=tf.float32) / batch_size
  good_ratio = tf.count_nonzero(dist1 < dist2, dtype=tf.float32) / batch_size
  bad_ratio = 1 - good_ratio

  return {
    'losses/triplet_loss': loss,
    'triplet/good_ratio': good_ratio,
    'triplet/bad_ratio': bad_ratio,
    'triplet/loss_ratio': loss_ratio,
  }


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

    # Text embedder.
    self._text_embedder = text_embedder_builder.build(
        model_proto.text_embedder)


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
  def text_embedder(self):
    """Returns text_embedder.

    Returns:
      text_embedder: an instance of TextEmbedder.
    """
    return self._text_embedder

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

  def build_text_model(self, text_lengths, text_strings, is_training):
    """Get image embedding vectors.

    Args:
      text_lengths: a [batch] tensor indicating lenghts of each text.
      text_strings: a [batch, max_text_len] tensor indicating multiple texts.
      is_training: if True, update batch norm parameters.

    Returns:
      text_embs: a [batch, embedding_size] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    text_embs = self.text_embedder.embed(
        text_lengths, text_strings, is_training)

    def _assign_fn(sess):
      tf.logging.info('Empty text assign_fn is called.')

    return text_embs, _assign_fn

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

  def _mine_negatives(self, positives):
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

  def build(self, images, num_captions, 
      caption_lengths, caption_strings, 
      num_detections, proposed_features, is_training=True):
    """Builds ads embedding model.

    Args:
      images: a [batch, height, width, 3] uint8 tensor.
      num_captions: a [batch] int64 tensor.
      caption_lengths: a [batch, max_num_captions] int64 tensor.
      caption_strings: a [batch, max_num_captions, max_caption_len] int64 tensor.
      num_detections: a [batch] int64 tensor.
      proposed_features: a [batch, max_detections, feature_size] tensor.
      is_training: if True, build a model for training.

    Returns:
      loss_summaries: a dictionary mapping loss name to loss tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    assign_fn_img = None
    if self.model_proto.from_feature:
      image_embs = self.build_image_model_from_feature(
          num_detections, proposed_features, is_training=is_training)
    else:
      image_embs, assign_fn_img = self.build_image_model(
          images, is_training=is_training) 

    caption_indices = self._mine_related_captions(num_captions)
    caption_lengths = tf.gather_nd(caption_lengths, caption_indices)
    caption_strings = tf.gather_nd(caption_strings, caption_indices)

    caption_embs, assign_fn_txt = self.build_text_model(
        caption_lengths, caption_strings, is_training=is_training)

    # Negative mining.
    image_embs = unit_norm(image_embs)
    caption_embs = unit_norm(caption_embs) 

    stacked_image_embs = []
    stacked_image_embs_unrelated = []
    stacked_caption_embs = []
    stacked_caption_embs_unrelated = []

    for _ in xrange(self.model_proto.triplet_negatives_multiplier):
      stacked_image_embs.append(image_embs)
      stacked_caption_embs.append(caption_embs)

      stacked_image_embs_unrelated.append(self._mine_negatives(image_embs))
      stacked_caption_embs_unrelated.append(self._mine_negatives(caption_embs))

    stacked_image_embs = tf.concat(
        stacked_image_embs, 0)
    stacked_image_embs_unrelated = tf.concat(
        stacked_image_embs_unrelated, 0)
    stacked_caption_embs = tf.concat(
        stacked_caption_embs, 0)
    stacked_caption_embs_unrelated = tf.concat(
        stacked_caption_embs_unrelated, 0)

    # Triplet loss.
    triplet_loss_summaries = {}
    loss_summaries = compute_triplet_loss(
        anchors=stacked_image_embs, 
        positives=stacked_caption_embs, 
        negatives=stacked_caption_embs_unrelated,
        alpha=self.model_proto.triplet_alpha)
    for k, v in loss_summaries.iteritems():
      triplet_loss_summaries[k + '_img'] = v

    loss_summaries = compute_triplet_loss(
        anchors=stacked_caption_embs, 
        positives=stacked_image_embs,
        negatives=stacked_image_embs_unrelated,
        alpha=self.model_proto.triplet_alpha)
    for k, v in loss_summaries.iteritems():
      triplet_loss_summaries[k + '_cap'] = v

    loss = triplet_loss_summaries['losses/triplet_loss_img'] + triplet_loss_summaries['losses/triplet_loss_cap']
    tf.losses.add_loss(loss * 0.5)

    def _assign_fn(sess):
      if assign_fn_img is not None:
        assign_fn_img(sess)
      assign_fn_txt(sess)

    return triplet_loss_summaries, _assign_fn
