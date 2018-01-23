
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.protos import hyperparams_pb2

from utils import triplet_loss
from protos import ads_emb_model_pb2
from object_detection.builders import model_builder
from region_proposal_networks import builder as rpn_builder
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


def _distance_fn(x, y):
  return 1 - tf.reduce_sum(tf.multiply(x, y), axis=1)

distance_fn = _distance_fn

def _triplet_loss_wrapper(anchors, positives, mine_fn, refine_fn,
    distance_fn, margin, anchor_name, positive_name):
  """Wrapper function to process triplet loss.

  Args:
    anchors: [batch, embedding_size] tensor.
    positives: [batch, embedding_size] tensor.
    mine_fn: method to mine <anchor, positive, negative> tuples.
    refine_fn: method to filter <anchor, positive, negative> tuples.
    anchor_name: name of the anchor.
    positive_name: name of the positive.

  Returns:
    triplet_loss_summaries: a dict containing basic triplets info.
  """
  distances = tf.reduce_sum(tf.squared_difference( 
        tf.expand_dims(anchors, 1), tf.expand_dims(positives, 0)), 2)

  pos_indices, neg_indices = mine_fn(distances)
  if refine_fn is not None:
    pos_indices, neg_indices = refine_fn(pos_indices, neg_indices)

  loss_summaries = triplet_loss.compute_triplet_loss(
      anchors=tf.gather(anchors, pos_indices), 
      positives=tf.gather(positives, pos_indices), 
      negatives=tf.gather(positives, neg_indices),
      distance_fn=distance_fn,
      alpha=margin)

  for k, v in loss_summaries.iteritems():
    key = k + '_%s_%s' % (anchor_name, positive_name)
    tf.summary.scalar(key, v)
  return loss_summaries['losses/triplet_loss']


class AdsEmbModel(object):
  """Ads embedding model."""

  def __init__(self, model_proto):
    """Initializes AdsEmbModel.

    Args:
      model_proto: an instance of AdsEmbModel proto.
    """
    self._model_proto = model_proto
    self._tensors = {}

    def _dn(x, y):
      return tf.reduce_sum(tf.square(x - y), axis=1)

    if model_proto.use_two_way_nets_loss:
      global distance_fn
      distance_fn = _dn

    # Region proposal network.
    self.detection_model = None
    if model_proto.HasField('region_proposal_network'):
      self.detection_model = rpn_builder.build(
          model_proto.region_proposal_network)

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
    self.caption_projector = None
    if model_proto.HasField('caption_projector'):
      self.caption_projector = feature_extractor_builder.build(
          model_proto.caption_projector)

    # Topic encoder.
    self.topic_encoder = None
    if model_proto.HasField('topic_encoder'):
      self.topic_encoder = text_encoder_builder.build(
          model_proto.topic_encoder)

    # Symbol encoder.
    self.symbol_encoder = None
    if model_proto.HasField('symbol_encoder'):
      self.symbol_encoder = text_encoder_builder.build(
          model_proto.symbol_encoder)

    # Densecap encoder.
    self.densecap_encoder = None
    if model_proto.HasField('densecap_encoder'):
      self.densecap_encoder = text_encoder_builder.build(
          model_proto.densecap_encoder)
    self.densecap_projector = None
    if model_proto.HasField('densecap_projector'):
      self.densecap_projector = text_encoder_builder.build(
          model_proto.densecap_projector)

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

    self.add_tensor('num_detections', num_detections)

    # Get features from region proposals.
    batch_size, max_detections, _ = proposed_features.get_shape().as_list()
    if max_detections is None:
      max_detections = tf.cast(tf.shape(proposed_features)[1], tf.int64)

    boolean_masks = tf.less(
      tf.range(max_detections, dtype=tf.int64),
      tf.expand_dims(tf.cast(num_detections, dtype=tf.int64), 1))

    self.add_tensor('proposed_features', proposed_features)
    proposed_features = tf.boolean_mask(proposed_features, boolean_masks)

    # Extract image embedding vectors using FC layers.
    raw_embs = tf.identity(proposed_features, 'image_raw_embs')

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
    raw_embs = tf.identity(caption_embs, 'caption_raw_embs')

    self.add_tensor('caption_lengths', caption_lengths)
    self.add_tensor('caption_strings', caption_strings)
    # self.add_tensor('caption_embs', caption_embs)

    if self.caption_projector is not None:
      caption_embs = self.caption_projector.extract_feature(
          caption_embs, is_training=is_training)

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

    if self.densecap_projector is not None:
      caption_embs = self.densecap_projector.extract_feature(
          caption_embs, is_training=is_training)

    def _assign_fn(sess):
      tf.logging.info('Empty caption assign_fn is called.')

    return caption_embs, _assign_fn

  def build_topic_model(self, topics, is_training):
    """Get topic embedding vectors.

    Args:
      topics: a [batch] int64 tensor indicating topics.
      is_training: if True, update batch norm parameters.

    Returns:
      topic_embs: a [batch, embedding_size] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    topic_lengths = tf.ones(shape=topics.get_shape(), dtype=tf.int64)
    topic_strings = tf.expand_dims(topics, 1)

    topic_embs = self.topic_encoder.encode(
        topic_lengths, topic_strings, is_training=is_training)

    def _assign_fn(sess):
      tf.logging.info('Empty topic assign_fn is called.')

    return topic_embs, _assign_fn

  def build_symbol_model(self, symbols, is_training):
    """Get symbol embedding vectors.

    Args:
      symbols: a [batch] int 64 tensor indicating symbols.
      is_training: if True, update batch norm parameters.

    Returns:
      symbol_embs: a [batch, embedding_size] float32 tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    symbol_lengths = tf.ones(shape=symbols.get_shape(), dtype=tf.int64)
    symbol_strings = tf.expand_dims(symbols, 1)

    symbol_embs = self.symbol_encoder.encode(
        symbol_lengths, symbol_strings, is_training=is_training)

    def _assign_fn(sess):
      tf.logging.info('Empty symbol assign_fn is called.')

    return symbol_embs, _assign_fn

  def _mine_related_examples(self, num_captions):
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
      topics, num_symbols, symbols,
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
      num_symbols: a [batch] int64 tensor.
      symbols: a [batch, max_num_symbols] tensor.
      densecap_num_captions: a [batch] int64 tensor.
      densecap_caption_lengths: a [batch, densecap_max_num_captions] int64 tensor.
      densecap_caption_strings: a [batch, densecap_max_num_captions, densecap_max_caption_len] int64 tensor.
      is_training: if True, build a model for training.

    Returns:
      loss_summaries: a dictionary mapping loss name to loss tensor.
      assign_fn: a function used to initialize weights from checkpoint.
    """
    model_proto = self.model_proto

    # Build image model.
    assign_fn_img = None
    if model_proto.from_feature:
      image_embs = self.build_image_model_from_feature(
          num_detections, proposed_features, is_training=is_training)
    else:
      image_embs, assign_fn_img = self.build_image_model(
          images, is_training=is_training) 
    self.add_tensor('image_embs', image_embs)

    # Build caption model.
    caption_indices = self._mine_related_examples(num_captions)
    caption_lengths = tf.gather_nd(caption_lengths, caption_indices)
    caption_strings = tf.gather_nd(caption_strings, caption_indices)

    caption_embs, assign_fn_txt = self.build_caption_model(
        caption_lengths, caption_strings, is_training=is_training)

    if model_proto.normalize_image_embedding:
      image_embs = unit_norm(image_embs)
    caption_embs = unit_norm(caption_embs)

    tf.summary.histogram('embedding/image', image_embs)
    tf.summary.histogram('embedding/caption', caption_embs)

    if model_proto.use_decov_loss:
      def regularize_decov(cov_x):
        return tf.reduce_sum(tf.square(cov_x)) - tf.reduce_sum(tf.square(tf.diag_part(cov_x)))

      cov_x = tf.matmul(tf.transpose(image_embs), image_embs) / image_embs.get_shape()[0].value
      cov_y = tf.matmul(tf.transpose(caption_embs), caption_embs) / caption_embs.get_shape()[0].value

      ratio = model_proto.loss_weight_decov
      loss_withen_x = ratio * regularize_decov(cov_x)
      loss_withen_y = ratio * regularize_decov(cov_y)

      tf.losses.add_loss(loss_withen_x)
      tf.losses.add_loss(loss_withen_y)
      tf.summary.scalar('losses/loss_withen_x', loss_withen_x)
      tf.summary.scalar('losses/loss_withen_y', loss_withen_y)

    # Two way nets loss.
    if model_proto.use_two_way_nets_loss:

      tf.logging.info('Use two way nets model.')
      # loss_h, i.e., L_h mentioned in two way nets paper.
      loss_h = tf.reduce_sum(tf.square(image_embs - caption_embs), axis=1)
      loss_h = tf.reduce_mean(loss_h)
      tf.losses.add_loss(loss_h)
      tf.summary.scalar('losses/regression_h', loss_h)

      # loss_x
      x = tf.get_default_graph().get_tensor_by_name('image_raw_embs:0')
      y = tf.get_default_graph().get_tensor_by_name('caption_raw_embs:0')
      #x = unit_norm(x)
      #y = unit_norm(y)

      # loss_y
      hyper_str = """
        op: FC
        activation: NONE
        regularizer {
          l2_regularizer {
            weight: 1e-6
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          train: true
          scale: true
          center: true
          decay: 0.999
          epsilon: 0.001
        }
      """
      params = hyperparams_pb2.Hyperparams()
      text_format.Merge(hyper_str, params)

      hyperparams = hyperparams_builder.build(params, is_training=is_training)

      with slim.arg_scope(hyperparams):
        image_to_y = slim.fully_connected(
            image_embs, 
            num_outputs=y.get_shape()[-1].value,
            scope='image_to_y')

      params.activation = hyperparams_pb2.Hyperparams.RELU_LEAKY
      hyperparams = hyperparams_builder.build(params, is_training=is_training)
      with slim.arg_scope(hyperparams):
        caption_to_x = slim.fully_connected(
            caption_embs, 
            num_outputs=x.get_shape()[-1].value,
            scope='caption_to_x')

      if is_training:
        image_to_y = tf.nn.dropout(image_to_y, 0.5)
        caption_to_x = tf.nn.dropout(caption_to_x, 0.5)

      #caption_to_x = unit_norm(caption_to_x)
      #image_to_y = unit_norm(image_to_y)
      r_gamma = []
      gamma_list = [var for var in tf.global_variables() if 'gamma' in var.op.name]
      for gamma in gamma_list:
        r_gamma.append(
            tf.reduce_sum(tf.square((1.0 / tf.maximum(gamma, 1e-12)))))
      r_gamma = tf.add_n(r_gamma) * 1e-6
      tf.losses.add_loss(r_gamma)
      tf.summary.scalar('losses/r_gamma', r_gamma)

      tf.summary.histogram('activation/x', x)
      tf.summary.histogram('activation/y', y)
      tf.summary.histogram('activation/x_h', image_embs)
      tf.summary.histogram('activation/y_h', caption_embs)
      tf.summary.histogram('activation/x_reconstruct', caption_to_x)
      tf.summary.histogram('activation/y_reconstruct', image_to_y)

      loss_x = tf.reduce_mean(tf.square(x - caption_to_x), axis=1)
      loss_x = tf.reduce_mean(loss_x)
      tf.losses.add_loss(loss_x)
      tf.summary.scalar('losses/regression_x', loss_x)

      loss_y = tf.reduce_mean(tf.square(y - image_to_y), axis=1)
      loss_y = tf.reduce_mean(loss_y)
      tf.losses.add_loss(loss_y)
      tf.summary.scalar('losses/regression_y', loss_y)

    else:
      # Function for mining triplets.
      triplet_mining_fn = None
      method = model_proto.triplet_mining_method

      if method == ads_emb_model_pb2.AdsEmbModel.ALL:
        triplet_mining_fn = triplet_loss.mine_all_examples

      elif method == ads_emb_model_pb2.AdsEmbModel.HARD:
        def _mine_hard_examples(distances):
          return triplet_loss.mine_hard_examples(distances,
              model_proto.num_negative_examples)
        triplet_mining_fn = _mine_hard_examples

      elif method == ads_emb_model_pb2.AdsEmbModel.SEMI_HARD:
        triplet_mining_fn = triplet_loss.mine_semi_hard_examples

      elif method == ads_emb_model_pb2.AdsEmbModel.HARD_TOPIC:
        triplet_mining_fn = triplet_loss.mine_all_examples

      elif method == ads_emb_model_pb2.AdsEmbModel.HARD_TOPIC2:
        def _mine_hard_examples(distances):
          return triplet_loss.mine_hard_examples(distances,
              model_proto.num_negative_examples)
        triplet_mining_fn = _mine_hard_examples

      assert triplet_mining_fn is not None

      # Get the distance measuring function.
      def _distance_fn_with_dropout(x, y):
        joint_emb = tf.multiply(x, y)
        if is_training:
          joint_emb = tf.nn.dropout(
              joint_emb, model_proto.joint_emb_dropout_keep_prob)
        return 1 - tf.reduce_sum(joint_emb, axis=1)

      def _refine_hard_triplets(pos_indices, neg_indices):
        """Refine topic triplets.

        1. Positive and negative example should be from different topics.
        2. Topic of positive example could not be 'unclear' (0).

        Args:
          pos_indices: a [triplet_batch] int32 tensor denoting index.
          neg_indices: a [triplet_batch] int32 tensor denoting index.
        """
        pos_topics = tf.gather(topics, pos_indices)
        neg_topics = tf.gather(topics, neg_indices)
        masks = tf.logical_and(
            tf.equal(pos_topics, neg_topics),
            tf.not_equal(pos_topics, 0))
        pos_indices = tf.boolean_mask(pos_indices, masks)
        neg_indices = tf.boolean_mask(neg_indices, masks)
        return pos_indices, neg_indices

      refine_fn = None
      if method == ads_emb_model_pb2.AdsEmbModel.HARD_TOPIC:
        refine_fn = _refine_hard_triplets
      if method == ads_emb_model_pb2.AdsEmbModel.HARD_TOPIC2:
        refine_fn = _refine_hard_triplets

      # Use image as anchor: related statements should be more similar.
      loss = _triplet_loss_wrapper(
          image_embs, caption_embs, 
          mine_fn=triplet_mining_fn, 
          refine_fn=refine_fn, 
          distance_fn=_distance_fn_with_dropout,
          margin=model_proto.triplet_alpha, 
          anchor_name='img', 
          positive_name='cap')
      tf.losses.add_loss(loss)

      # Use caption as anchor: related images should be more similar.
      loss = _triplet_loss_wrapper(
          caption_embs, image_embs, 
          mine_fn=triplet_mining_fn, 
          refine_fn=refine_fn, 
          distance_fn=_distance_fn_with_dropout,
          margin=model_proto.triplet_alpha, 
          anchor_name='cap', 
          positive_name='img')
      tf.losses.add_loss(loss)

      # Topic model.
      if method == ads_emb_model_pb2.AdsEmbModel.HARD_TOPIC:
        assert self.topic_encoder is not None
      if method == ads_emb_model_pb2.AdsEmbModel.HARD_TOPIC2:
        assert self.topic_encoder is not None

      if self.topic_encoder is not None:

        topic_embs, _ = self.build_topic_model(topics, is_training=is_training)
        topic_embs = unit_norm(topic_embs)

        def _refine_topic_triplets(pos_indices, neg_indices):
          """Refine topic triplets.

          1. Positive and negative example should be from different topics.
          2. Topic of positive example could not be 'unclear' (0).

          Args:
            pos_indices: a [triplet_batch] int32 tensor denoting index.
            neg_indices: a [triplet_batch] int32 tensor denoting index.
          """
          pos_topics = tf.gather(topics, pos_indices)
          neg_topics = tf.gather(topics, neg_indices)
          if method == ads_emb_model_pb2.AdsEmbModel.HARD_TOPIC:
            masks = tf.logical_and(
                tf.not_equal(pos_topics, neg_topics),
                tf.not_equal(pos_topics, 0))
          else:
            masks = tf.not_equal(pos_topics, 0)
          pos_indices = tf.boolean_mask(pos_indices, masks)
          neg_indices = tf.boolean_mask(neg_indices, masks)
          return pos_indices, neg_indices

        # Use topic as anchor: images from the same topic should be more similar.
        loss = _triplet_loss_wrapper(
            topic_embs, image_embs, 
            mine_fn=triplet_mining_fn, 
            refine_fn=_refine_topic_triplets, 
            distance_fn=_distance_fn_with_dropout,
            margin=model_proto.triplet_alpha, 
            anchor_name='topic', 
            positive_name='img')
        tf.losses.add_loss(loss * model_proto.loss_weight_topic)

        # Use topic as anchor: captions from the same topic should be more similar.
        loss = _triplet_loss_wrapper(
            topic_embs, caption_embs, 
            mine_fn=triplet_mining_fn, 
            refine_fn=_refine_topic_triplets, 
            distance_fn=_distance_fn_with_dropout,
            margin=model_proto.triplet_alpha, 
            anchor_name='topic', 
            positive_name='cap')
        tf.losses.add_loss(loss * model_proto.loss_weight_topic)

      # Symbol model.
      if self.symbol_encoder is not None:
        num_symbols = tf.maximum(num_symbols, 1)
        #self.add_tensor('num_symbols', num_symbols)

        symbol_indices = self._mine_related_examples(num_symbols)
        symbols = tf.gather_nd(symbols, symbol_indices)

        symbol_embs, _ = self.build_symbol_model(symbols, is_training=is_training)
        symbol_embs = unit_norm(symbol_embs)

        def _refine_symbol_triplets(pos_indices, neg_indices):
          """Refine symbol triplets.

          1. Positive and negative example should be from different symbols.
          2. Topic of positive example could not be 'unclear' (0).

          Args:
            pos_indices: a [triplet_batch] int32 tensor denoting index.
            neg_indices: a [triplet_batch] int32 tensor denoting index.
          """
          pos_symbols = tf.gather(symbols, pos_indices)
          neg_symbols = tf.gather(symbols, neg_indices)
          #masks = tf.logical_and(
          #    tf.not_equal(pos_symbols, neg_symbols),
          #    tf.not_equal(pos_symbols, 0))
          masks = tf.not_equal(pos_symbols, 0)
          pos_indices = tf.boolean_mask(pos_indices, masks)
          neg_indices = tf.boolean_mask(neg_indices, masks)
          return pos_indices, neg_indices

        # Use symbol as anchor: images from the same symbol should be more similar.
        loss = _triplet_loss_wrapper(
            symbol_embs, image_embs, 
            mine_fn=triplet_mining_fn, 
            refine_fn=_refine_symbol_triplets, 
            distance_fn=_distance_fn_with_dropout,
            margin=model_proto.triplet_alpha, 
            anchor_name='sym', 
            positive_name='img')
        tf.losses.add_loss(loss * model_proto.loss_weight_symbol)

        # Use symbol as anchor: captions from the same symbol should be more similar.
        loss = _triplet_loss_wrapper(
            symbol_embs, caption_embs, 
            mine_fn=triplet_mining_fn, 
            refine_fn=_refine_symbol_triplets, 
            distance_fn=_distance_fn_with_dropout,
            margin=model_proto.triplet_alpha, 
            anchor_name='sym', 
            positive_name='cap')
        tf.losses.add_loss(loss * model_proto.loss_weight_symbol)

      # Densecap model.
      if self.densecap_encoder is not None:
        densecap_caption_indices = self._mine_related_examples(densecap_num_captions)
        densecap_caption_lengths = tf.gather_nd(
            densecap_caption_lengths, densecap_caption_indices)
        densecap_caption_strings = tf.gather_nd(
            densecap_caption_strings, densecap_caption_indices)

        densecap_caption_embs, _ = self.build_densecap_caption_model(
            densecap_caption_lengths, densecap_caption_strings, is_training=is_training)
        densecap_caption_embs = unit_norm(densecap_caption_embs)

        # Use densecap as anchor: related images should be more similar.
        loss = _triplet_loss_wrapper(
            densecap_caption_embs, image_embs,
            mine_fn=triplet_mining_fn, 
            refine_fn=None, 
            distance_fn=_distance_fn_with_dropout,
            margin=model_proto.triplet_alpha, 
            anchor_name='densecap', 
            positive_name='img')
        tf.losses.add_loss(loss * model_proto.loss_weight_densecap)

        # Use densecap as anchor: related captions should be more similar.
        loss = _triplet_loss_wrapper(
            densecap_caption_embs, caption_embs,
            mine_fn=triplet_mining_fn, 
            refine_fn=None, 
            distance_fn=_distance_fn_with_dropout,
            margin=model_proto.triplet_alpha, 
            anchor_name='densecap', 
            positive_name='cap')
        tf.losses.add_loss(loss * model_proto.loss_weight_densecap)

    def _assign_fn(sess):
      if assign_fn_img is not None:
        assign_fn_img(sess)
      assign_fn_txt(sess)

    return _assign_fn
