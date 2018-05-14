from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import logging

from models import utils
from models import image_stmt_model
from losses import triplet_loss
from protos import advise_model_pb2
from text_encoders import builder
from utils import mlp
from readers.utils import load_symbol_cluster

from models.image_stmt_model import triplet_loss_wrap_func

slim = tf.contrib.slim


class Model(image_stmt_model.Model):
  """ADVISEModel."""

  def __init__(self, model_proto, is_training=False):
    """Initializes ads model.

    Args:
      model_proto: an instance of advise_model_pb2.ADVISEModel.
      is_training: if True, training graph would be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, advise_model_pb2.ADVISEModel):
      raise ValueError("The model_proto has to be an instance of ADVISEModel.")

    self._stmt_encoder = builder.build(model_proto.stmt_encoder, is_training)
    if model_proto.densecap_loss_weight > 0:
      self._densecap_encoder = builder.build(
          model_proto.densecap_encoder, is_training)

    if model_proto.symbol_loss_weight > 0:
      self._symbol_encoder = builder.build(
          model_proto.symbol_encoder, is_training)

    self._mining_fn = triplet_loss.build_mining_func(model_proto.triplet_mining)

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      variables: a list of model variables.
    """
    model_proto = self._model_proto

    if model_proto.knowledge_training_mode \
      == advise_model_pb2.ADVISEModel.RESIDUAL_TRAINING:
      if model_proto.use_knowledge_branch:
        return tf.trainable_variables('confidence')

    return tf.trainable_variables()

  def encode_image(self, img_features, roi_features):
    """Encodes image into embedding vector.

    Args:
      img_features: a [batch, feature_dimensions] tf.float32 tensor.
      roi_features: a [batch, number_of_regions, feature_dimensions] tf.float32 tensor.

    Raises:
      ValueError: if the pooling method of the model_proto is invalid.

    Returns:
      img_encoded: a [batch, common_dimentions] tf.float32 tensor, l2_normalize
        is NOT applied.
      img_attention: a [batch, number_of_regions] tf.float32 tensor, output of
        the softmax. 
    """
    model_proto = self._model_proto
    is_training = self._is_training

    if model_proto.use_image_as_proposal:
      roi_features = tf.concat(
          [tf.expand_dims(img_features, 1), roi_features], axis=1)
      roi_features = roi_features[:, :-1, :]

    _, num_of_regions, feature_dimensions = roi_features.get_shape().as_list()

    # Encode roi regions.
    roi_features_reshaped = tf.reshape(roi_features, [-1, feature_dimensions])
    roi_encoded_reshaped = utils.encode_feature( 
        roi_features_reshaped, model_proto.image_encoder, is_training)
    roi_encoded = tf.reshape(roi_encoded_reshaped,
        shape=[-1, num_of_regions, model_proto.image_encoder.num_outputs])

    img_attention = None

    # Average pooling.
    if model_proto.pooling_method == advise_model_pb2.ADVISEModel.AVG_POOL:
      img_encoded = tf.reduce_mean(roi_encoded, 1)

    # Attention pooling.
    else:
      assert model_proto.HasField('image_attention_predictor')
      assert model_proto.image_attention_predictor.num_outputs == 1

      if model_proto.pooling_method == advise_model_pb2.ADVISEModel.ATT_POOL:
        attention_inputs = roi_encoded

      elif model_proto.pooling_method == advise_model_pb2.ADVISEModel.ATT_POOL_DS_SUM:
        attention_inputs = tf.subtract(
            roi_encoded,
            tf.tile(
              tf.reduce_sum(roi_encoded, 1, keepdims=True), 
              [1, num_of_regions, 1]))

      elif model_proto.pooling_method == advise_model_pb2.ADVISEModel.ATT_POOL_DS_MAX:
        attention_inputs = tf.subtract(
            roi_encoded,
            tf.tile(
              tf.reduce_max(roi_encoded, 1, keepdims=True), 
              [1, num_of_regions, 1]))

      else:
        raise ValueError('Unknown pooling method %i.' % (model_proto.pooling_method))

      attention_inputs_reshaped = tf.reshape( 
          attention_inputs, [-1, attention_inputs.get_shape()[-1].value])

      # Image attention, [batch, num_of_regions].
      img_attention = tf.reshape(
          utils.encode_feature(
            attention_inputs_reshaped, 
            model_proto.image_attention_predictor, 
            is_training),
          shape=[-1, num_of_regions])
      img_attention = tf.nn.softmax(img_attention)
      img_encoded = tf.squeeze(tf.matmul(
            tf.expand_dims(img_attention, 1), roi_encoded), [1])

    # Autoencoder: reconstruction loss.
    roi_decoded_reshaped = utils.encode_feature( 
        roi_encoded_reshaped, model_proto.image_decoder, is_training)

    if model_proto.autoencoder_loss_weight > 0:
      squared_diff = tf.squared_difference(
          roi_features_reshaped, roi_decoded_reshaped)
      squared_error = tf.reduce_mean(tf.reduce_sum(squared_diff, 1))

      tf.losses.add_loss(model_proto.autoencoder_loss_weight * squared_error)
      tf.summary.scalar('losses/autoencoder', squared_error)

    # Log-prob loss, force the model to focus on limited regions.
    if img_attention is not None and model_proto.log_prob_loss_weight > 0:
      log_prob_losses = tf.log(0.01 + img_attention)
      log_prob_loss = tf.reduce_mean(log_prob_losses)

      tf.summary.scalar('losses/log_prob', log_prob_loss)
      tf.losses.add_loss(model_proto.log_prob_loss_weight * log_prob_loss)

    # For tensorboard.
    if img_attention is not None:
      tf.summary.scalar(
          '{}/softmax_max'.format(model_proto.image_attention_predictor.scope), 
          tf.reduce_mean(tf.reduce_max(img_attention, 1)))
      tf.summary.scalar(
          '{}/softmax_min'.format(model_proto.image_attention_predictor.scope), 
          tf.reduce_mean(tf.reduce_min(img_attention, 1)))

    return img_encoded, img_attention

  def encode_text(self, text_strings, text_lengths, encoder):
    """Encodes text into embedding vector.

    Args:
      text_strings: a [batch, max_text_len] tf.int32 tensor.
      text_lengths: a [batch] tf.int32 tensor.
      encoder: an instance of TextEncoder used to encode text.

    Raises:
      ValueError: if the pooling method of the model_proto is invalid.

    Returns:
      text_encoded: a [batch, common_dimentions] tf.float32 tensor, l2_normalize
        is NOT applied.
      text_attention: a [batch, max_text_len] tf.float32 tensor, output of
        the softmax. 
    """
    (text_encoded, _, _) = encoder.encode(text_strings, text_lengths)
    self._init_fn_list.append(encoder.get_init_fn())
    return text_encoded, None

  def build_inference_graph(self, examples, **kwargs):

    """Builds tensorflow graph for inference.

    Args:
      examples: a python dict involving at least following k-v fields:
        img_features: a [batch, feature_dimensions] tf.float32 tensor.
        roi_features: a [batch, number_of_regions, feature_dimensions] tf.float32 tensor.
        statement_strings: a [batch, statement_max_sent_len] tf.int64 tensor.
        statement_lengths: a [batch] tf.int64 tensor.

    Returns:
      predictions: a dict mapping from output names to output tensors.

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    model_proto = self._model_proto
    is_training = self._is_training

    # Encode image features.
    (img_encoded, img_attention
     ) = self.encode_image(
       examples['img_features'], examples['roi_features'])

    # Encode statement features.
    (stmt_encoded, stmt_attention
     ) = self.encode_text(
       text_strings=examples['statement_strings'], 
       text_lengths=examples['statement_lengths'],
       encoder=self._stmt_encoder)

    # For optional constraints.
    if model_proto.densecap_loss_weight > 0:  
      # For densecap constraint.
      (densecap_encoded, densecap_attention
       ) = self.encode_text(
         text_strings=examples['densecap_strings'], 
         text_lengths=examples['densecap_lengths'],
         encoder=self._densecap_encoder)

    if model_proto.symbol_loss_weight > 0:
      # For symbol constraint.
      (symbol_encoded, symbol_attention
       ) = self.encode_text(
         text_strings=examples['symbols'], 
         text_lengths=examples['number_of_symbols'],
         encoder=self._symbol_encoder)

    # Encode knowledge if specified.
    if model_proto.use_knowledge_branch:
      # Symbol probability distribution from pre-trained MLP model.
      symbol_logits, symbol_init_fn = mlp.model(
          model_proto.symbol_classifier, 
          examples['img_features'], 
          is_training=is_training)
      self._init_fn_list.append(symbol_init_fn)
      symbol_proba = tf.sigmoid(symbol_logits)[:, 1:]

      # Assign weight to each symbol classifier.
      with tf.variable_scope('confidence'):
        symbol_classifier_weights = tf.get_variable(
            name='weights', 
            shape=[symbol_proba.get_shape()[-1].value], 
            initializer=tf.constant_initializer(-3))
      symbol_classifier_weights = 2 * tf.sigmoid(symbol_classifier_weights)
      weights = symbol_proba * symbol_classifier_weights

      word_to_id, id_to_symbol = load_symbol_cluster(model_proto.symbol_cluster_path)
      for symbol_id, symbol_name in id_to_symbol.iteritems():
        if symbol_id != 0:
          tf.summary.scalar(
              'confidence/{}'.format(symbol_name), 
              symbol_classifier_weights[symbol_id - 1])

      # Add encoded symbol prediction as a residual branch.
      symbol_embedding_mat = self._symbol_encoder.embedding_weights[1:, :]
      symbol_pred_encoded = tf.matmul(weights, symbol_embedding_mat)
      img_encoded += symbol_pred_encoded

    # Joint embedding and cosine distance computation.
    predictions = {
      'image_id': examples['image_id'],
      'img_encoded': tf.nn.l2_normalize(img_encoded, 1),
      'stmt_encoded': tf.nn.l2_normalize(stmt_encoded, 1),
    }
    if model_proto.densecap_loss_weight > 0:  
      predictions.update({
          'dense_encoded': tf.nn.l2_normalize(densecap_encoded, 1)})
    if model_proto.symbol_loss_weight > 0:
      predictions.update({
          'number_of_symbols': examples['number_of_symbols'],
          'symb_encoded': tf.nn.l2_normalize(symbol_encoded, 1)})

    return predictions


  def build_evaluation_graph(self, examples, **kwargs):
    """Builds tensorflow graph for evaluation.

    Args:
      examples: a python dict involving at least following k-v fields:
        img_features: a [batch, feature_dimensions] tf.float32 tensor.
        statement_strings: a [batch, statement_max_sent_len] tf.int64 tensor.
        statement_lengths: a [batch] tf.int64 tensor.

    Returns:
      predictions: a dict mapping from output names to output tensors.

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    model_proto = self._model_proto
    is_training = self._is_training

    image_id = examples['image_id']

    # Encode image features.
    (img_encoded, img_attention
     ) = self.encode_image(
       examples['img_features'], examples['roi_features'])

    # Encode statement features.
    statement_strings = examples['eval_statement_strings']
    statement_lengths = examples['eval_statement_lengths']

    (number_of_val_stmts_per_image, max_stmt_len
     ) = statement_strings.get_shape().as_list()[1:]
    statement_strings_reshaped = tf.reshape(statement_strings, [-1, max_stmt_len])
    statement_lengths_reshaped = tf.reshape(statement_lengths, [-1])

    (stmt_encoded, stmt_attention
     ) = self.encode_text(
       text_strings=statement_strings_reshaped, 
       text_lengths=statement_lengths_reshaped,
       encoder=self._stmt_encoder)

    if model_proto.symbol_loss_weight > 0:
      # For symbol constraint.
      (symbol_encoded, symbol_attention
       ) = self.encode_text(
         text_strings=examples['symbols'], 
         text_lengths=examples['number_of_symbols'],
         encoder=self._symbol_encoder)

    # Encode knowledge if specified.
    if model_proto.use_knowledge_branch:
      # Symbol probability distribution from pre-trained MLP model.
      symbol_logits, symbol_init_fn = mlp.model(
          model_proto.symbol_classifier, 
          examples['img_features'], 
          is_training=is_training)
      self._init_fn_list.append(symbol_init_fn)
      symbol_proba = tf.sigmoid(symbol_logits)[:, 1:]

      # Assign weight to each symbol classifier.
      with tf.variable_scope('confidence'):
        symbol_classifier_weights = tf.get_variable(
            name='weights', 
            shape=[symbol_proba.get_shape()[-1].value], 
            initializer=tf.constant_initializer(-1))
      symbol_classifier_weights = 2 * tf.sigmoid(symbol_classifier_weights)
      weights = symbol_proba * symbol_classifier_weights

      # Add encoded symbol prediction as a residual branch.
      symbol_embedding_mat = self._symbol_encoder.embedding_weights[1:, :]
      symbol_pred_encoded = tf.matmul(weights, symbol_embedding_mat)
      img_encoded += symbol_pred_encoded

    # Joint embedding and cosine distance computation.
    img_encoded = tf.nn.l2_normalize(img_encoded, 1)
    stmt_encoded = tf.nn.l2_normalize(stmt_encoded, 1)
    stmt_encoded = tf.reshape(
        stmt_encoded, 
        [-1, number_of_val_stmts_per_image, stmt_encoded.get_shape()[-1].value])

    distance = 1 - tf.reduce_sum(tf.multiply(tf.expand_dims(img_encoded, 1), stmt_encoded), axis=2)
    predictions = {
      'image_id': image_id,
      'distance': distance,
    }
    return predictions


  def build_loss(self, predictions, **kwargs):
    """Build tensorflow graph for computing loss.

    Args:
      predictions: a dict mapping from names to predicted tensors, involving:
        img_encoded: a [batch, common_dimensions] tf.float32 tensor.
        stmt_encoded: a [batch, common_dimensions] tf.float32 tensor.

    Returns:
      loss_dict: a dict mapping from names to loss tensors.
    """
    loss_dict = {}

    model_proto = self._model_proto
    is_training = self._is_training
    mining_fn = self._mining_fn

    image_id = predictions['image_id']
    img_encoded = predictions['img_encoded']
    stmt_encoded = predictions['stmt_encoded']

    # Compute the triplet loss.
    margin = model_proto.triplet_margin
    keep_prob = model_proto.joint_emb_dropout_keep_prob

    def distance_fn(x, y):
      """Distance function."""
      distance = slim.dropout(tf.multiply(x, y), keep_prob,
          is_training=is_training)
      distance = 1 - tf.reduce_sum(distance, 1)
      return distance

    def refine_fn(pos_indices, neg_indices):
      """Refine function."""
      pos_ids = tf.gather(image_id, pos_indices)
      neg_ids = tf.gather(image_id, neg_indices)

      masks = tf.not_equal(pos_ids, neg_ids)
      pos_indices = tf.boolean_mask(pos_indices, masks)
      neg_indices = tf.boolean_mask(neg_indices, masks)
      return pos_indices, neg_indices

    # The main loss terms.
    loss_img_stmt, summary = triplet_loss_wrap_func(
        img_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin,
        'img_stmt')
    loss_stmt_img, summary = triplet_loss_wrap_func(
        stmt_encoded, img_encoded, distance_fn, mining_fn, refine_fn, margin,
        'stmt_img')

    loss_dict = {
      'triplet_img_stmt': loss_img_stmt,
      'triplet_stmt_img': loss_stmt_img,
    }

    # For optional constraints.
    if model_proto.densecap_loss_weight > 0:
      dense_encoded = predictions['dense_encoded']
      loss_dense_img, summary = triplet_loss_wrap_func(
          dense_encoded, img_encoded, distance_fn, mining_fn, refine_fn, margin,
          'dense_img')
      loss_dense_stmt, summary = triplet_loss_wrap_func(
          dense_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin,
          'dense_stmt')
      loss_dict.update({
        'triplet_dense_img': loss_dense_img * model_proto.densecap_loss_weight,
        'triplet_dense_stmt': loss_dense_stmt * model_proto.densecap_loss_weight,
      })

    if model_proto.symbol_loss_weight > 0:
      number_of_symbols = predictions['number_of_symbols']
      symb_encoded = predictions['symb_encoded']

      # Since not all images have symbol annotations. Mask them out.
      indices = tf.squeeze(tf.where(tf.greater(number_of_symbols, 0)), axis=1)
      symb_encoded = tf.gather(symb_encoded, indices)
      img_encoded = tf.gather(img_encoded, indices)
      stmt_encoded = tf.gather(stmt_encoded, indices)

      loss_symb_img, summary = triplet_loss_wrap_func(
          symb_encoded, img_encoded, distance_fn, mining_fn, refine_fn, margin,
          'symb_img')
      loss_symb_stmt, summary = triplet_loss_wrap_func(
          symb_encoded, stmt_encoded, distance_fn, mining_fn, refine_fn, margin,
          'symb_stmt')
      loss_dict.update({
        'triplet_symb_img': loss_symb_img * model_proto.symbol_loss_weight,
        'triplet_symb_stmt': loss_symb_stmt * model_proto.symbol_loss_weight,
      })

    return loss_dict
