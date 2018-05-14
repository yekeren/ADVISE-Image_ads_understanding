
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from models import vse_model
from models import advise_model


def build(config, is_training=False):
  """Builds a Model based on the config.

  Args:
    config: a model_pb2.Model instance.
    is_training: True if this model is being built for training.

  Returns:
    a Model instance.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, model_pb2.Model):
    raise ValueError('The config has to be an instance of model_pb2.Model.')

  model = config.WhichOneof('model')

  if 'vse_model' == model:
    return vse_model.Model(config.vse_model, is_training)

  if 'advise_model' == model:
    return advise_model.Model(config.advise_model, is_training)

  raise ValueError('Unknown model: {}'.format(model))
