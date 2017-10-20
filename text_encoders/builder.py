
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import text_encoders_pb2
from text_encoders.bow_encoder import BOWEncoder
from text_encoders.rnn_encoder import RNNEncoder

def build(config):
  """Build text encoder from config.

  Args:
    config: an instance of TextEncoder proto.

  Returns:
    text_encoder: an instance of TextEncoder.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, text_encoders_pb2.TextEncoder):
    raise ValueError('Config is not an instance of TextEncoder.')

  which_oneof = config.WhichOneof('text_encoder')

  if 'bow_encoder' == which_oneof:
    return BOWEncoder(config.bow_encoder)

  if 'rnn_encoder' == which_oneof:
    return RNNEncoder(config.rnn_encoder)

  raise ValueError('Invalid text encoder %s.' % (which_oneof))

