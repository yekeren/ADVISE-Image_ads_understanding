
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import ads_emb_model_pb2
from text_embedders.bow_embedder import BOWEmbedder
from text_embedders.rnn_embedder import RNNEmbedder

def build(config):
  """Build text embedder from config.

  Args:
    config: an instance of TextEmbedder proto.

  Returns:
    text_embedder: an instance of TextEmbedder.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, ads_emb_model_pb2.TextEmbedder):
    raise ValueError('Config is not an instance of TextEmbedder.')

  which_oneof = config.WhichOneof('text_embedder')

  if 'bow_embedder' == which_oneof:
    return BOWEmbedder(config.bow_embedder)

  if 'rnn_embedder' == which_oneof:
    return RNNEmbedder(config.rnn_embedder)

  raise ValueError('Invalid text embedder %s.' % (which_oneof))

