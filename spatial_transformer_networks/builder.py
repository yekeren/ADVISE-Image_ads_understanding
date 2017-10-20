
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import spatial_transformer_networks_pb2
from spatial_transformer_networks.simple_transformer import SimpleTransformer
from spatial_transformer_networks.affine_transformer import AffineTransformer

def build(config):
  """Build spatial transformer network from config.

  Args:
    config: an instance of SpatialTransformer proto.

  Returns:
    transformer: an instance of SpatialTransformer.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, spatial_transformer_networks_pb2.SpatialTransformer):
    raise ValueError('Config is not an instance of SpatialTransformer.')

  which_oneof = config.WhichOneof('spatial_transformer')

  if 'simple_transformer' == which_oneof:
    return SimpleTransformer(config.simple_transformer)

  if 'affine_transformer' == which_oneof:
    return AffineTransformer(config.affine_transformer)

  raise ValueError('Invalid spatial transformer network %s.' % (which_oneof))



