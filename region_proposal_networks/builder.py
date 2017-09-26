
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import region_proposal_networks_pb2
from region_proposal_networks.simple_proposal_network import SimpleProposalNetwork
from region_proposal_networks.multi_objects_proposal_network import MultiObjectsProposalNetwork

def build(config):
  """Build region proposal network from config.

  Args:
    config: an instance of RegionProposalNetwork proto.

  Returns:
    proposal_network: an instance of RegionProposalNetwork.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, region_proposal_networks_pb2.RegionProposalNetwork):
    raise ValueError('Config is not an instance of RegionProposalNetwork.')

  which_oneof = config.WhichOneof('region_proposal_network')

  if 'simple_proposal_network' == which_oneof:
    return SimpleProposalNetwork(config.simple_proposal_network)

  if 'multi_objects_proposal_network' == which_oneof:
    return MultiObjectsProposalNetwork(config.multi_objects_proposal_network)

  raise ValueError('Invalid region proposal network %s.' % (which_oneof))


