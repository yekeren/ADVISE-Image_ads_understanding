
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import feature_extractors_pb2

from feature_extractors.mobilenet_v1_extractor import MobilenetV1Extractor
from feature_extractors.inception_v4_extractor import InceptionV4Extractor
from feature_extractors.vgg_16_extractor import VGG16Extractor
from feature_extractors.fc_extractor import FCExtractor

def build(config):
  """Build feature extractor from config.

  Args:
    config: an instance of FeatureExtractor proto.

  Returns:
    feature_extractor: an instance of FeatureExtractor.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, feature_extractors_pb2.FeatureExtractor):
    raise ValueError('Config is not an instance of FeatureExtractor.')

  which_oneof = config.WhichOneof('feature_extractor')

  if 'mobilenet_v1_extractor' == which_oneof:
    return MobilenetV1Extractor(config.mobilenet_v1_extractor)

  if 'inception_v4_extractor' == which_oneof:
    return InceptionV4Extractor(config.inception_v4_extractor)

  if 'vgg_16_extractor' == which_oneof:
    return VGG16Extractor(config.vgg_16_extractor)

  if 'fc_extractor' == which_oneof:
    return FCExtractor(config.fc_extractor)

  raise ValueError('Invalid feature extractor %s.' % (which_oneof))
