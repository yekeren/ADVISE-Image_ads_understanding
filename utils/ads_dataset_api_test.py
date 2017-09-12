import unittest

import cv2
from ads_dataset_api import AdsDatasetApi

dataset_root = 'raw_data/ads'

class TestAdsDatasetApi(unittest.TestCase):

  def testIndexImages(self):
    api = AdsDatasetApi()
    with self.assertRaises(ValueError):
      api._index_images(None)

    images_dict = api._index_images('%s/images' % (dataset_root))
    self.assertEqual(64832, len(images_dict))

  def testIndexActionReason(self):
    api = AdsDatasetApi()
    images_dict = api._index_images('%s/images' % (dataset_root))
    api._index_action_reason_pairs(
        '%s/annotations/QA_Action.json' % (dataset_root),
        '%s/annotations/QA_Reason.json' % (dataset_root),
        '%s/annotations/QA_Combined_Action_Reason.json' % (dataset_root))

#  def testIndexRegions(self):
#    api = AdsDatasetApi()
#    images_dict = api._index_images('%s/images' % (dataset_root))
#    meta_list = api._index_entities('%s/annotations/Symbols.json' % (dataset_root))
#    self.assertEqual(13938, len(meta_list))
#
#  def testMajorityVote(self):
#    api = AdsDatasetApi()
#    self.assertEqual(api._majority_vote([1, 1, 1, 3, 2]), (1, 3))
#    self.assertEqual(api._majority_vote([2, 2, 1, 3, 2]), (2, 3))
#    self.assertEqual(api._majority_vote([2, 2, 1, 3, 2, 3, 3, 3]), (3, 4))
#
#  def testIndexTopics(self):
#    api = AdsDatasetApi()
#    images_dict = api._index_images('%s/images' % (dataset_root))
#
#    name_to_topic, topic_to_name, topic_to_images = api._index_topics(
#        topic_list_file='%s/annotations/Topics_List.txt' % (dataset_root),
#        topic_annot_file='%s/annotations/Topics.json' % (dataset_root))
#    self.assertEqual(len(name_to_topic), 39)
#    self.assertEqual(len(topic_to_name), 39)
#    self.assertEqual(len(topic_to_images), 39)
#    expected_topic_to_name =  {
#      "1" : "restaurant",
#      "2": "chocolate",
#      "3": "chips",
#      "4": "seasoning",
#      "5": "petfood",
#      "6": "alcohol",
#      "7": "coffee",
#      "8": "soda",
#      "9": "cars",
#      "10": "electronics",
#      "11": "phone_tv_internet_providers",
#      "12": "financial",
#      "13": "education",
#      "14": "security",
#      "15": "software",
#      "16": "other_service",
#      "17": "beauty",
#      "18": "healthcare",
#      "19": "clothing",
#      "20": "baby",
#      "21": "game",
#      "22": "cleaning",
#      "23": "home_improvement",
#      "24": "home_appliance",
#      "25": "travel",
#      "26": "media",
#      "27": "sports",
#      "28": "shopping",
#      "29": "gambling",
#      "30": "environment",
#      "31": "animal_right",
#      "32": "human_right",
#      "33": "safety",
#      "34": "smoking_alcohol_abuse",
#      "35": "domestic_violence",
#      "36": "self_esteem",
#      "37": "political",
#      "38": "charities",
#      "39": "Unclear"}
#    expected_name_to_topic = dict(
#        [(v, k) for k, v in expected_topic_to_name.iteritems()])
#    self.assertEqual(topic_to_name, expected_topic_to_name)
#    self.assertEqual(name_to_topic, expected_name_to_topic)
#    self.assertEqual(set(api.get_topic_names()), 
#        set(expected_name_to_topic.keys()))
#
#    # All topic annotations.
#    meta_list = api.get_meta_list_by_topics()
#    self.assertEqual(64325, len(meta_list))
#
#    # Beauty annotations.
#    meta_list = api.get_meta_list_by_topics(topic_name_list=['beauty'])
#    self.assertEqual(5833, len(meta_list))
#
#    meta_list = api.get_meta_list_by_topics(topic_name_list=['beauty'], min_votes=1)
#    self.assertEqual(5833, len(meta_list))
#
#    meta_list = api.get_meta_list_by_topics(topic_name_list=['beauty'], min_votes=2)
#    self.assertGreater(5833, len(meta_list))
#    self.assertEqual(5818, len(meta_list))

if __name__ == '__main__':
  unittest.main()
