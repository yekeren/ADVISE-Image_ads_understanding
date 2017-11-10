import json
import sys
from utils.ads_api import AdsApi
from nltk import pos_tag, word_tokenize

api = AdsApi('configs/ads_api.config')

meta_list = api.get_meta_list()
result = {}

for i, meta in enumerate(meta_list[:10]):
  if i % 100 == 0:
    print >> sys.stderr, 'On image %s/%s' % (i, len(meta_list))
  if 'statements' in meta:
    for statement in meta['statements']:
      action, topic, reason = [], [], []
      flag = False
      for (word, postag) in pos_tag(word_tokenize(statement.lower())):
        if 'because' == word:
          flag = True
        elif not flag:
          if 'V' in postag: 
            action.append(word)
          elif 'NN' in postag and word != 'i': 
            topic.append(word)
        else:
          if 'JJ' in postag or 'NN' in postag and word != 'i': 
            reason.append(word)
      if action and topic and reason:
        result[meta['image_id']] = {
            'action': action,
            'topic': topic,
            'reason': reason
            }
with open('attributes.json', 'w') as fp:
  fp.write(json.dumps(result))
