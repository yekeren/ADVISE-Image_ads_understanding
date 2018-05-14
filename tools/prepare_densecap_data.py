
import os
import sys
import json
import string
import argparse

import nltk
from readers.utils import load_action_reason_annots
from readers.utils import load_densecap_raw_annots


def main(args):
  """Main."""
  # Load dataset.
  annots = load_action_reason_annots(args.action_reason_annot_path)
  print >> sys.stderr, 'Load annotations for %i images.' % (len(annots))

  densecap_annots = load_densecap_raw_annots(args.densecap_raw_annot_path)
  print >> sys.stderr, 'Load densecap annotations for %i images.' % (len(densecap_annots))

  examples = {}
  for image_id in annots:
    assert image_id in densecap_annots
    regions = []
    for region in densecap_annots[image_id]:
      regions.append({
          'id': 1,
          'name': region['caption'],
          'score': round(region['score'], 3),
          'bbox': {
            'xmin': round(region['xmin'], 3),
            'xmax': round(region['xmax'], 3),
            'ymin': round(region['ymin'], 3),
            'ymax': round(region['ymax'], 3),
          }
          })
    examples[image_id] = {
      'image_id': image_id,
      'regions': regions
    }

  # Write to output json file.
  with open(args.output_json_path, 'w') as fp:
    fp.write(json.dumps(examples))
  print >> sys.stderr, 'Wrote %i records to %s.' % (
      len(examples), args.output_json_path)
  print >> sys.stderr, 'Done'

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--action_reason_annot_path', type=str,
      default='data/train/QA_Combined_Action_Reason_train.json', 
      help='Path to the action-reason annotation file.')
  parser.add_argument(
      '--densecap_raw_annot_path', type=str,
      default='data/additional/Densecap_raw.json', 
      help='Path to the raw densecap annotation file.')
  parser.add_argument(
      '--output_json_path', type=str,
      default='output/densecap_train.json', 
      help='Path to the output densecap json file.')

  args = parser.parse_args()
  assert os.path.isfile(args.action_reason_annot_path)

  print >> sys.stderr, 'parsed input parameters:'
  print >> sys.stderr, json.dumps(vars(args), indent=2)

  main(args)

  exit(0)
