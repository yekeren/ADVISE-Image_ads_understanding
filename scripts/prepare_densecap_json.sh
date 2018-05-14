#!/bin/sh

python "tools/prepare_densecap_data.py" \
    --action_reason_annot_path="data/train/QA_Combined_Action_Reason_train.json" \
    --output_json_path="output/densecap_train.json" \
    || exit -1

python "tools/prepare_densecap_data.py" \
    --action_reason_annot_path="data/test/QA_Combined_Action_Reason_test.json" \
    --output_json_path="output/densecap_test.json" \
    || exit -1

exit 0
