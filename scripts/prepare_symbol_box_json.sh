#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/object_detection/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/object_detection/tensorflow_models/slim:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=2
python "tools/prepare_symbol_box_json.py" \
    --model_proto="configs/ads_detection.pbtxt" \
    --label_map="../object_detection/configs/ads_symbols_label_map.pbtxt" \
    --checkpoint="../object_detection/models.advise/ads-challenge/train/model.ckpt-10409" \
    --image_dir="data/test_images/" \
    --action_reason_annot_path="data/test/QA_Combined_Action_Reason_test.json" \
    --output_json="output/symbol_box_test.json" \
    > "log/det_test.log" 2>&1 &

export CUDA_VISIBLE_DEVICES=3
python "tools/prepare_symbol_box_json.py" \
    --model_proto="configs/ads_detection.pbtxt" \
    --label_map="../object_detection/configs/ads_symbols_label_map.pbtxt" \
    --checkpoint="../object_detection/models.advise/ads-challenge/train/model.ckpt-10409" \
    --image_dir="data/train_images/" \
    --action_reason_annot_path="data/train/QA_Combined_Action_Reason_train.json" \
    --output_json="output/symbol_box_train.json" \
    > "log/det_train.log" 2>&1 &

exit 0
