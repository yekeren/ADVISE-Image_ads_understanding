#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/object_detection/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/object_detection/tensorflow_models/slim:$PYTHONPATH"

## Image features.
#export CUDA_VISIBLE_DEVICES=0
#python "tools/prepare_img_features.py" \
#    --action_reason_annot_path="data/train/QA_Combined_Action_Reason_train.json" \
#    --feature_extractor_name="inception_v4" \
#    --feature_extractor_scope="InceptionV4" \
#    --feature_extractor_endpoint="PreLogitsFlatten" \
#    --feature_extractor_checkpoint="zoo/inception_v4.ckpt" \
#    --image_dir="data/train_images/" \
#    --output_feature_path="output/img_features_train.npy" \
#    > "log/create_train.log" 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=1
#python "tools/prepare_img_features.py" \
#    --action_reason_annot_path="data/test/QA_Combined_Action_Reason_test.json" \
#    --feature_extractor_name="inception_v4" \
#    --feature_extractor_scope="InceptionV4" \
#    --feature_extractor_endpoint="PreLogitsFlatten" \
#    --feature_extractor_checkpoint="zoo/inception_v4.ckpt" \
#    --image_dir="data/test_images/" \
#    --output_feature_path="output/img_features_test.npy" \
#    > "log/create_test.log" 2>&1 &
#
## ROI features.
#export CUDA_VISIBLE_DEVICES=0
#python "tools/prepare_roi_features.py" \
#    --bounding_box_json="output/symbol_box_train.json" \
#    --feature_extractor_name="inception_v4" \
#    --feature_extractor_scope="InceptionV4" \
#    --feature_extractor_endpoint="PreLogitsFlatten" \
#    --feature_extractor_checkpoint="zoo/inception_v4.ckpt" \
#    --image_dir="data/train_images/" \
#    --output_feature_path="output/roi_features_train.npy" \
#> "log/create_train.log" 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=1
#python "tools/prepare_roi_features.py" \
#    --bounding_box_json="output/symbol_box_test.json" \
#    --feature_extractor_name="inception_v4" \
#    --feature_extractor_scope="InceptionV4" \
#    --feature_extractor_endpoint="PreLogitsFlatten" \
#    --feature_extractor_checkpoint="zoo/inception_v4.ckpt" \
#    --image_dir="data/test_images/" \
#    --output_feature_path="output/roi_features_test.npy" \
#> "log/create_test.log" 2>&1 &

# ROI features using densecap.
export CUDA_VISIBLE_DEVICES=0
python "tools/prepare_roi_features.py" \
    --bounding_box_json="output/densecap_train.json" \
    --feature_extractor_name="inception_v4" \
    --feature_extractor_scope="InceptionV4" \
    --feature_extractor_endpoint="PreLogitsFlatten" \
    --feature_extractor_checkpoint="zoo/inception_v4.ckpt" \
    --image_dir="data/train_images/" \
    --output_feature_path="output/densecap_roi_features_train.npy" \
> "log/create_train.log" 2>&1 &

export CUDA_VISIBLE_DEVICES=1
python "tools/prepare_roi_features.py" \
    --bounding_box_json="output/densecap_test.json" \
    --feature_extractor_name="inception_v4" \
    --feature_extractor_scope="InceptionV4" \
    --feature_extractor_endpoint="PreLogitsFlatten" \
    --feature_extractor_checkpoint="zoo/inception_v4.ckpt" \
    --image_dir="data/test_images/" \
    --output_feature_path="output/densecap_roi_features_test.npy" \
> "log/create_test.log" 2>&1 &

exit 0
