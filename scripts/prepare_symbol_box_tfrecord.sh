#!/bin/sh

set -x

export PYTHONPATH=${HOME}/work2/object_detection/tensorflow_models:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${HOME}/work2/object_detection

export CUDA_VISIBLE_DEVICES=1

# Create tf record.
output_dir="output"
vis_dir="tmp"

mkdir -p "${vis_dir}"
mkdir -p "${output_dir}"

python "tools/prepare_symbol_box_tfrecord.py" \
  --vis_dir="${vis_dir}" \
  --img_dir="data/train_images" \
  --symbol_annot_path="data/train/Symbols_train.json" \
  --train_output_path="${output_dir}/symbol_box.train.record" \
  --valid_output_path="${output_dir}/symbol_box.valid.record" \
  --label_map_path="data/symbol_box_label_map.pbtxt" \
  > "log/box_train_val.log" 2>&1 &

python "tools/prepare_symbol_box_tfrecord.py" \
  --vis_dir="${vis_dir}" \
  --img_dir="data/test_images" \
  --symbol_annot_path="data/test/Symbols_test.json" \
  --test_output_path="${output_dir}/symbol_box.test.record" \
  --label_map_path="data/symbol_box_label_map.pbtxt" \
  > "log/box_test.log" 2>&1 &

exit 0
