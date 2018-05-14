#!/bin/sh

PYTHON="python"  # Path to the python installed.

$PYTHON "tools/prepare_symbol_data.py" \
  --symbol_raw_annot_path="data/train/Symbols_train.json" \
  --output_json_path="output/symbol_train.json" \
  || exit -1

$PYTHON "tools/prepare_symbol_data.py" \
  --symbol_raw_annot_path="data/test/Symbols_test.json" \
  --output_json_path="output/symbol_test.json" \
  || exit -1

exit 0
