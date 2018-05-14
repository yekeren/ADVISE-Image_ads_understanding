#!/bin/sh

python "tools/prepare_symbol_data.py" || exit -1

python "tools/prepare_symbol_data.py" \
    --symbol_raw_annot_path="data/test/Symbols_test.json" \
    --output_json_path="output/symbol_test.json" \
    || exit -1

exit 0
