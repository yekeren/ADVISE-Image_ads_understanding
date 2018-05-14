#!/bin/sh

# python "tools/prepare_word_embedding.py" \
#     --vocab_path="output/densecap_vocab.txt" \
#     --output_emb_path="output/densecap_vocab_200d.npy" \
#     --output_vocab_path="output/densecap_vocab_200d.txt" \
#     || exit -1

python "tools/prepare_word_embedding.py" \
    --vocab_path="output/symbol_vocab.txt" \
    --output_emb_path="output/symbol_vocab_200d.npy" \
    --output_vocab_path="output/symbol_vocab_200d.txt" \
    || exit -1


exit 0
