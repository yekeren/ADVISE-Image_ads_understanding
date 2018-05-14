#!/bin/sh

set -x

export PYTHONPATH="$PYTHONPATH:`pwd`"

mkdir -p "log"
mkdir -p "saved_results"

# Train for only 10,000 steps. 
# The results we put in the challenge trained for 200,000 steps.

#number_of_steps=200000
number_of_steps=10000

name="vse++"
#name="advise.densecap_0.1.symbol_0.1"
#name="advise.kb"

export CUDA_VISIBLE_DEVICES=0
python "train/train.py" \
    --pipeline_proto="configs/${name}.pbtxt" \
    --train_log_dir="logs/${name}/train" \
    --number_of_steps="${number_of_steps}" \
    > "log/${name}.train.log" 2>&1 &

# Also specify --restore_from if fine-tune the knowledge branch (advise.kb).
#    --restore_from="logs/advise.densecap_0.1.symbol_0.1/saved_ckpts/model.ckpt-10000" \

python "train/eval.py" \
    --pipeline_proto="configs/${name}.pbtxt" \
    --action_reason_annot_path="data/train/QA_Combined_Action_Reason_train.json" \
    --train_log_dir="logs/${name}/train" \
    --eval_log_dir="logs/${name}/eval" \
    --saved_ckpt_dir="logs/${name}/saved_ckpts" \
    --continuous_evaluation="true" \
    --number_of_steps="${number_of_steps}" \
    > "log/${name}.eval.log" 2>&1 &

wait

#########################################################
# Export the results for testing.
#########################################################
python "train/eval.py" \
    --pipeline_proto="configs/${name}_test.pbtxt" \
    --action_reason_annot_path="data/test/QA_Combined_Action_Reason_test.json" \
    --saved_ckpt_dir="logs/${name}/saved_ckpts" \
    --continuous_evaluation="false" \
    --final_results_path="saved_results/${name}.json" \
    || exit -1

exit 0
