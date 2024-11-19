CONFIGSTR="configs/polIter_metallama3.1instruct_rloo_CD.jsonnet"
APP_DIRECTORY="/scr/kanishkg/vine/experiments/"

export APP_SEED="2746318213"
export WANDB_RUN_ID="rloo" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
     src/treetune/main.py --configs "$CONFIGSTR" \
     run_iteration_loop

