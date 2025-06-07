#!/bin/bash
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345

# MODIFY HERE: please prepare the env related variables
PR1_PATH="<path to pr1>" # path to pr1
CHECKPOINT_PATH="<path to checkpoint>" # directory to save the checkpoint
RUN_NAME="<run_name>" # describe what your experiment is about

# Default Setting
OUTPUT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}" # path to save the output
SRC_PATH="${OUTPUT_DIR}/src" # path to backup the source code

export LOG_DIR="${OUTPUT_DIR}/logs" # path to save the log
export WANDB_PROJECT="PR1" # project name in wandb
export WANDB_TAGS="qwen2-vl-ocr" # tags for the experiment in wandb

if [ ! -d "${OUTPUT_DIR}"/src ]; then
    mkdir -p ${OUTPUT_DIR}/src
fi

# backup the source code
cp -r ${PR1_PATH}/src ${SRC_PATH}
mkdir -p ${LOG_DIR}

# run the training
torchrun \
    --nproc_per_node="7" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${PR1_PATH}/src/open_r1/grpo_vllm.py \
    --deepspeed ${PR1_PATH}/configs/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name <your dataset path> \
    --max_prompt_length 2048 \
    --max_completion_length 2048 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 7 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 400 \
    --reward_funcs "pr1_ocr" \
    --save_only_model true \
    --system_prompt_template "default" \
    --question_template "pr1_ocr" \
    --train_sample_size 5000 \
    --skip_special_tokens false \
    --answer_template "pr1_ocr" 
