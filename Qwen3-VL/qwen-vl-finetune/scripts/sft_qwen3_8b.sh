#!/bin/bash


# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=8 WANDB_MODE=offline bash scripts/sft_qwen3_8b.sh

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# llm=Qwen/Qwen3-VL-8B-Instruct  # Using HuggingFace model ID
llm=../huggingface/Qwen3-VL-8B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=16
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=SpatialCorpus_Absolute_Depth,SpatialCorpus_Absolute_Distance,SpatialCorpus_Camera_Extrinsics,SpatialCorpus_Camera_Intrinsics,SpatialCorpus_Camera_Motion,SpatialCorpus_Homography,SpatialCorpus_Existence,SpatialCorpus_Size,SpatialCorpus_Point_Tracking,SpatialCorpus_Reconstruction,SpatialCorpus_Relative_Depth,SpatialCorpus_Relative_Distance,SpatialCorpus_Relative_Size,SpatialCorpus_3D_Object_Detection,SpatialCorpus_2D_2D_Rotation,SpatialCorpus_2D_3D_Rotation,SpatialCorpus_Spatial_Map,SpatialCorpus_View_Matching,SpatialCorpus_View_Identification,SpatialCorpus_2D_Object_Detection

# Output configuration
run_name="qwen3vl"
output_dir=./output_8b_SpatialCorpus

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 518400 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ${entry_file} ${args}