#!/bin/bash

# 提供的变量
MODEL_NAME_OR_PATH=""
INPUT_DATASET=""
OUTPUT_DIR="/"

# 从MODEL_NAME_OR_PATH和INPUT_DATASET提取最后一部分的文件名或目录名
MODEL_BASENAME=$(basename "$MODEL_NAME_OR_PATH")
DATASET_BASENAME=$(basename "$INPUT_DATASET" .json)  # 移除.json扩展名

# 拼接新的输出目录路径
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_BASENAME}_${MODEL_BASENAME}.json"

python batch_generation_IQA.py \
    --model_id ${MODEL_NAME_OR_PATH} \
    --input_file ${INPUT_DATASET} \
    --output_file ${OUTPUT_FILE}