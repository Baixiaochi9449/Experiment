#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
python3 /home/gwj/omni-video-r1/luqi/Experiment/video_demo.py \
    --model-path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
    --video_path /home/gwj/omni-video-r1/data/eval_data/Evaluation/VideoMMMU \
    --output_dir /home/gwj/omni-video-r1/luqi/eval_result \
    --output_name text_mmvu_third \
    --dataset_path /home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_videommmu.json
