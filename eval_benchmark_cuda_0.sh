#!/bin/bash
# run_models.sh


model_paths=(
    "/opt/data/private/gwj/Omni-Vision-RL/train/outputs/Main/GRPO/Qwen2.5-VL-7B-GRPO/checkpoint-1000"
    "/opt/data/private/gwj/Omni-Vision-RL/train/outputs/Main/GRPO/Qwen2.5-VL-7B-GRPO/checkpoint-1500"
    "/opt/data/private/gwj/Omni-Vision-RL/train/outputs/Main/GRPO/Qwen2.5-VL-7B-GRPO/checkpoint-2000"
    "/opt/data/private/gwj/Omni-Vision-RL/train/outputs/Main/GRPO/Qwen2.5-VL-7B-GRPO/checkpoint-2500"
    "/opt/data/private/gwj/Omni-Vision-RL/train/outputs/Main/GRPO/Qwen2.5-VL-7B-GRPO/checkpoint-3000"
)


######### Dataset ###########
datasets=(
   "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->None"
   "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->None"
   "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->None"
   "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->None"
   "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->None"
)
# Options: 
# "/opt/data/private/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_mmvu.json->mmvu->None"
# "/opt/data/private/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_mvbench_modify_2.json"
# "/opt/data/private/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_tempcompass.json->tempcompass->None"
# "/opt/data/private/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_videommmu.json"
# "/opt/data/private/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_vsibench.json"
# "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/MMSTAR->MMSTAR->val"
# "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/MMBench/en->MMBench->validation"
# "/opt/data/private/gwj/Omni-Vision-RL/data/train_data/A-OKVQA->AOKVQA->validation"
# "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/ChartQA->ChartQA->test"
# "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/ClevrMath->ClevrMath->test"
# "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/MathVista->MathVista->testmini"
# "/opt/data/private/gwj/Omni-Vision-RL/train/experiment/InfoNCELoss/data/test.json->MMathCoT->None"
# "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/HallusionBench->HallusionBench->train"
# "/opt/data/private/gwj/Omni-Vision-RL/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->None"


methods=(
    "TA2"
    "TA2"
    "TA2"
    "TA2"
    "TA2"
)
# Options:
# Think + Answer TA
# Prethink + Caption + Deepthink + Answer PCDA
# Info + Think + Answer ITA


savePaths=(
    "/opt/data/private/gwj/Omni-Vision-RL/eval/experiment/EXPERIMENT/MAIN/GRPO"
    "/opt/data/private/gwj/Omni-Vision-RL/eval/experiment/EXPERIMENT/MAIN/GRPO"
    "/opt/data/private/gwj/Omni-Vision-RL/eval/experiment/EXPERIMENT/MAIN/GRPO"
    "/opt/data/private/gwj/Omni-Vision-RL/eval/experiment/EXPERIMENT/MAIN/GRPO"
    "/opt/data/private/gwj/Omni-Vision-RL/eval/experiment/EXPERIMENT/MAIN/GRPO"
)
# Options:
# "/opt/data/private/gwj/Omni-Vision-RL/eval/eval_results"
# "/opt/data/private/gwj/Omni-Vision-RL/eval/experiment/fewSample"

tips=(
    "dft"
    "dft"
    "dft"
    "dft"
    "dft"
)


export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    dataset="${datasets[$i]}"
    method="${methods[$i]}"
    savepath="${savePaths[$i]}"
    tip="${tips[$i]}"
    CUDA_VISIBLE_DEVICES=0 python /opt/data/private/gwj/Omni-Vision-RL/eval/eval_scripts/eval_benchmark.py --model_path "$model" --dataset "$dataset" --cot "$method" --savepath "$savepath" --tips "$tip"
done
