Llama-3-VILA1_5-8B_MathVista:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=4
    python "/home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py" \
        --model_path "/home/gwj/omni-video-r1/luqi/model/Llama-3-VILA1.5-8B" \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MathVista->MathVista->testmini" \
        --cot TA \
        --savepath "/home/gwj/omni-video-r1/luqi/eval_result" \
        --batchsize 64 \
        --tips test


Llama_3_2V_11B_cot_MathVista:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=0
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/luqi/model/Llama-3.2V-11B-cot \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MathVista->MathVista->testmini" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_forth

Llama_3_2V_11B_cot_ChartQA:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=9
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/luqi/model/Llama-3.2V-11B-cot \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/ChartQA->ChartQA->test" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_forth

Llama_3_2V_11B_cot_ClevrMath:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=2
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/luqi/model/Llama-3.2V-11B-cot \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/ClevrMath->ClevrMath->test" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_forth

Llama_3_2V_11B_cot_MMBench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=3
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/luqi/model/Llama-3.2V-11B-cot \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MMBench/en->MMBench->validation" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_forth

Llama_3_2_11B_Vision_Instruct_MathVista:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=2
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/Llama-3.2-11B-Vision-Instruct \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MathVista->MathVista->testmini" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_third

Llama_3_2_11B_Vision_Instruct_ChartQA:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=0
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/Llama-3.2-11B-Vision-Instruct \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/ChartQA->ChartQA->test" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_third


Llama_3_2_11B_Vision_Instruct_MMbench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=1
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/Llama-3.2-11B-Vision-Instruct \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MMBench/en->MMBench->validation" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_third


llava-onevision-qwen2-7b-ov_ChartQA:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=5
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/ChartQA->ChartQA->test" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips test


R1-Onevision-7B_MathVista:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=5
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-Onevision-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MathVista->MathVista->testmini" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips test


R1-VL-7B_MathVista:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=4
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MathVista->MathVista->testmini" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_second

R1-VL-7B_ChartQA:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=5
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/ChartQA->ChartQA->test" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_third

R1-VL-7B_MMBench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=4
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MMBench/en->MMBench->validation" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_third

R1-VL-7B_tempcompass:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=6
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_tempcompass.json->tempcompass->" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_third

R1-VL-7B_videommmu:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=6
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_videommmu.json->videommmu->" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_third

R1-VL-7B_mmvu:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=0
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_mmvu.json->mmvu->" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_third

LLava_onevision_7B_MathVista:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=7
    python /home/gwj/omni-video-r1/luqi/Experiment/LLaVA-OneVision-7B.py \
        --model_path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MathVista->MathVista->testmini" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_second


