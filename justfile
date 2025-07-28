LlaVa_OneV_mmvu:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=2
    python "/home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py" \
        --model_path "/home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov" \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_mmvu.json->mmvu->" \
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

Llama_3_2V_11B_cot_HallusionBench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=7
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/luqi/model/Llama-3.2V-11B-cot \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/HallusionBench->HallusionBench->train" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_first

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

Llama_3_2_11B_Vision_HallusionBench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=4
    python /home/gwj/omni-video-r1/luqi/Experiment/llama_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/Llama-3.2-11B-Vision-Instruct \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/HallusionBench->HallusionBench->train" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_fisrt


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


R1-Onevision-7B_HallusionBench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=8
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-onevision-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/HallusionBench->HallusionBench->train" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_third

R1-Onevision-7B_Video_Hullucer:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=6
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-onevision-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_third

Vision-R1-7B-HallusionBench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=4
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/luqi/model/Vision-R1-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/HallusionBench->HallusionBench->train" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_first

Vision-R1-7B-Video_Hallucer:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=8
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/luqi/model/Vision-R1-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_first


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

R1-VL-7B_Hallusion:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=1
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/HallusionBench->HallusionBench->train" \
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

R1-VL-7B_Video_Hallucer:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=6
    python /home/gwj/omni-video-r1/luqi/Experiment/qwen_eval.py \
        --model_path /home/gwj/omni-video-r1/eval/model/R1-VL-7B \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer->" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 64 \
        --tips the_second

LLava_onevision_7B_MathVista:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=8
    python /home/gwj/omni-video-r1/luqi/Experiment/LLaVA-OneVision-7B_single.py \
        --model_path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MathVista->MathVista->testmini" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_Forth

LLava_onevision_7B_ChartQA:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=2
    python /home/gwj/omni-video-r1/luqi/Experiment/LLaVA-OneVision-7B_single.py \
        --model_path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/ChartQA->ChartQA->test" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_Third


LLava_onevision_7B_HallusionBench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=1
    python /home/gwj/omni-video-r1/luqi/Experiment/LLaVA-OneVision-7B_single.py \
        --model_path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/HallusionBench->HallusionBench->train" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_first

LLava_onevision_7B_MMbench:
    export DECORD_EOF_RETRY_MAX=20480
    export CUDA_VISIBLE_DEVICES=4
    python /home/gwj/omni-video-r1/luqi/Experiment/LLaVA-OneVision-7B_single.py \
        --model_path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --dataset "/home/gwj/omni-video-r1/data/eval_data/MMBench/en->MMBench->validation" \
        --cot TA \
        --savepath /home/gwj/omni-video-r1/luqi/eval_result \
        --batchsize 1 \
        --tips the_first


LLava_onevision_7B_videommmu:
    export CUDA_VISIBLE_DEVICES=3
    export PYTHONWARNINGS=ignore
    export TOKENIZERS_PARALLELISM=false
    python3 /home/gwj/omni-video-r1/luqi/Experiment/video_demo.py \
        --model-path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --video_path /home/gwj/omni-video-r1/data/eval_data/Evaluation/VideoMMMU \
        --output_dir /home/gwj/omni-video-r1/luqi/eval_result \
        --output_name videommmu_third \
        --dataset_path "/home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_videommmu.json->videommmu"

LLava_onevision_7B_Temcompass:
    export CUDA_VISIBLE_DEVICES=8
    export PYTHONWARNINGS=ignore
    export TOKENIZERS_PARALLELISM=false
    python3 /home/gwj/omni-video-r1/luqi/Experiment/video_demo.py \
        --model-path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --video_path /home/gwj/omni-video-r1/data/eval_data/Evaluation/TempCompass \
        --output_dir /home/gwj/omni-video-r1/luqi/eval_result \
        --output_name tempcompass_third \
        --dataset_path "/home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_tempcompass.json->tempcompass"

LLava_onevision_7B_mmvu:
    export CUDA_VISIBLE_DEVICES=5
    export PYTHONWARNINGS=ignore
    export TOKENIZERS_PARALLELISM=false
    python3 /home/gwj/omni-video-r1/luqi/Experiment/video_demo.py \
        --model-path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --video_path /home/gwj/omni-video-r1/data/eval_data/Evaluation/MMVU/videos \
        --output_dir /home/gwj/omni-video-r1/luqi/eval_result \
        --output_name mmvu_third \
        --dataset_path "/home/gwj/omni-video-r1/data/eval_data/Evaluation/Video-R1-eval/eval_mmvu.json->mmvu"

LLava_onevision_7B_VideoHallucer:
    export CUDA_VISIBLE_DEVICES=9
    export PYTHONWARNINGS=ignore
    export TOKENIZERS_PARALLELISM=false
    python3 /home/gwj/omni-video-r1/luqi/Experiment/video_demo.py \
        --model-path /home/gwj/omni-video-r1/eval/model/llava-onevision-qwen2-7b-ov \
        --video_path /home/gwj/omni-video-r1/data/eval_data/VideoHallucer \
        --output_dir /home/gwj/omni-video-r1/luqi/eval_result \
        --output_name LLava_onevision_7B_VideoHallucer_second \
        --dataset_path "/home/gwj/omni-video-r1/data/eval_data/VideoHallucer/Video_Hullucer.json->Video_Hullucer"