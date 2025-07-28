import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoTokenizer,AutoModelForCausalLM
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
from datasets import load_dataset, load_from_disk
from qwen_tools import get_question_template,get_answer_template,get_data_with_templete,Extractor,Conversation
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--dataset', type=str, required=True, help="Path to the Dataset")
parser.add_argument('--cot', type=str, required=True, help="COT To inference")
parser.add_argument('--savepath', type=str, required=True, help="Path To Save")
parser.add_argument('--batchsize', type=int, required=True, default=1, help="Batch size for evaluation")
parser.add_argument('--tips', type=str, required=False, default='default', help="tips")
args = parser.parse_args()

BSZ = args.batchsize
MODEL_PATH = args.model_path
DATASET = args.dataset
COT = args.cot
TIPS = args.tips
PATHSAVE = args.savepath
DATASET_PATH = DATASET.split('->')[0]
DATASETNAME = DATASET.split('->')[1]
DATASET_SPLIT_VAL = DATASET.split('->')[2]

MODEL_NAME = MODEL_PATH.split('/')[-1]  # 输出: "R1-Onevision-7B"
MODEL_NAME = MODEL_NAME.replace("-", "_")  # 输出: "R1_Onevision_7B"
print("MODEL_NAME",MODEL_NAME)
# 加载模型

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len = 8192,   #8192
    gpu_memory_utilization=0.9,
    limit_mm_per_prompt={"image": 1, "video": 1},
    trust_remote_code=True
)

# 推理时的采样超参数设置(默认下面的设置即可)
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=4096,# 最多输出token数量
    stop_token_ids=[],
)

# 加载tokenizer 和 processor
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

for dataset_name in [DATASETNAME]:

    # 测试结果保存地址和eval数据集目录地址
    OUTPUT_PATH = f"{PATHSAVE}/eval_{dataset_name}_{COT}_{MODEL_PATH.split('/')[-2]}_{MODEL_PATH.split('/')[-1]}_{TIPS}_output.json"
    
    if DATASET_PATH.endswith('.jsonl'):
        data = Dataset.from_json(DATASET_PATH)
    elif DATASET_PATH.endswith('.json'):
        data = Dataset.from_json(DATASET_PATH)
    else:
        data = load_dataset(DATASET_PATH)[DATASET_SPLIT_VAL]
    
    #1、处理输入的数据 
    QUESTION_TEMPLATE=get_question_template(COT,MODEL_NAME) # 获取问题模板
    TYPE_TEMPLATE = get_answer_template(MODEL_NAME)
    print("QUESTION_TEMPLATE:",QUESTION_TEMPLATE)
    print(" TYPE_TEMPLATE", TYPE_TEMPLATE)
    
    #填入对话格式模板
    make_conversation_cot_image=Conversation.get_conversation_fuchtion(dataset_name) # 获取对话格式模板函数
    
    data = data.map(make_conversation_cot_image) 
    messages = []
    for x in tqdm(data): # 构造输入数据
        msg = get_data_with_templete(dataset_name,x,QUESTION_TEMPLATE,TYPE_TEMPLATE)
        messages.append(msg) #变为role content的格式
        
    print(f"=======================Total samples:{messages[0]}==============")
    
    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):    #上一次没测完的，可以接着续写
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    #2、开始处理数据
    mean_acc = []
    mean_mra = []
    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ] 
        
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]    #变为<>
        
        try:# 解析batch数据里面的图像/视频
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            
            image_idx = 0
            video_idx = 0

            llm_inputs = [] # 构造输入到llm之前的input数据（一个batch）

            for idx, prompt in enumerate(prompts): # 遍历prompts
                mm_type = batch_messages[idx][0]['content'][0]['type'] # 当前输入多模态信息的类型 video or image
                sample_mm_data = {} # 构造最终的视频输入数据
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx] # 得到视频输入数据放入sample_mm_data
                    for key, value in video_kwargs.items(): # 将视频信息放入sample_mm_data
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                
                if(MODEL_NAME=="R1_VL_7B"):
                    llm_inputs.append({ # 加入到llm_inputs中
                        "prompt": prompt,
                        "multi_modal_data": sample_mm_data,
                        # "mm_processor_kwargs": sample_video_kw,
                    })
                else:
                    llm_inputs.append({ # 加入到llm_inputs中
                        "prompt": prompt,
                        "multi_modal_data": sample_mm_data,
                        "mm_processor_kwargs": sample_video_kw,
                    })                    
                
            #（2）输入给模型
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs] # 记录一个batch推理之后的输出答案
            
        except Exception as e:
            print(e,'error:', data[i]['path'])
            batch_output_text = ['<answer>error</answer>'] * BSZ
            
        #（3）处理模型输出的结果
        for j, model_output in enumerate(batch_output_text):# sample是原始数据集
            sample = data[j + i]
            result = {}
            
            if(dataset_name == 'MathVista'):
                final_ans = Extractor.extract_mathvista_answer(sample, model_output)
            else:
                final_ans = Extractor.extract_answer_special(model_output)
                
            if final_ans == "":
                final_ans = model_output
            else:   
                if(dataset_name in {'MMBench','tempcompass','HallusionBench','Video_Hullucer'}):
                    final_ans = final_ans[0]
            if(MODEL_NAME == 'R1_Onevision_7B'):
                # if(dataset_name in {'MMBench','tempcompass','HallusionBench','Video_Hullucer'}):
                    final_ans = model_output[-1]
            
            
            q_type = sample.get("problem_type", "")
            if(dataset_name in {"mmvu","videommmu"} and q_type=="multiple choice"):
                final_ans = final_ans[0]
                
            result['question_id'] = sample['q_id']
            result['format_question'] = sample['format_question']
            result["output"] = model_output # 记录输出
            result["prediction"] = final_ans # 记录预测
            result["solution"] = sample["solution"]
            
            result["reward"] = Extractor.reward_fn(sample, final_ans, q_type) # 记录准确率奖励分数
            result['correct'] = True if result["reward"]==1.0 else False
            
            if sample['problem_type'] != 'regression':
                mean_acc.append(result["reward"])
            else:
                mean_mra.append(result["reward"])

            final_output.append(result) # 记录最后输出(原数据+输出结果)
        
        try: # 保存结果
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item() 
    if mean_mra != []:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f: # 记录正确率
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")