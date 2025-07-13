# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoTokenizer, MllamaForConditionalGeneration
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
from datasets import load_dataset, load_from_disk
from llama_tools import get_question_template,get_answer_template,get_data_with_templete,Extractor,Conversation
import requests
from PIL import Image
from torch.nn import DataParallel
from torchvision.transforms import Compose, Resize
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


BSZ = args.batchsize   #11B的模型 2已经是极限了
MODEL_PATH = args.model_path
DATASET = args.dataset
COT = args.cot
TIPS = args.tips
PATHSAVE = args.savepath
DATASET_PATH = DATASET.split('->')[0]
DATASETNAME = DATASET.split('->')[1]
DATASET_SPLIT_VAL = DATASET.split('->')[2]

MODEL_NAME = MODEL_PATH.split('/')[-1]  # 输出: "R1-Onevision-7B"
MODEL_NAME = MODEL_NAME.replace("-", "_")  # 输出: 
print("MODEL_NAME",MODEL_NAME)

#加载模型
model_name = "llava_qwen"
device = "cuda"
tokenizer, model, image_processor, max_length = load_pretrained_model(MODEL_PATH, None, model_name, device_map='auto')  

model.eval()

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
    
    make_conversation_cot_image=Conversation.get_conversation_fuchtion(dataset_name) # 获取对话格式模板函数
    data = data.map(make_conversation_cot_image)   #存储了所有用到的数据

    all_urls=[]
    for x in tqdm(data): # 构造输入数据
        all_urls.append(x['url']) # 存储所有的url地址   #可能是地址，也可能是已经加载的图片
        
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


    mean_acc = []
    mean_mra = []
    for i in tqdm(range(start_idx, len(data), BSZ), desc="Processing batches"):
        batch_data = data[i:i + BSZ] 
        batch_urls= all_urls[i:i + BSZ]  # 获取当前batch的url地址
        
        if(dataset_name == 'MathVista'): 
            image_inputs = [Image.open(img_path) for img_path in batch_urls]
        else:
            image_inputs = batch_urls
        
        try:
            #处理图片
            image_tensor = process_images(image_inputs, image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            question = DEFAULT_IMAGE_TOKEN + data[i]["format_question"]
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image_inputs[0].size]
                
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            
            
            batch_output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            print("==============output================:", batch_output_text[:10])  # 打印前10个输出结果

        except Exception as e:
            print(e,'error:', data[i]['q_id'])
            batch_output_text = ['<answer>error</answer>'] * BSZ



        for j, answer in enumerate(batch_output_text):# sample是原始数据集
            model_output = answer.split('assistant\n\n')[1]
            sample = data[j + i]
            result = {}
    
            if(dataset_name == 'MathVista'):
                final_ans = Extractor.extract_mathvista_answer(sample, model_output)
            else:
                final_ans = Extractor.extract_answer_special(model_output)
                              
            if final_ans == "":
                final_ans = model_output
            else:   
                if(dataset_name=='MMBench'):
                    final_ans = final_ans[0]
            
            result['question_id'] = sample['q_id']
            result['format_question'] = sample['format_question']
            result["output"] = model_output # 记录输出
            result["prediction"] = final_ans # 记录预测
            result["solution"] = sample["solution"]
            
            q_type = sample.get("problem_type", "")
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