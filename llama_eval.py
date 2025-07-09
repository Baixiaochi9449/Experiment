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
MODEL_NAME = MODEL_NAME.replace("-", "_")  # 输出: "R1_Onevision_7B"
print("MODEL_NAME:", MODEL_NAME)

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()  # 设置模型为评估模式


if(MODEL_NAME=='Llama_3.2V_11B_cot'):
    kwargs = dict(do_sample=True, 
                 max_new_tokens=2048, 
                 temperature=0.6, 
                 top_p=0.9)
else:
    kwargs = dict(do_sample=True, 
                max_new_tokens=4096, 
                temperature=0.1, 
                top_p=0.001)
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
    print("QUESTION_TEMPLATE:", QUESTION_TEMPLATE)
    print("TYPE_TEMPLATE:", TYPE_TEMPLATE)
    #填入对话格式模板
    make_conversation_cot_image=Conversation.get_conversation_fuchtion(dataset_name) # 获取对话格式模板函数
    data = data.map(make_conversation_cot_image)   #存储了所有用到的数据
    
    messages = []
    all_urls=[]
    for x in tqdm(data): # 构造输入数据
        msg = get_data_with_templete(dataset_name,x,QUESTION_TEMPLATE,TYPE_TEMPLATE)
        messages.append(msg) #变为role content的格式
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

    #2、开始处理数据
    mean_acc = []
    mean_mra = []
    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ] 
        batch_urls= all_urls[i:i + BSZ]  # 获取当前batch的url地址
        
        #（1）处理图片输入
        # transform = Compose([
        #     Resize((336, 336)),  # 调整为模型接受的尺寸
        #     lambda x: x.convert("RGB")
        # ])        
        # if(dataset_name == 'MathVista'): 
        #     image_inputs = [transform(Image.open(img_path)) for img_path in batch_urls]
        # else:
        #     image_inputs = [transform(img_path) for img_path in batch_urls]
        if(dataset_name == 'MathVista'): 
            image_inputs = [Image.open(img_path) for img_path in batch_urls]
        else:
            image_inputs = batch_urls
        #（2）处理文本输入
        text_inputs = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in batch_messages]  
        
        try:
            inputs_batch = processor(    
                text=text_inputs,
                images=image_inputs,
                # padding=True,
                padding="longest",
                return_tensors="pt"
            ).to(model.device)
        
            outputs = model.generate(
                **inputs_batch, 
                **kwargs
                )
            
            batch_output_text = processor.batch_decode(outputs, skip_special_tokens=True)
            
            print("==============output================:", batch_output_text[:10])  # 打印前10个输出结果

        except Exception as e:
            print(e,'error:', data[i]['q_id'])
            batch_output_text = ['<answer>error</answer>'] * BSZ
            
        #（3）处理模型输出的结果
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