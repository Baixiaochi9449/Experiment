# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import copy
import torch
import os
import json
from tqdm import tqdm
import torch
from datasets import Dataset
import argparse
from datasets import load_dataset
from llama_tools import get_question_template,get_answer_template
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from llama_tools import get_question_template,get_answer_template,get_data_with_templete,Extractor,Conversation

# def construct_conv(text_data_batch):
#     batch_prompt_question = []
#     for idx in range(len(text_data_batch["format_question"])):  # 遍历字典的列表值
#         question = (
#             "<image>" 
#             + QUESTION_TEMPLATE.format(Question=text_data_batch["format_question"][idx]) 
#             + TYPE_TEMPLATE[text_data_batch["problem_type"][idx]]
#         )
#         print("QUESTION:",question)
        
#         conv_template = "qwen_1_5"  
#         conv = copy.deepcopy(conv_templates[conv_template])
#         conv.append_message(conv.roles[0], question)
#         conv.append_message(conv.roles[1], None)
#         prompt_question = conv.get_prompt()
#         batch_prompt_question.append(prompt_question)
        
#     return batch_prompt_question

def construct_conv(text_data_batch):
    
    batch_prompt_question=[]
    for x in tqdm(text_data_batch):
        question = "<image>" + QUESTION_TEMPLATE.format(Question=x['format_question']) + TYPE_TEMPLATE[x['problem_type']]

        print("QUESTION:",question)
        
        conv_template = "qwen_1_5"  
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        batch_prompt_question.append(prompt_question)
            
    return batch_prompt_question

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
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(MODEL_PATH, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

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

    print("Sample data item:", data[0])  # 检查单条数据的结构
    print("Type of data:", type(data))  # 是否是 Dataset 或 list？
    print("Slicing test:", type(data[0:2]))  # 切片返回的是列表还是字典？
    
    #原本的data不知道是什么格式的
    
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
    # for i,(text_data,url) in enumerate(zip(data,all_urls)):
        # text_data_batch=data[i:i+BSZ]
        text_data_batch = [data[j] for j in range(i, min(i+BSZ, len(data)))]  #当BSZ>1的时候能运行，但是BSZ=1就变成了一个字典了，所以需要强制转换为列表
        url_batch = all_urls[i:i+BSZ]
        # print(f"Type of text_data_batch: {type(text_data_batch)}")
        # print(f"First item type: {type(text_data_batch[0]) if len(text_data_batch) > 0 else 'empty'}")
        # print(f"Keys in first item: {text_data_batch[0].keys() if isinstance(text_data_batch[0], dict) else 'not dict'}")
        
        if(dataset_name == 'MathVista'): 
            image_inputs = [Image.open(url) for url in url_batch]   #一个列表
        else:
            image_inputs = url_batch  #已经处理过的
        
        try:
            #处理图片
            batch_image_tensor = process_images(image_inputs, image_processor, model.config)  #既然原本是列表，说明能处理一个列表的图片，这个image_tensor就是列表图片
            batch_image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in batch_image_tensor]

            batch_prompt_question =construct_conv(text_data_batch)  #一个列表，存储了输入的conv格式的数据
            
            #还是只能处理单条数据
            batch_outputs=[]
            for index in range(len(url_batch)):
                prompt_question=batch_prompt_question[index]
                image = image_inputs[index]
                image_tensor = batch_image_tensor[index]
                
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                image_sizes = [image.size]
                    
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
                )
                batch_outputs.append(outputs)
            
            batch_output_text = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            
            print("==============output================:", batch_output_text)  # 打印前10个输出结果

        except Exception as e:
            print(e,'error:', data[i]['q_id'])
            batch_output_text = ['<answer>error</answer>'] * BSZ



        for j, answer in enumerate(batch_output_text):# sample是原始数据集
            model_output = answer
            sample = data[i+j]
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
    