import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
from datasets import load_dataset, load_from_disk
 
def get_question_template(COT,MODEL_NAME):
    if COT == 'TA':
        if MODEL_NAME=='R1_Onevision_7B':
            QUESTION_TEMPLATE = (
                "{Question}\n"
                "Please think about this question as if you were a human pondering deeply. "
                "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
                "It's encouraged to include self-reflection or verification in the reasoning process. "
                "Provide your detailed reasoning between the <think> </think> tags."
                "Please ensure your final answer is provided in the exact format: Answer:[your_answer]"
            )
        else:
            QUESTION_TEMPLATE = (
                "{Question}\n"
                "Please think about this question as if you were a human pondering deeply. "
                "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
                "It's encouraged to include self-reflection or verification in the reasoning process. "
                "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
            )
    
    if COT == 'PCDA':
        QUESTION_TEMPLATE= (
            "{Question}\n"
            "Please carefully analyze the pictures (or videos) and problems according to the following requirements"
            "In <prethink> </prethink> tags, carefully analyze the problem and briefly explain the steps to explain the problem and the key thinking direction of reasoning the problem"
            "In <caption> </caption> tags, Please describe the image carefully, paying special attention to the details related to the problem and the reasoning direction of solving the problem"
            "In <think> </think> tags, outline a step-by-step thought process you would use to solve the problem based on the image"
            "In <answer> </answer> tags, give the final answer in a direct format, and it must match the correct answer exactly."
            "Please sort out the output in the format of '<prethink>...</prethink>\n<caption>...</caption>\n<think>...</think>\n<answer>...</answer>' according to the above requirements"
        )
    if COT == 'ITA':
        QUESTION_TEMPLATE = (
            "{Question}\n"
            "You are tasked with analyzing an image to generate an exhaustive and detailed description. "
            "Your goal is to extract and describe all possible information from the image, including but not limited to objects, numbers, text, and the relationships between these elements. "
            "The description should be as fine and detailed as possible, capturing every nuance. After generating the detailed description, you need to analyze it and provide step-by-step detailed reasoning for the given question based on the information. "
            "Finally, provide a single word or phrase answer to the question. The description, reasoning process and answer are enclosed within <info> </info>, <think> </think> and <answer> </answer> tags, respectively, i.e., <info> image description here </info> <think> reasoning process here </think> <answer> answer here </answer>."
        )
    return QUESTION_TEMPLATE
    


def get_answer_template(MODEL_NAME):
    # 回答问题模板
    if MODEL_NAME=='R1_Onevision_7B':
        TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) in the format 'Answer:[your_answer]'. Example: 'Answer:A'",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) in the format 'Answer:[your_answer]'. Example: 'Answer:42'.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer in the format 'Answer:[your_answer]'. Example: 'Answer:Apple'.",
            "free-form": " Please provide your text answer in the format 'Answer:[your_answer]'. Example: 'Answer:Yes,it is....'.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) in the format 'Answer:[your_answer]'. Example: 'Answer:42'"
        }
        
    else:
        TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
        }
    return TYPE_TEMPLATE
 
    


def get_data_with_templete(dataset_name,x,QUESTION_TEMPLATE,TYPE_TEMPLATE):
    
    if dataset_name == 'ClevrMath':
        image_or_video = x['images'][0]
    elif dataset_name == 'MathVista':
        image_or_video = x['decoded_image']
    elif dataset_name == 'MMBench' or dataset_name == 'ChartQA':
        image_or_video = x['image']
    else:
        image_or_video = '/home/gwj/omni-video-r1/data/eval_data' + x['path'][1:]
    msg = [{
        "role": "user",
        "content": [
            {
                "type": x['data_type'],
                x['data_type']: image_or_video
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=x['format_question']) + TYPE_TEMPLATE[x['problem_type']]
            }
        ]
    }]
    return msg


class Conversation:
    def get_conversation_fuchtion(dataset_name):
        if dataset_name == 'MathVista':
            return Conversation.make_conversation_MathVista
        elif dataset_name == 'ClevrMath':
            return Conversation.make_conversation_ClevrMath
        elif dataset_name == 'ChartQA':
            return Conversation.make_conversation_ChartQA
        elif dataset_name == 'AOKVQA':
            return Conversation.make_conversation_AOKVQA
        elif dataset_name == 'MMBench':
            return Conversation.make_conversation_MMBench
        elif dataset_name == 'MMStar':
            return Conversation.make_conversation_MMStar
        elif dataset_name == 'MMathCoT':
            return Conversation.make_conversation_MMathCoT
        else:
            return Conversation.make_conversation_image_and_video
    
    def make_conversation_MathVista(example):
        question = example['question']
        answer = example['answer']
        ans_type = example['answer_type']
        image = '/home/gwj/omni-video-r1/data/eval_data/MathVista/' + example['image']
        
        if ans_type == 'text':
            problem_type = 'multiple choice'
        else:
            problem_type = 'numerical'
        
        if problem_type == 'multiple choice':
            question += "Options:\n"
            for i,op in enumerate(example['choices']):
                ans = chr(ord('A') + i)
                question += ans + ". " + op + "\n"
                
                if op == answer:
                    answer = ans
        
        msg ={
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": problem_type,
            "data_type": "image",
            "format_question": question,
            "q_id": example['pid'],
            }
        
        return msg

    def make_conversation_ClevrMath(example):
        question = example['problem']
        answer = example['answer']
        
        msg ={
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'numerical',
            "data_type": "image",
            "format_question": question,
            "q_id": question,
            }
        
        return msg

    def make_conversation_ChartQA(example):
        question = example['query']
        answer = example['label'][0]
        
        msg ={
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'numerical',
            "data_type": "image",
            "format_question": question,
            "q_id": question
            }
        
        return msg

    def make_conversation_AOKVQA(example):
        question = example['question'] + "Options:\n"

        for c,op in zip(['A','B','C','D'],example["choices"]):
            question += c + ". " + op + "\n"
        
        answer = chr(ord('A') + int(example['correct_choice_idx']))

        msg ={
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": example['question_id']
            }
        
        return msg

    def make_conversation_MMBench(example):
        question = example['question'] + "Options:\n"

        for c in ['A','B','C','D']:
            op = example[c]
            question += c + ". " + op + "\n"
        
        answer = example['answer']

        msg ={
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": example['index']
            }
        
        return msg

    def make_conversation_MMStar(example):
        question = example['question']
        
        answer = example['answer']

        msg ={
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": example['index']
            }
        
        return msg

    def make_conversation_MMathCoT(example):
        question = example['problem']
        msg ={
            "format_question":question,
            "q_id": example['problem_id'],
            "image": '/home/gwj/omni-video-r1/data/eval_data/train_data/MMathCoT' + example['path'][1:]
            }
        
        return msg

    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']
        
        msg ={
            "format_question":question,
            "q_id": example['problem_id'],
            "image": '/home/gwj/omni-video-r1/data/eval_data' + example['path'][1:]
            }
        
        return msg

class Extractor:
    # 下面是一些后面处理数据会用到的工具函数
    def extract_think(output_str): # 提取<think>中间的内容
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text): # 提取<answer>中间的内容
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_answer_R1_OneVision(text):
        """
        提取 Answer: 或 **Answer:** 后的内容，直到遇到换行符 \n 或 HTML 标签（如 </think>）。
        返回提取的字符串（自动去除首尾空格），若无匹配则返回空字符串。
        """
        pattern = r'(?i)(?:\*{0,2}Answer\*{0,2}:\s*)(.*?)(?=\s*(?:\n|</think>|$))'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def extract_answer_special(text):
        """
        提取以下两种格式的答案：
        1.提取<answer>中间的内容
        2. Answer: xxx 或 **Answer:** xxx（原逻辑）
        3. <answer> Final Answer:xxx </answer>（改进后支持文本和数字）
        返回提取的内容（自动去除首尾空格），若无匹配则返回空字符串。
        3.提取<answer>中间的内容
        """
        # 情况1：匹配 Answer: 或 **Answer:**
        pattern1 = r'(?i)(?:\*{0,2}Answer\*{0,2}:\s*)(.*?)(?=\s*(?:\n|</think>|$))'
        # 情况2：匹配 <answer> Final Answer:xxx </answer>（支持任意字符，非贪婪匹配）
        pattern2 = r'<answer>\s*Final Answer:\s*(.*?)\s*</answer>'
        pattern3 = r'<answer>\s*(.*?)\s*</answer>'
        # 优先尝试匹配情况2
        match = re.search(pattern2, text, re.IGNORECASE)  # 忽略大小写
        if match:
            return match.group(1).strip()
        
        # 如果没有匹配情况2，再尝试情况1
        match = re.search(pattern1, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        match = re.search(pattern3, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_mathvista_answer(dataset,text): # 提取<answer>中间的内容
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            precision = dataset['precision']
            answer_type = dataset['answer_type']
            question_type = dataset['question_type']
            
            if question_type == 'multi_choice':
                return answer
            elif answer_type == 'integer':
                try:
                    answer = str(int(float(answer)))
                except Exception:
                    answer = str(answer)
                return answer
            elif answer_type == 'float':
                try:
                    answer = str(round(float(answer), int(precision)))
                except Exception:
                    answer = str(answer)
                return answer
                
            return answer
        return ""

    def normalize_number(num_str): # 将字符串格式的数字转换为float
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):#平均相对准确度
        # 转换为tensor格式
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()


    def reward_fn(sample, final_ans, question_type):# 记录当前数据模型推理结果的准确奖励得分 用来计算准确率
        try:
            
            gt_ans = Extractor.extract_answer(sample.get("solution", ""))
            if final_ans.strip() == gt_ans.strip():
                return 1.0
            else:
                if question_type == "multiple choice":
                    return 1.0 if final_ans.strip() == gt_ans.strip() else 0.0
                elif question_type == "numerical":
                    gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                    out_has_decimal = ("." in final_ans) or ("," in final_ans)
                    if gt_has_decimal != out_has_decimal:
                        return 0.0
                    gt_number = Extractor.normalize_number(gt_ans)
                    out_number = Extractor.normalize_number(final_ans)
                    if gt_number is None or out_number is None:
                        return 0.0
                    return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
                elif question_type == "regression":
                    gt_number = Extractor.normalize_number(gt_ans)
                    out_number = Extractor.normalize_number(final_ans)
                    if gt_number is None or out_number is None:
                        return 0.0
                    mra = Extractor.mean_relative_accuracy(out_number, gt_number)
                    return mra
    
                else:
                    return 0.0
        except Exception as e:
            return 0.0

    