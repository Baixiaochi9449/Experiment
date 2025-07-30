import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoConfig
import cv2
import base64
import openai
from PIL import Image
import numpy as np
from datasets import load_dataset,Dataset
from qwen_tools import get_question_template,get_answer_template,get_data_with_templete,Extractor,Conversation

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    parser.add_argument("--dataset_path", type=str, default="")
    return parser.parse_args()

def load_video(video_path,args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()

    return spare_frames,frame_time,video_time




def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    DATASET = args.dataset_path
    DATASET_PATH = DATASET.split('->')[0]
    dataset_name = DATASET.split('->')[1]
    COT="TA"
    MODEL_NAME = args.model_path.split('/')[-1]  # 输出: "R1-Onevision-7B"
    MODEL_NAME = MODEL_NAME.replace("-", "_")  # 输出: "R1_Onevision_7B"
    print("MODEL_NAME",MODEL_NAME)
    
    # 1、加载模型
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    else:
        pass

    # import pdb;pdb.set_trace()
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    # import pdb;pdb.set_trace()

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False

    # 2、创建文件夹
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_name = args.output_name
    OUTPUT_PATH = os.path.join(args.output_dir, f"{output_name}.json")
    

    # #3、处理视频路径
    # video_path = args.video_path  #video的文件夹，应该不可以是很多个的那种

    # all_video_pathes = []

    # # Check if the video_path is a directory or a file
    # if dataset_name in {"mmvu","Video_Hullucer"}:
    #     if os.path.isdir(video_path):
    #         # 使用 os.walk 递归遍历所有子目录
    #         for root, dirs, files in os.walk(video_path):
    #             for filename in files:
    #                 if filename.lower().endswith(('.mp4', '.avi', '.mov')):  # 检查是否是视频文件
    #                     file_path = os.path.join(root, filename)
    #                     all_video_pathes.append(file_path)        
    
    # else:
    #     if os.path.isdir(video_path):
    #         # If it's a directory, loop over all files in the directory
    #         for filename in os.listdir(video_path):
    #             if filename.lower().endswith(('.mp4', '.avi', '.mov')):
    #                 cur_video_path = os.path.join(video_path, f"{filename}")
    #                 all_video_pathes.append(os.path.join(video_path, cur_video_path))
    # import pdb;pdb.set_trace()
    #4、加载文本数据

    data = Dataset.from_json(DATASET_PATH)
    
    QUESTION_TEMPLATE=get_question_template(COT,MODEL_NAME) # 获取问题模板
    TYPE_TEMPLATE = get_answer_template(MODEL_NAME)
    print("QUESTION_TEMPLATE:",QUESTION_TEMPLATE)
    print(" TYPE_TEMPLATE", TYPE_TEMPLATE)
    make_conversation_cot_image=Conversation.get_conversation_fuchtion(dataset_name) # 获取对话格式模板函数
    data = data.map(make_conversation_cot_image)
    #开始处理数据
    mean_acc=[]
    mean_mra=[]
    final_output=[]
    for i,data_text in enumerate(data):
        # data_text = data[i]
        video_path = data_text['image']
        if( dataset_name =="Video_Hullucer"):
            video_path = video_path.replace("/home/gwj/omni-video-r1/data/eval_datahome/", "/home/")
        sample_set = {}
        question = QUESTION_TEMPLATE.format(Question=data_text['format_question']) + TYPE_TEMPLATE[data_text['problem_type']]
        sample_set["Q"] = data_text['format_question']
        sample_set["video_name"] = video_path
        print("==========Processing video:========", video_path)

        # Check if the video exists
        if os.path.exists(video_path):
            if "gpt4v" != args.model_path:
                video,frame_time,video_time = load_video(video_path, args)
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                video = [video]
            else:
                spare_frames,frame_time,video_time = load_video_base64(video_path)
                interval = int(len(video) / args.for_get_frames_num)

        # try:
        # Run inference on the video and add the output to the list
        if "gpt4v" != args.model_path:
            qs = question
            if args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                qs = f'{time_instruciton}\n{qs}'
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates["qwen_1_5"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            if tokenizer.pad_token_id is None:
                if "qwen" in tokenizer.name_or_path.lower():
                    print("Setting pad token to bos token for qwen model.")
                    tokenizer.pad_token_id = 151643
                    
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = question
        else:
            prompt = question

        system_error = ""

        if "gpt4v" != args.model_path:


            with torch.inference_mode():
                # model.update_prompt([[cur_prompt]])
                # import pdb;pdb.set_trace()
                # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
                if "mistral" not in cfg_pretrained._name_or_path.lower():
                    output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True, stopping_criteria=[stopping_criteria])
                    # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
                else:
                    output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1, use_cache=True)
                    # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)
        

        if "gpt4v" != args.model_path:
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        print(f"Question: {prompt}\n")
        print(f"Response: {outputs}\n")

        # import pdb;pdb.set_trace()
        if "mistral" not in cfg_pretrained._name_or_path.lower():
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()

        result = {}
        
        if(dataset_name == 'MathVista'):
            final_ans = Extractor.extract_mathvista_answer(data_text, outputs)
        else:
            final_ans = Extractor.extract_answer_special(outputs)
            
        if final_ans == "":
            final_ans = outputs[-1]
        else:   
            if(dataset_name in {'MMBench','tempcompass','Video_Hullucer'}):
                final_ans = final_ans[0]
                
        q_type = data_text.get("problem_type", "")
        if(dataset_name in {"mmvu","videommmu"} and q_type=="multiple choice"):
            final_ans = final_ans[0]
            
        result['question_id'] = data_text['q_id']
        result['format_question'] = data_text['format_question']
        result["output"] = outputs # 记录输出
        result["prediction"] = final_ans # 记录预测
        result["solution"] = data_text["solution"]
        
        result["reward"] = Extractor.reward_fn(data_text, final_ans, q_type) # 记录准确率奖励分数
        result['correct'] = True if result["reward"]==1.0 else False
        
        if data_text['problem_type'] != 'regression':
            mean_acc.append(result["reward"])
        else:
            mean_mra.append(result["reward"])

        final_output.append(result) # 记录最后输出(原数据+输出结果)
    
        try: # 保存结果
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False) #不是一条一条的写，而是每次都重复覆盖
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


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)