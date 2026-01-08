import math
import os
import argparse
import json
import sys
sys.path.insert(0,"/leonardo_work/EUHPC_D26_093/jli/Video-LLaVA")
import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
# from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# from videollava.train.train import smart_tokenizer_and_embedding_resize
from utils.station import STATION

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # print(video_tensor.shape)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    model = model.to(args.device)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))

    answers_file = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in tqdm(video_formats):  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
                sample_set['pred'] = output
                ans_file.write(json.dumps(sample_set) + "\n")
                print(f"-------- response: {output} --------")
                break
        if index == args.len: break

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/leonardo_scratch/fast/EUHPC_D26_093/ckpt_backup/Video_LLaVA_7B")
    parser.add_argument('--dataset', default="msrvtt")
    parser.add_argument('--output_dir', default="./predicts_prune")
    parser.add_argument('--modify', default="mi_max")
    parser.add_argument("--v_ratio", type=int, default=194)
    parser.add_argument("--len", type=int, default=1000)
    parser.add_argument("--T", type=float, default=0.01)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model_max_length", type=int, default=2048)
    args = parser.parse_args()
    args.conv_mode = "llava_v1"

    STATION["v_ratio"] = args.v_ratio
    STATION["modify"] = args.modify
    STATION["temperature"] = args.T


    args.output_dir = os.path.join(args.output_dir, args.dataset)
    args.output_name = f"{args.modify}{args.T}_v{args.v_ratio}_predictions.json"

    root = {"tgif": "/leonardo_scratch/fast/EUHPC_D26_093/dataset/video/TGIF_Zero_Shot_QA",
            "msvd": "/leonardo_scratch/fast/EUHPC_D26_093/dataset/video/MSVD_Zero_Shot_QA",
            "msrvtt": "/leonardo_scratch/fast/EUHPC_D26_093/dataset/video/MSRVTT_Zero_Shot_QA",
            }
    if args.dataset == "tgif":
        args.video_dir = os.path.join(root[args.dataset], "mp4")
    elif args.dataset == "msvd":
        args.video_dir = os.path.join(root[args.dataset], "videos")
    elif args.dataset == "msrvtt":
        args.video_dir = os.path.join(root[args.dataset], "videos/all")
    args.gt_file_question = os.path.join(root[args.dataset], "test_q.json")
    args.gt_file_answers = os.path.join(root[args.dataset], "test_a.json")

    run_inference(args)
