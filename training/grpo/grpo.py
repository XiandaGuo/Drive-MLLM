# Modified from: https://github.com/om-ai-lab/VLM-R1/blob/main/src/open-r1-multimodal/src/open_r1/grpo.py

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import copy
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

import torch
from trl.models import unwrap_model_for_generation
# from Levenshtein import ratio
from transformers.utils import logging
logger = logging.get_logger(__name__)


# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "loc", "accuracy", "needmodel_think2output"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    # {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        # FIXME
        # This is only for Grounding task
        # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            # {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                            {"type": "text", "text": example["problem"]},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
        else:
            image = None
        

        return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }


import string
_ARTICLE_RE = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)
def normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLE_RE.sub('', text)    
    text = ' '.join(text.split())          
    return text
def check_correct(answer, gt):
    return normalize(answer) == normalize(gt)

def accuracy_reward(content, sol, **kwargs):
    reward = 0.0
    try:
        # Extract answer from solution 
        answer_pattern = r'<answer>(.*?)</answer>'
        sol_answer_match = re.search(answer_pattern, sol)
        sol_answer = sol_answer_match.group(1).strip()
        
        # Extract answer from content 
        content_answer_matches = re.findall(answer_pattern, content, re.DOTALL)
        student_answer = content_answer_matches[-1].strip()

        reward = 1.0 if check_correct(student_answer, sol_answer) else 0.0

    except Exception:
        pass  # Keep reward as 0.0 if all methods fail

    return reward

def acc_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = accuracy_reward(content, sol)  
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")  

        
    return rewards


def compute_iou(box1, box2):

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

def compute_centerness(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2


    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    if w2 <= 0 or h2 <= 0:
        return 0.0  

    dx = abs(cx1 - cx2) / (w2 / 2)
    dy = abs(cy1 - cy2) / (h2 / 2)

    dx = min(dx, 1.0)
    dy = min(dy, 1.0)

    centerness = ((1 - dx) * (1 - dy)) ** 0.5

    return centerness


def location_reward(content, sol, is_iou, **kwargs):
    reward = 0.0
    
    try:
        answer_pattern = r'<location>(.*?)</location>'
        data_pattern = r'([^:]+):\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'

        # Extract answer from solution 
        sol_answer_match = re.search(answer_pattern, sol)
        sol_answer = sol_answer_match.group(1).strip()
        sol_box_data_match = re.findall(data_pattern, sol_answer)
        sol_boxes = {}
        for obj, x1, y1, x2, y2 in sol_box_data_match:
            sol_boxes[normalize(obj.lstrip(', '))] = list(map(int, (x1, y1, x2, y2)))
 
        # Extract answer from content 
        content_answer_matches = re.findall(answer_pattern, content, re.DOTALL)
        student_answer = content_answer_matches[-1].strip()
        stu_box_data_match = re.findall(data_pattern, student_answer)
        stu_boxes = {}
        for obj, x1, y1, x2, y2 in stu_box_data_match:
            stu_boxes[normalize(obj.lstrip(', '))] = list(map(int, (x1, y1, x2, y2)))


        # calc score
        scores = []
        for name, gt_box in sol_boxes.items():
            if name in stu_boxes:
                pb = stu_boxes[name]
                if is_iou:
                    iou = compute_iou(gt_box, pb)  
                    if iou >0.5:
                        score = 1.0
                    else:
                        score = 0.0
                else:
                    centerness = 2 * compute_centerness(gt_box, pb) - 1 
                    if centerness >0.5:
                        score = 1.0
                    else:
                        score = 0.0
            else:
                score = 0.0
            scores.append(score)
            
        reward = float(sum(scores) / len(scores))


    except Exception:
        pass  # Keep reward as 0.0 if all methods fail

    return reward

def loc_reward(completions, solution, is_iou, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = location_reward(content, sol, is_iou)  
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Loc reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")  

        
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<location>.*?</location>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    flags = re.DOTALL | re.VERBOSE
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, flags) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


import copy, math, re
from typing import List
import torch
import torch.distributed as dist
from accelerate import Accelerator
from trl.data_utils import maybe_apply_chat_template
def build_chat_example(think: str) -> dict:
    tmpl = ("You are given a reasoning process, and your task is to infer the final answer based only on it.\n\n"
            "<reasoning>\n"
            "{think_content}\n"
            "</reasoning>\n\n"
            "Please extract the final answer based on this reasoning.\n\n"
            "Output format:\n"
            "<answer>[Your concise answer]</answer>")
    user_text = tmpl.format(think_content=think)
    chat_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text}
            ],
        }
    ]
    return {
        "prompt": chat_prompt, 
    }
def needmodel_think_to_output_reward(
    completions: List[list[dict]],
    solution:    List[str],
    model,
    processor,
    accelerator: Accelerator,
    generation_config,
    **kwargs,
):
    device = accelerator.device


    # Extract <think> and <answer> segments from input
    think_texts, gt_answers, idx_map = [], [], []
    for idx, (comp, sol) in enumerate(zip(completions, solution)):
        content = comp[0]["content"] 
        m_think = re.search(r"<think>(.*?)</think>",  content, re.S)
        m_gt    = re.search(r"<answer>(.*?)</answer>", sol,     re.S)
        if m_think and m_gt:
            think_texts.append(m_think.group(1).strip())
            gt_answers.append(m_gt.group(1).strip())
            idx_map.append(idx)

    # Count valid samples in this rank
    local_valid = torch.tensor([len(think_texts)], device=device)
    world_size  = accelerator.num_processes

    # Perform all_gather to find max batch size across all ranks
    if world_size > 1:
        sizes = [torch.zeros_like(local_valid) for _ in range(world_size)]
        dist.all_gather(sizes, local_valid)
        max_valid = max([t.item() for t in sizes])
    else:
        max_valid = local_valid.item()

    # Pad with placeholders to match max_valid size
    PAD_THINK = "<pad_think>"
    PAD_ANS   = "<pad_ans>"
    while len(think_texts) < max_valid:
        think_texts.append(PAD_THINK)
        gt_answers.append(PAD_ANS)    

    # Construct prompt and run batch generation
    examples = [build_chat_example(t) for t in think_texts]
    batch_prompts = [maybe_apply_chat_template(ex, processor)["prompt"] for ex in examples]
    inputs = processor(
        text=batch_prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ).to(device)

    # Copy and update generation config
    local_cfg = copy.deepcopy(generation_config)
    local_cfg.max_new_tokens = 100
    local_cfg.do_sample      = False   
    
    # Set model to eval mode for inference
    was_training = model.training
    model.eval() 
    with torch.no_grad():
        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            gen_ids = unwrapped_model.generate(
                **inputs, 
                generation_config=local_cfg
            )
    if was_training:          
        model.train()

    # Decode generated output
    prompt_len  = inputs["input_ids"].size(1)
    pred_texts  = processor.batch_decode(
        gen_ids[:, prompt_len:], skip_special_tokens=True
    )


    # Compute reward only for valid (non-padded) samples
    rewards = [0.0] * len(completions)
    for pred, gt, orig_i in zip(pred_texts, gt_answers, idx_map):
        if gt == PAD_ANS:
            continue
        m = re.search(r"<answer>(.*?)</answer>", pred, re.S)
        if m:
            reward = 1.0 if check_correct(m.group(1), gt) else 0.0
            rewards[orig_i] = reward

    # Synchronize all processes
    accelerator.wait_for_everyone()


    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, reward in zip(contents, solution, rewards):
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Logic reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n") 


    return rewards

reward_funcs_registry = {
    "accuracy": acc_reward,
    "loc": loc_reward,
    "format": format_reward,
    "needmodel_think2output": needmodel_think_to_output_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
