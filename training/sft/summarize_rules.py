from inference.interface.qwen2vl_interface import Qwen2VLInterface
import logging
from pathlib import Path
import json
from PIL import Image
import random
import re
import time
import torch
import gc
import argparse


GET_REASONING_PROMPT = """
Analyze the following task step by step to derive the best possible answer.

Task: {task}
Answer: {answer}

Please provide a detailed reasoning process, verify its accuracy, and then give your final answer clearly.
"""

SUMMARIZE_RULES_PROMPT = """
You are given the following reasoning examples. Analyze these examples to identify the underlying, generalizable problem-solving principles.

Examples:
{examples}

Present your findings as bullet points in this format:
- Step 1: [core principle]
- Step 2: [core principle]
...
Ensure these rules can be applied broadly to similar questions.
"""

def main(config):

    vqas_dir = Path(config.vqas_dir)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # List all json
    vqa_files = list(vqas_dir.glob('*.json'))
    vqa_files.sort()

    
    print(f"Generating examples...")
    model = "Qwen/QVQ-72B-Preview"
    qvq_inference_interface = Qwen2VLInterface(model_name=model, not25=False if "2.5" in model else True)
    inference_args = {}
    sampling_k = 4
    all_summarize_examples = []
    for vqa_file_idx, vqa_file in enumerate(vqa_files):
        with open(vqa_file, 'r') as file:
            vqas = json.load(file)
        
        # Randomly sample K VQA entries from the current file
        summarize_vqas = random.choices(vqas,k=sampling_k) 
        summarize_examples = []

        for vqa_idx, vqa in enumerate(summarize_vqas):
            image_path = Path(vqa['image_path'])
            image = Image.open(str(image_path))
            prompt = vqa['prompt']

            # Parse prompt format with options
            if "Options" in prompt:
                task_description_match = re.search(r'Task Description:\s*(.*?)\nQuestion:', prompt, re.DOTALL).group(1).strip()
                question_match = re.search(r'Question:\s*(.*?)\nOptions:', prompt, re.DOTALL).group(1).strip()
                options_match = re.search(r'Options:\s*(.*?)\nProvide a concise solution with key reasoning', prompt, re.DOTALL).group(1).strip()
                options_match_list = [line.strip().split('- ')[1] for line in options_match.splitlines() if line.strip()]

                formatted_prompt = (
                    f"{task_description_match} Answer the question: {question_match} "
                    f"by selecting from the options: {', '.join(options_match_list)}."
                )
                gen_reasoning_prompt = GET_REASONING_PROMPT.format(task=formatted_prompt, answer=vqa['answer'])
            else:
                
                task_description_match = re.search(r'Task Description:\s*(.*?)\nQuestion:', prompt, re.DOTALL).group(1).strip()
                question_match = re.search(r'Question:\s*(.*?)\nProvide a concise solution with key reasoning', prompt, re.DOTALL).group(1).strip()
                formatted_prompt = (
                    f"{task_description_match} Answer the question: {question_match}"
                )
                gen_reasoning_prompt = GET_REASONING_PROMPT.format(task=formatted_prompt, answer=vqa['answer'])
            print(f"Processing | File: {vqa_file} ({vqa_file_idx + 1}/{len(vqa_files)}) | VQAs: {image_path.name} ({vqa_idx + 1}/{len(summarize_vqas)}) ...")
            
            # Run model inference with image and prompt
            try:
                vlm_output = qvq_inference_interface.inference(image, gen_reasoning_prompt, **inference_args)
                print(f"Output: \n{vlm_output}\n")
                summarize_examples.append(dict(question=gen_reasoning_prompt ,image_path=image_path.name, output=vlm_output))

            except Exception as e:
                print(f"Failed to process vqa {vqa_idx}: {e}")
        
        all_summarize_examples.append(summarize_examples)
    print(f"Finish.")
    
    # Clean up model and memory
    if hasattr(qvq_inference_interface.model, "clear_kv_cache"):
        print(f"Clear kv cache.")
        qvq_inference_interface.model.clear_kv_cache()
    qvq_inference_interface.model.cpu()
    del qvq_inference_interface.model
    del qvq_inference_interface.processor
    del qvq_inference_interface
    torch.cuda.empty_cache()  
    gc.collect()


    
    # Summarize general reasoning rules
    model = "Qwen/Qwen2.5-VL-72B-Instruct"
    qwen_inference_interface = Qwen2VLInterface(model_name=model, not25=False if "2.5" in model else True)
    inference_args = {}
    print(f"Sumarizing rules...")
    for vqa_file_idx, vqa_file in enumerate(vqa_files):
        summarize_examples = all_summarize_examples[vqa_file_idx]
        vlm_outpus = summarize_examples[:]

        # Compose the summarization prompt from selected examples
        summarize_prompt = SUMMARIZE_RULES_PROMPT.format(examples="\n\n" + "-"*40 + "\n\n".join([f"Example {i+1}:\n"
                                                                                                f"Question:\n{example['question'].strip()}\n\n"
                                                                                                f"Answer:\n{example['output'].strip()}\n"
                                                                                            for i, example in enumerate(summarize_examples)]))
        print(f"summarize prompt length:{len(summarize_prompt)}")

        # Run model inference to generate reasoning rules
        try:
            summarize_rule = qwen_inference_interface.inference(None, summarize_prompt, **inference_args)
            print(f"Rules: \n{summarize_rule}\n")
            vlm_outpus.append(dict(question=summarize_prompt ,image_path="", rule=summarize_rule))
        except Exception as e:
            print(f"Failed to process vqa {vqa_idx}: {e}")
        
        # Save rules to a JSON file
        save_json_file = save_dir /   f"{Path(vqa_file).stem}_rules.json"
        with open(str(save_json_file), 'w') as json_file:
            json.dump(vlm_outpus, json_file, indent=4)
        print(f"VLM output saved to {str(save_json_file)}.")

    del qwen_inference_interface


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning and summarize rules from VQA data.")
    parser.add_argument('--vqas_dir', type=str, default="", help="Directory containing VQA JSON files.")
    parser.add_argument('--save_dir', type=str, default="", help="Directory to save summarized rules.")
    args = parser.parse_args()

    main(args)