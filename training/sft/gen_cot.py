from inference.interface.qwen2vl_interface import Qwen2VLInterface
import logging
from pathlib import Path
import json
from PIL import Image
import random
import re
import time
from tqdm import tqdm
from inference.utils import pil2base64
import argparse

from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
if is_in_ci(): # False
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd
from openai import Client
from openai import AsyncOpenAI
import asyncio


GEN_WITH_RULES_PROMPT = """
Use the following principles to answer the question:

{rules}

Question: {question}
Answer: {answer}

Provide a concise solution with key reasoning steps in the following format:
<think>[Your step-by-step reasoning]</think>
<answer>[Final answer]</answer>
"""

VERIFY_PROMPT = """
{response}

Evaluate the structured response above for logical consistency and completeness. Specifically:

1. Does the reasoning in <think> logically support the conclusion in <answer>?
2. Are there any internal contradictions, logical errors, or missing steps in the reasoning?
3. Is the reasoning chain complete and valid?

Provide your evaluation in the following format:

<reason>[A concise justification of your assessment or a brief note confirming the reasoning's validity]</reason>
<validation>Valid / Invalid</validation>

Then, regardless of validity, output the full response in the following format:
- Keep <answer> unchanged.
- Modify <think> only if necessary to ensure logical soundness.

<think>[final version of reasoning steps]</think>
<answer>[original final answer]</answer>
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):

    vqas_dir = Path(config.vqas_dir)
    rules_dir = Path(config.rules_dir)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # List all json
    vqa_files = list(vqas_dir.glob('*.json'))
    vqa_files.sort()


    # launch sglang
    model = "Qwen/Qwen2.5-VL-72B-Instruct"
    vision_process, port = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {model} \
        --chat-template=qwen2-vl --dtype=bfloat16 --mem-fraction-static=0.7 --tp=4
    """
    )
    wait_for_server(f"http://localhost:{port}")
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    # def async function
    async def launch_sglang():
        for vqa_file_idx, vqa_file in enumerate(vqa_files):
            questions = []
            image_urls = []
            image_names = []
            vlm_outpus = []
            image_pixels = []
            descs = []
            obj_bbox = []

            # Load rules
            summarize_rule = ""
            rule_file = rules_dir / f"{Path(vqa_file).stem}_rules.json"
            with open(rule_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("rule", "") != "":
                            summarize_rule = item.get("rule")
            
            # Load VQAs
            with open(vqa_file, 'r') as file:
                vqas = json.load(file)
            
            # Process
            for vqa_idx, vqa in enumerate(vqas):
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
                    gen_reasoning_with_rules_prompt = GEN_WITH_RULES_PROMPT.format(rules=summarize_rule,question=formatted_prompt, answer=vqa['answer'])
                else:
                    pass
                    task_description_match = re.search(r'Task Description:\s*(.*?)\nQuestion:', prompt, re.DOTALL).group(1).strip()
                    question_match = re.search(r'Question:\s*(.*?)\nProvide a concise solution with key reasoning', prompt, re.DOTALL).group(1).strip()
                    formatted_prompt = (
                        f"{task_description_match} Answer the question: {question_match}"
                    )
                    gen_reasoning_with_rules_prompt = GEN_WITH_RULES_PROMPT.format(rules=summarize_rule,question=formatted_prompt, answer=vqa['answer'])
                logger.info(f"Processing | File: {vqa_file} ({vqa_file_idx + 1}/{len(vqa_files)}) | VQAs: {image_path.name} ({vqa_idx + 1}/{len(vqas)}) ...")
                

                questions.append(gen_reasoning_with_rules_prompt)
                image_urls.append(pil2base64(image))
                image_names.append(Path(vqa['image_path']).name)
                image_pixels.append(vqa['image_pixel'])
                descs.append(vqa['descs'])
                obj_bbox.append(vqa['obj_bbox'])

        
        
            # Create requests
            tasks = []
            for idx in tqdm(range(len(questions))):
                tasks.append(client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": questions[idx]},
                            {"type": "image_url", "image_url": {"url": image_urls[idx]}},
                        ]
                    }],
                    timeout=len(questions)*2
                ))

            # Execute batched inference
            start_infer = time.time()
            responses = []
            batch_size = 1000 
            for i in range(0, len(tasks), batch_size):
                chunk = tasks[i:i+batch_size]
                responses += await asyncio.gather(*chunk)
                print(f"Processed {min(i+batch_size, len(tasks))}/{len(tasks)} tasks")
            print(f"Total cost time: {time.time()-start_infer:.3f}s")

            # get generation results
            for idx, response in enumerate(responses):
                vlm_outpus.append(dict(
                    question=questions[idx],
                    image_path=image_names[idx],
                    output=response.choices[0].message.content,
                    image_pixel = image_pixels[idx],
                    descs = descs[idx],
                    obj_bbox = obj_bbox[idx],
                ))

            # Save outputs to a JSON file
            save_json_file = save_dir /   f"{Path(vqa_file).stem}_output.json"
            with open(str(save_json_file), 'w') as json_file:
                json.dump(vlm_outpus, json_file, indent=4)
            print(f"VLM output saved to {str(save_json_file)}.")

            ############################################ verify ############################################
            # Create requests
            verify_tasks = []
            verify_questions = [VERIFY_PROMPT.format(response=response.choices[0].message.content) for response in responses]
            for idx in tqdm(range(len(questions))):
                verify_tasks.append(client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": verify_questions[idx]},
                            {"type": "image_url", "image_url": {"url": image_urls[idx]}},
                        ]
                    }],
                    timeout=len(verify_questions)*2
                ))
            
            # Execute batched inference
            start_infer = time.time()
            verify_responses = []
            batch_size = 1000
            for i in range(0, len(verify_tasks), batch_size):
                chunk = verify_tasks[i:i+batch_size]
                verify_responses += await asyncio.gather(*chunk)
                print(f"Processed {min(i+batch_size, len(verify_tasks))}/{len(verify_tasks)} tasks")
            print(f"Total cost time: {time.time()-start_infer:.3f}s")

            # get generation results
            vlm_v_outpus = []
            for idx, v_response in enumerate(verify_responses):
                vlm_v_outpus.append(dict(
                    question=verify_questions[idx],
                    image_path=image_names[idx],
                    output=v_response.choices[0].message.content,
                    image_pixel = image_pixels[idx],
                    descs = descs[idx],
                    obj_bbox = obj_bbox[idx],
                ))

            # Save outputs to a JSON file
            v_save_json_file = save_dir /   f"{Path(vqa_file).stem}_output_verify.json"
            with open(str(v_save_json_file), 'w') as json_file:
                json.dump(vlm_v_outpus, json_file, indent=4)
            print(f"VLM output saved to {str(v_save_json_file)}.")


    # Run async inference and cleanup        
    asyncio.run(launch_sglang())
    terminate_process(vision_process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VQA reasoning and verification.")
    parser.add_argument('--vqas_dir', type=str, default="", help="Path to the directory containing VQA JSON files.")
    parser.add_argument('--rules_dir', type=str, default="", help="Path to the directory containing reasoning rules.")
    parser.add_argument('--save_dir', type=str, default="", help="Directory to save the output reasoning results.")
    args = parser.parse_args()
    main(args)

