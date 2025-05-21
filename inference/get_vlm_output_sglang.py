import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, Any
from tqdm import tqdm
from PIL import Image

from sglang.utils import wait_for_server, terminate_process
from sglang.utils import launch_server_cmd
from openai import AsyncOpenAI
import asyncio

from inference.utils import pil2base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    # Set a random seed for reproducibility
    random.seed(42)

    # Extract configuration parameters
    save_dir = Path(config.save_dir)
    save_sub_dir = Path(config.save_sub_dir)
    vqas_dir = Path(config.vqas_dir)
    sglang_model = config.sglang_model
    sglang_tpl = config.sglang_tpl
    sglang_dtype = config.sglang_dtype
    sglang_mem = config.sglang_mem
    sglang_maxreq = config.sglang_maxreq
    sglang_dp = config.sglang_dp
    sglang_tp = config.sglang_tp
    bs_per_req = config.bs_per_req

    start_time = time.time()
    
    # Launch sglang
    vision_process, port = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {sglang_model} \
        --chat-template={sglang_tpl} --dtype={sglang_dtype} \
        --mem-fraction-static={sglang_mem} --max-running-requests={sglang_maxreq} \
        --dp={sglang_dp} --tp={sglang_tp}
    """
    )
    wait_for_server(f"http://localhost:{port}")
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    # def async function
    async def launch_sglang():
        vqa_files = list(vqas_dir.glob('*.json'))
        vqa_files.sort()
        for vqa_file_idx, vqa_file in enumerate(vqa_files):
            logger.info(f"Processing | File: {vqa_file} ({vqa_file_idx + 1}/{len(vqa_files)})...")
            with open(vqa_file, 'r') as file:
                vqas = json.load(file)

            task_datas = [] 
            for vqa_idx in tqdm(range(len(vqas)), desc='Get datas:'):
                vqa = vqas[vqa_idx]
                image_path = Path(vqa['image_path'])
                image = Image.open(str(image_path))
                image_url = f"data:image/jpeg;base64,{pil2base64(image, 'JPEG')}" 
                prompt = vqa['prompt']

                task_datas.append(dict(
                    vqa_idx=vqa_idx, 
                    image=image_path.name, 
                    image_url=image_url,
                    prompt=prompt))
            
            # fill tasks
            prompts = [d['prompt'] for d in task_datas]
            image_urls = [d['image_url'] for d in task_datas]
            tasks = []
            for idx in tqdm(range(len(prompts)), desc='Get tasks:'):
                prompt = prompts[idx]
                image_url = image_urls[idx]
                tasks.append(client.chat.completions.create(
                    model=sglang_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ]
                    }],
                    timeout=len(prompts)*2
                ))

                
            # Process tasks in batch
            logger.info(f"Handling tasks...")
            start_infer = time.time()
            responses = []
            batch_size = bs_per_req 
            for i in range(0, len(tasks), batch_size):
                chunk = tasks[i:i+batch_size]
                responses += await asyncio.gather(*chunk)
                print(f"Processed {min(i+batch_size, len(tasks))}/{len(tasks)} tasks")
            print(f"Total cost time: {time.time()-start_infer:.3f}s")

            # Process response
            logger.info(f"Processing response datas...")
            vlm_outpus = []
            for idx in tqdm(range(len(responses))):
                response = responses[idx]
                vlm_output = response.choices[0].message.content
                task_data = task_datas[idx]
                task_data['output'] = vlm_output
                task_data.pop("image_url") 
                vlm_outpus.append(task_data)
            
            # Save outputs to a JSON file for each prompt file
            save_json_dir = save_dir / save_sub_dir / Path(sglang_model).name 
            save_json_dir.mkdir(exist_ok=True, parents=True)
            save_json_file = save_json_dir /   f"{Path(vqa_file).stem}_output.json"
            with open(str(save_json_file), 'w') as json_file:
                json.dump(vlm_outpus, json_file, indent=4)
            logger.info(f"VLM output saved to {str(save_json_file)}.")


    # Run async main
    asyncio.run(launch_sglang())
    logger.info(f"Total time: {(time.time()- start_time):.3f}s.")
    
    # End sglang
    terminate_process(vision_process)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate outputs using the SGLang Vision-Language Model (VLM).")
    
    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory to save the generated output files.')
    parser.add_argument('--save_sub_dir', type=str, default='',
                        help='Subdirectory inside save_dir to organize outputs.')
    parser.add_argument('--vqas_dir', type=str, default='',
                        help='Path to the folder containing input VQA files.')
    
    parser.add_argument('--sglang_model', type=str, default='',
                        help='Path or identifier for the SGLang model to be loaded.')
    parser.add_argument('--sglang_tpl', type=str, default='',
                        help='Chat template name used by the SGLang model (e.g., "qwen2-vl").')
    parser.add_argument('--sglang_dtype', type=str, default='',
                        help='Precision type for inference (e.g., "bfloat16").')
    parser.add_argument('--sglang_mem', type=str, default='',
                        help='Static memory fraction reserved for the model (e.g., "0.7").')
    parser.add_argument('--sglang_maxreq', type=str, default='',
                        help='Maximum number of concurrent running requests.')
    parser.add_argument('--sglang_dp', type=str, default='',
                        help='Data parallelism degree.')
    parser.add_argument('--sglang_tp', type=str, default='',
                        help='Tensor parallelism degree.')
    
    parser.add_argument('--bs_per_req', type=int, help='Batch size for each individual request.')
    
    args = parser.parse_args()
    main(args)

