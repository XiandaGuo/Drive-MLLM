import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, Any
from PIL import Image
import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_random_output(prompt_index, image, prompt):
    if prompt_index != 1:
        options_match = re.search(r'Options:.*?\n(.*?)(?=\n\n|\Z)', prompt, re.DOTALL).group(1).strip()
        options = [line.strip("- ").strip() for line in options_match.split("\n") if line.strip()]
        random_option = random.choice(options)
        return f"<answer>{random_option}</answer>"
    else:
        image_width, image_height = image.size
        random_x = random.randint(0, image_width - 1)
        random_y = random.randint(0, image_height - 1)
        return f"<answer>{str([random_x, random_y])}</answer>"



def main(config):

    # Set a random seed for reproducibility
    random.seed(42)

    # Extract configuration parameters
    model_type = "random"
    model = "random"
    save_dir = Path(config.save_dir)
    vqas_dir = Path(config.vqas_dir)
    
    vqa_files = list(vqas_dir.glob('*.json'))
    vqa_files.sort()
    for vqa_file_idx, vqa_file in enumerate(vqa_files):
        with open(vqa_file, 'r') as file:
            vqas = json.load(file)

        vlm_outpus = []
        for vqa_idx, vqa in enumerate(vqas):

            image_path = Path(vqa['image_path'])
            image = Image.open(str(image_path))
            prompt = vqa['prompt']

            # Run inference
            logger.info(f"Processing | File: {vqa_file} ({vqa_file_idx + 1}/{len(vqa_files)}) | VQAs: {image_path.name} ({vqa_idx + 1}/{len(vqas)}) ...")
            try:
                vlm_output = generate_random_output(vqa_file_idx, image, prompt)
                logger.info(f"Output: {vlm_output}")
                vlm_outpus.append(dict(vqa_idx=vqa_idx, image=image_path.name, prompt=prompt, output=vlm_output))
                
            except Exception as e:
                logger.error(f"Failed to process vqa {vqa_idx}: {e}")
                vlm_outpus.append(dict(vqa_idx=vqa_idx, image=image_path.name, output=''))   

        # Save outputs to a JSON file for each prompt file
        save_json_dir = save_dir / model_type / Path(model).name 
        save_json_dir.mkdir(exist_ok=True, parents=True)
        save_json_file = save_json_dir /   f"{Path(vqa_file).stem}_output.json"
        with open(str(save_json_file), 'w') as json_file:
            json.dump(vlm_outpus, json_file, indent=4)
        logger.info(f"VLM output saved to {str(save_json_file)}.")
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate outputs using the MLLM model.")
    parser.add_argument('--save_dir', type=str, default='',
                        help='Define the directory where the generated output files will be saved.')
    parser.add_argument('--vqas_dir', type=str, default='',
                        help='Specify the folder for the VQAs.')
    args = parser.parse_args()
    main(args)

