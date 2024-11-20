import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, Any

from PIL import Image
from datasets import load_dataset
from prompt.prompt_idx import PromptIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_random_output(prompt_index, image, descriptions):
    # Enumerate to get different random results
    if prompt_index == PromptIndex.XY_QA.value:
        image_width, image_height = image.size
        random_x = random.randint(0, image_width - 1)
        random_y = random.randint(0, image_height - 1)
        return f"<OUTPUT FORMAT START>[{random_x}, {random_y}]<OUTPUT FORMAT END>"
    elif prompt_index == PromptIndex.BOX_QA.value:
        image_width, image_height = image.size
        random_x_min = random.randint(0, image_width - 1 - 1)
        random_y_min = random.randint(0, image_height - 1 - 1)
        random_x_max = random.randint(random_x_min + 1, image_width - 1)
        random_y_max = random.randint(random_y_min + 1, image_height - 1)
        return f"<OUTPUT FORMAT START>[{random_x_min}, {random_y_min}, {random_x_max}, {random_y_max}]<OUTPUT FORMAT END>"
    elif prompt_index in [PromptIndex.LEFT_QA.value, PromptIndex.RIGHT_QA.value]:
        select_object = random.choice(descriptions)
        return f"<OUTPUT FORMAT START>{select_object}<OUTPUT FORMAT END>"
    elif prompt_index in [PromptIndex.FRONT_QA.value, PromptIndex.BEHIND_QA.value]:
        judgement = random.choice(['Yes', 'No'])
        return f"<OUTPUT FORMAT START>{judgement}<OUTPUT FORMAT END>"
    elif prompt_index in [PromptIndex.DA_QA.value, PromptIndex.DZ_QA.value, PromptIndex.DAB_QA.value, PromptIndex.DX_QA.value]:
        distance = round(random.uniform(0, 70), 1)
        return f"<OUTPUT FORMAT START>{distance}<OUTPUT FORMAT END>"

    return None

def main(config):
    # Set a random seed for reproducibility
    random.seed(42)

    # Extract configuration parameters
    model_type = 'random'
    model = 'random'
    hf_dataset = config.hf_dataset
    save_dir = Path(config.save_dir)
    prompt_files_dir_list_dir = Path(config.prompts_dir)
    
    # Load dataset from Hugging Face
    logger.info(f"Loading Dataset...")
    dataset = load_dataset(hf_dataset, split='validation')
    dataset_num = len(dataset) # len(dataset)
    logger.info(f"Dataset loaded successfully with {dataset_num} data.")
 
    # To get a random result, only need a shot folder
    prompt_files_dir_list = os.listdir(str(prompt_files_dir_list_dir))
    prompt_files_dir_list.sort()
    prompt_files_dir_list = [prompt_files_dir_list[0]]

    # Iterate over each directory of prompt files
    for files_index in range(len(prompt_files_dir_list)):
        files_dir = prompt_files_dir_list[files_index]
        prompt_files_dir = prompt_files_dir_list_dir / files_dir

        # List and sort all prompt files within the current directory
        prompt_files = os.listdir(prompt_files_dir)
        prompt_files.sort()

        # Iterate over each prompt file
        prompt_dataset_sample_index = []
        for prompt_index in range(len(prompt_files)):

            prompt_file_name = str(prompt_files[prompt_index])
            prompt_file = str(prompt_files_dir / prompt_file_name)

            # Read prompt content from the file
            with open(prompt_file) as f:
                prompt_str = f.read()
            need_objects = prompt_str.count("{}")

            # Process each data sample in the dataset
            mllm_outputs = []
            dataset_sample_index = []
            for data_index in range(dataset_num):
                logger.info(f"Processing | Dir: {files_dir} ({files_index + 1}/{len(prompt_files_dir_list)}) | "
                        f"Prompt: {Path(prompt_file).name} ({prompt_index + 1}/{len(prompt_files)}) | "
                        f"Image ({data_index + 1}/{dataset_num})...")
                
                image = dataset[data_index]['image']
                original_descriptions = dataset[data_index]['descriptions']
                descriptions_num = len(original_descriptions)

                if descriptions_num < need_objects:
                    dataset_sample_index.append(dict(inference_sample_index = None))
                    continue
                else:
                    
                    # Randomly select descriptions to get the prompt
                    descriptions = random.sample(original_descriptions, need_objects)
                    sample_index = [original_descriptions.index(d) for d in descriptions]
                    dataset_sample_index.append(dict(inference_sample_index = sample_index))
                    descriptions = ["the " + d for d in descriptions] 
                    prompt = (prompt_str).format(*descriptions)

                    mllm_output = generate_random_output(prompt_index, image, descriptions)
                    logger.info(f"Output: {mllm_output}")
                    mllm_outputs.append(dict(dataset_index=data_index, output=mllm_output))     

            # Append the sample index list for the current prompt file
            prompt_dataset_sample_index.append(dataset_sample_index)

            # Save MLLM outputs to a JSON file for each prompt file
            save_json_dir = save_dir / model_type / Path(model).name / files_dir
            save_json_dir.mkdir(exist_ok=True, parents=True)
            save_json_file = save_json_dir /   f"{Path(prompt_file_name).stem}.json"
            with open(str(save_json_file), 'w') as json_file:
                json.dump(mllm_outputs, json_file, indent=4)
            logger.info(f"MLLM output saved to {str(save_json_file)}.")

        # Save the dataset sample indices for all prompt files in the current directory
        sample_index_json_file = save_json_dir /  f"inference_sample_index.json"
        with open(str(sample_index_json_file), 'w') as json_file:
            json.dump(prompt_dataset_sample_index, json_file, indent=4)
        logger.info(f"Inference sample index saved to {str(sample_index_json_file)}.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate random outputs.")
    parser.add_argument('--hf_dataset', type=str, default='bonbon-rj/MLLM_eval_dataset',
                        help='Specify the path to the Hugging Face dataset.')
    parser.add_argument('--prompts_dir', type=str, default='prompt/prompts',
                        help='Specify the directory that contains the input prompt files.')
    parser.add_argument('--save_dir', type=str, default='inference/mllm_outputs',
                        help='Define the directory where the generated output files will be saved.')
    args = parser.parse_args()
    main(args)

