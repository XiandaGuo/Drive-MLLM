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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def inference_with_retry(
        inference: Callable,
        *args: Any,
        max_retries: int = 3,
        delay: int = 3,
        **kwargs: Any
) -> str:
    """
    Executes an inference function with automatic retries on failure.

    Args:
        inference (Callable[..., str]): Inference function to call.
        *args (Any): Positional arguments for the inference function.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        delay (int, optional): Time delay (seconds) between retry attempts. Defaults to 3 seconds.
        **kwargs (Any): Keyword arguments for the inference function.

    Returns:
        str: Output from the inference function on success.

    Raises:
        Exception: If the maximum number of retries is exceeded without success.
    """
    retries: int = 0
    while retries < max_retries:
        try:
            output_data: str = inference(*args, **kwargs)
            return output_data
        except Exception as e:
            logger.error(f"Error calling inference: {e}. Retrying {retries + 1}/{max_retries}...")
            retries += 1
            time.sleep(delay)

    raise Exception(f"Failed to complete inference after {max_retries} retries.")


def main(config):
    # Set a random seed for reproducibility
    random.seed(42)

    # Extract configuration parameters
    model_type = config.model_type
    model = config.model
    hf_dataset = config.hf_dataset
    save_dir = Path(config.save_dir)
    prompt_files_dir_list_dir = Path(config.prompts_dir)
    
    # Load dataset from Hugging Face
    logger.info(f"Loading Dataset...")
    dataset = load_dataset(hf_dataset, split='validation')
    dataset_num = len(dataset) # len(dataset)
    logger.info(f"Dataset loaded successfully with {dataset_num} data.")

    # Initialize inference interface based on the specified model type
    if model_type=="gpt":
        # Ensure OpenAI API key is available for GPT model
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError("OpenAI API key is missing. Set 'OPENAI_API_KEY' in environment variables.")
        
        # Import and initialize GPT interface
        from interface.gpt_interface import GptInterface
        inference_interface = GptInterface(api_key)
        inference_args = {"model": model}

    elif model_type=="gemini":
        # Ensure Google API key is available for Gemini model
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise EnvironmentError("Google API key is missing. Set 'GOOGLE_API_KEY' in environment variables.")
        
        # Ensure Google API key is available for Gemini model
        from interface.gemini_interface import GeminiInterface
        inference_interface = GeminiInterface(api_key)
        inference_args = {"model": model}

    elif model_type== "llava":
        # Initialize LLaVA interface
        from interface.llava_interface import LlavaInterface
        inference_interface = LlavaInterface(
            pretrained=model, 
            model_name="llava_qwen", 
            conv_template="qwen_1_5")
        inference_args = {}

    elif model_type== "qwen":
        # Initialize Qwen interface
        from interface.qwen2vl_interface import Qwen2VLInterface
        inference_interface = Qwen2VLInterface(model_name=model)
        inference_args = {}

    else:
        raise ValueError(f"Unsupported model type: {model_type}.")
    
    # List and sort all directories containing prompt files
    prompt_files_dir_list = os.listdir(str(prompt_files_dir_list_dir))
    prompt_files_dir_list.sort()

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

                    # Run inference
                    try:
                        # mllm_output = inference_interface.inference(image, prompt, **inference_args)
                        mllm_output = inference_with_retry(inference_interface.inference, image, prompt, **inference_args)
                        logger.info(f"Output: {mllm_output}")
                        mllm_outputs.append(dict(dataset_index=data_index, output=mllm_output))
                    except Exception as e:
                        logger.error(f"Failed to process image {data_index}: {e}")

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

    parser = argparse.ArgumentParser(description="Generate outputs using the MLLM model.")
    parser.add_argument('--model_type', type=str, default='gemini',
                        help='Specify the type of model to be used.')
    parser.add_argument('--model', type=str, default='models/gemini-1.5-flash',
                        help='Specify the name of the model within the selected type.')
    parser.add_argument('--hf_dataset', type=str, default='bonbon-rj/MLLM_eval_dataset',
                        help='Specify the path to the Hugging Face dataset.')
    parser.add_argument('--prompts_dir', type=str, default='prompt/prompts',
                        help='Specify the directory that contains the input prompt files.')
    parser.add_argument('--save_dir', type=str, default='inference/mllm_outputs',
                        help='Define the directory where the generated output files will be saved.')
    args = parser.parse_args()
    main(args)

