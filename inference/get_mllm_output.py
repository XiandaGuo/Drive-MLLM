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
    save_dir = Path(config.save_dir)
    vqas_dir = Path(config.vqas_dir)

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
        inference_interface = Qwen2VLInterface(model_name=model, not25=False if "2.5" in model else True)
        inference_args = {}

    else:
        raise ValueError(f"Unsupported model type: {model_type}.")
    
    # List all json
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

            logger.info(f"Processing | File: {vqa_file} ({vqa_file_idx + 1}/{len(vqa_files)}) | VQAs: {image_path.name} ({vqa_idx + 1}/{len(vqas)}) ...")
            
            # Run inference
            try:
                # mllm_output = inference_interface.inference(image, prompt, **inference_args)
                vlm_output = inference_with_retry(inference_interface.inference, image, prompt, **inference_args)
                logger.info(f"Output: {vlm_output}")
                vlm_outpus.append(dict(vqa_idx=vqa_idx, image=image_path.name, output=vlm_output))
                
            except Exception as e:
                logger.error(f"Failed to process vqa {vqa_idx}: {e}")
                vlm_outpus.append(dict(vqa_idx=vqa_idx, image=image_path.name, output=''))   

        # Save MLLM outputs to a JSON file for each prompt file
        save_json_dir = save_dir / model_type / Path(model).name 
        save_json_dir.mkdir(exist_ok=True, parents=True)
        save_json_file = save_json_dir /   f"{Path(vqa_file).stem}_output.json"
        with open(str(save_json_file), 'w') as json_file:
            json.dump(vlm_outpus, json_file, indent=4)
        logger.info(f"MLLM output saved to {str(save_json_file)}.")
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate outputs using the MLLM model.")
    parser.add_argument('--model_type', type=str, default='gemini',
                        help='Specify the type of model to be used.')
    parser.add_argument('--model', type=str, default='models/gemini-1.5-flash',
                        help='Specify the name of the model within the selected type.')
    parser.add_argument('--save_dir', type=str, default='inference/mllm_outputs',
                        help='Define the directory where the generated output files will be saved.')
    parser.add_argument('--vqas_dir', type=str, default='eval_vqas',
                        help='Specify the folder for the VQAs.')
    args = parser.parse_args()
    main(args)

