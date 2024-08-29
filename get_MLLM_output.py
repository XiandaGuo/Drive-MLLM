from openai import OpenAI
from datasets import load_dataset
import io
import base64
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def pil2base64(pil_image):
    """Converts a PIL image object to a base64-encoded string."""
    binary_stream = io.BytesIO()
    pil_image.save(binary_stream, format="PNG")
    binary_data = binary_stream.getvalue()
    return base64.b64encode(binary_data).decode('utf-8')


def process_sample(pic_index, sample, text_input, headers, url, max_retry, max_tokens):
    image = sample['image']
    image_base64 = pil2base64(image)
    image_url = f"data:image/jpeg;base64,{image_base64}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_input
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    for i in range(max_retry):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_json = response.json()
            mllm_output = response_json['choices'][0]['message']['content']
            return pic_index, mllm_output
        except Exception as e:
            print(f"Error: {e}. Retry {i + 1} out of {max_retry}")
            continue
    return pic_index, None  # If all retries fail, return None


if __name__ == "__main__":
    max_retry = 5
    MODEL = "gpt-4o-2024-05-13"
    MAX_TOKENS = 500
    MAX_WORKERS = 5  # Number of parallel threads
    url = "your URL base"
    
    PROMPT_FILE = "./eval_prompt.txt"
    HF_DATASET = "bonbon-rj/MLLM_eval_dataset"
    SAVE_JSON_FILE = "./mllm_output_full.json"
    REQUEST_BY_OPENAI_CLIENT = False

    # get prompt
    object_type = "vehicle"
    with open(PROMPT_FILE) as f:
        eval_prompt = (f.read()).format(object_type)
    text_input = eval_prompt
    print(f"{'=' * 30}Eval prompt{'=' * 30}")
    print(eval_prompt)

    # get dataset
    print(f"{'=' * 30}Get dataset{'=' * 30}")
    print("This will take a long time.")
    dataset = load_dataset(HF_DATASET)
    # dataset_num = 2  # Adjusting for testing purposes
    dataset_num = len(dataset['train'])
    
    # get output
    print(f"{'=' * 30}Get MLLM output{'=' * 30}")
    mllm_outputs = []

    if REQUEST_BY_OPENAI_CLIENT:
        # Handle OpenAI client requests (omitted here)
        pass
    else:
        headers = {
            "Content-Type": "application/json",
            "Caller": "yourname"
        }

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_sample, 
                    pic_index,
                    dataset['train'][pic_index], 
                    text_input, 
                    headers, 
                    url, 
                    max_retry, 
                    MAX_TOKENS
                )
                for pic_index in range(dataset_num)
            ]

            for future in as_completed(futures):
                pic_index, result = future.result()
                if result:
                    mllm_outputs.append({'pic_index': pic_index, 'output': result})
                else:
                    print(f"Failed to get output after retries for pic_index {pic_index}")

    # save
    with open(SAVE_JSON_FILE, 'w') as json_file:
        json.dump(mllm_outputs, json_file, indent=4)
