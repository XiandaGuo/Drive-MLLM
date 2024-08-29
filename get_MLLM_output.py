from openai import OpenAI
from datasets import load_dataset
import io
import base64
import json


def pil2base64(pil_image):
    """Converts a PIL image object to a base64-encoded string."""
    binary_stream = io.BytesIO()
    pil_image.save(binary_stream, format="PNG")
    binary_data = binary_stream.getvalue()
    return base64.b64encode(binary_data).decode('utf-8')


if __name__ == "__main__":

    MODEL = "gpt-4o"
    MAX_TOKENS = 300

    PROMPT_FILE = "./eval_prompt.txt"
    HF_DATASET = "bonbon-rj/MLLM_eval_dataset"
    SAVE_JSON_FILE = "./mllm_output.json"
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
    dataset_num = 2  # len(dataset['train'])

    # get output
    print(f"{'=' * 30}Get MLLM output{'=' * 30}")
    mllm_outputs = []

    if REQUEST_BY_OPENAI_CLIENT:
        client = OpenAI()
        for pic_index in range(dataset_num):
            print(f"Handing: {pic_index}")
            sample = dataset['train'][pic_index]
            image = sample['image']

            image_base64 = pil2base64(image)
            image_url = f"data:image/jpeg;base64,{image_base64}"

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text_input
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, },
                            },
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
            )

            mllm_output = response.choices[0].message.content
            print(mllm_output)
            print()
            mllm_outputs.append(mllm_output)
    else:
        import requests
        api_key = ''
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        for pic_index in range(dataset_num):
            print(f"Handing: {pic_index}")
            sample = dataset['train'][pic_index]
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
                "max_tokens": MAX_TOKENS
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload,timeout=200)
            response_json = response.json()
            mllm_output = response_json['choices'][0]['message']['content']
            print(mllm_output)
            print()
            mllm_outputs.append(mllm_output)

    # save
    with open(SAVE_JSON_FILE, 'w') as json_file:
        json.dump(mllm_outputs, json_file, indent=4)
