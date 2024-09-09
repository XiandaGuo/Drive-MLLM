import time

from datasets import load_dataset
import io
import base64
import json
import os


def get_mllm_output_with_retry(pil_image, prompt, api_type, max_retries=3, delay=3):
    retries = 0
    while retries < max_retries:
        try:
            mllm_output = get_mllm_output(pil_image, prompt, api_type)
            return mllm_output
        except Exception as e:
            print(f"Failed to call get_mllm_output: {e}, retrying {retries + 1} time(s)...")
            retries += 1
            time.sleep(delay)

    raise Exception(f"Failed to call get_mllm_output after {max_retries} retries")


def get_mllm_output(pil_image, prompt, api_type):
    assert api_type == "openai" or api_type == "gemini"

    if api_type == "openai":
        from openai import OpenAI

        def pil2base64(pil_image):
            """Converts a PIL image object to a base64-encoded string."""
            binary_stream = io.BytesIO()
            pil_image.save(binary_stream, format="PNG")
            binary_data = binary_stream.getvalue()
            return base64.b64encode(binary_data).decode('utf-8')

        api_key = os.environ.get('OPENAI_API_KEY')
        model = "gpt-4o"
        max_tokens = 300
        client = OpenAI(api_key=api_key)

        image_base64 = pil2base64(pil_image)
        image_url = f"data:image/jpeg;base64,{image_base64}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        output = response.choices[0].message.content
        return output

    elif api_type == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get('GOOGLE_API_KEY')
        genai.configure(api_key=api_key, transport='rest')
        # for m in genai.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(m.name)
        # model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")  # flash pro...
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")  # flash pro...

        output = model.generate_content([prompt, pil_image]).text

        return output


if __name__ == "__main__":
    API_TYPE = "gemini"  # openai gemini
    PROMPT_FILE = "./eval_prompt.txt"
    HF_DATASET = "bonbon-rj/MLLM_eval_dataset"
    SAVE_JSON_FILE = f"./mllm_output_{API_TYPE}.json"

    # get prompt
    object_type = "vehicle"
    with open(PROMPT_FILE) as f:
        eval_prompt = (f.read()).format(object_type)
    prompt = eval_prompt
    print(f"{'=' * 30}Eval prompt{'=' * 30}")
    print(eval_prompt)

    # get dataset
    print(f"{'=' * 30}Get dataset{'=' * 30}")
    print("This will take a long time.")
    dataset = load_dataset(HF_DATASET, split='validation')
    data = dataset
    dataset_num = len(data)
    print(f"Get {dataset_num} data.")

    # get output
    print(f"{'=' * 30}Get MLLM output{'=' * 30}")

    mllm_outputs = []

    for data_index in range(dataset_num):
        print(f"Handing {data_index}...")
        image = data[data_index]['image']

        # InternalServerError
        # TooManyRequests
        mllm_output = get_mllm_output_with_retry(image, prompt, API_TYPE, max_retries=10)

        print(mllm_output)
        print()
        mllm_outputs.append(dict(pic_index=data_index, output=mllm_output))

    # save
    with open(SAVE_JSON_FILE, 'w') as json_file:
        json.dump(mllm_outputs, json_file, indent=4)
