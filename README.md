# Benchmark-For-MLLM



## Dataset

We are using the Hugging Face dataset [MLLM_eval_dataset](https://huggingface.co/datasets/bonbon-rj/MLLM_eval_dataset) for evaluation. The images are sourced from the `CAM_FRONT` in the validation set of [nuScenes](https://www.nuscenes.org/). We have provided a `metadata.jsonl` file for all images, allowing users to easily access properties such as `bbox`.



## Getting Started

### Setup Environment

Follow these steps to create and activate a new Conda environment, and install the required packages:

```shell
conda create -n mllm_benchmark python=3.10
source activate mllm_benchmark
pip install -r requirements.txt
```

**Reference Links**:

- [OpenAI API quick start](https://platform.openai.com/docs/quickstart?language-preference=python)
- [Gemini API quick start](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)



## Get MLLM Output

### Setting Up API Keys

Depending on the API you are using, set the appropriate environment variable:

- For OpenAI:

```shell
export OPENAI_API_KEY=your_api_key
```

- For Gemini:

```shell
export GOOGLE_API_KEY=your_api_key
```



### Running the Script

To generate responses using the MLLM model based on prompts and images from a Hugging Face dataset, run the following script. Be sure to set the `api_type` parameter to either `gemini` or `openai` depending on your API key:

```shell
python get_MLLM_output.py --api_type gemini --prompt_file ./eval_prompt.txt --hf_dataset bonbon-rj/MLLM_eval_dataset
```

After running the script, you will receive a JSON file containing the results: `mllm_output_{api_type}.json`.



### Tips and Adjustments

- By default, the script uses the `gpt-4o` model for OpenAI and the `gemini-1.5-flash` model for Gemini. If you wish to modify the API call details, refer to `api_clients.py`, which contains detailed implementations of the API calls.
- For small batch testing, adjust the `dataset_num` parameter as needed to control the number of samples processed.

