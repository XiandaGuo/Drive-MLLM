# Benchmark-For-MLLM



### Script

Script name: `get_MLLM_output.py`

Overview: This script utilizes OpenAI's MLLM model to generate responses based on a given `PROMPT_FILE` and images from a specified Hugging Face dataset (`HF_DATASET`). The responses are then saved in a JSON file (`SAVE_JSON_FILE`).

Implementation: The script reads images from the specified Hugging Face dataset in PIL format, converts them to base64 encoding, and then uses OpenAI's API to generate responses.

Tips：

Before using the script, ensure the following:

- `MODEL` is correctly specified.
- `MAX_TOKENS` is set to a sufficient value.
- `PROMPT_FILE` points to a valid prompt file.
- `HF_DATASET` corresponds to the correct dataset.
- `SAVE_JSON_FILE` is the correct path for saving the output.

To perform small-batch testing, adjust the `dataset_num` parameter accordingly.