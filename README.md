# Benchmark-For-MLLM



### Script

Script Name: `get_MLLM_output.py`

Overview: This script leverages the MLLM models from OpenAI or Gemini to generate responses based on a given prompt and images from a Hugging Face dataset. The generated responses are saved in a JSON file.

Tips:

Before using the script, please ensure the following:

- The `API_TYPE` is correctly specified. It must be either `openai` or `gemini`.
- Set the appropriate `api_key` environment variable based on the `API_TYPE`:
  - For OpenAI, set the environment variable `OPENAI_API_KEY`.

  - For Gemini, set the environment variable `GOOGLE_API_KEY`.
- The `PROMPT_FILE` points to a valid prompt file.
- The `HF_DATASET` corresponds to the correct dataset.
- The `SAVE_JSON_FILE` is the correct path to save the output.
- By default, the script calls the `"gpt-4o"` model for OpenAI and the `gemini-1.5-flash` model for Gemini. To modify this, update the `get_mllm_output` function accordingly.

For small batch testing, adjust the `dataset_num` parameter as needed.