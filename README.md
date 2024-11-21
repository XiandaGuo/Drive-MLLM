# DriveMLLM: A Benchmark for Spatial Understanding with Multimodal Large Language Models in Autonomous Driving

### [Paper](https://arxiv.org/abs/2411.13112)

> [DriveMLLM: A Benchmark for Spatial Understanding with Multimodal Large Language Models in Autonomous Driving](https://arxiv.org/abs/2411.13112)

> [Xianda Guo*](https://scholar.google.com/citations?user=jPvOqgYAAAAJ), Ruijun Zhang*, [Yiqun Duan*](https://scholar.google.com.hk/citations?user=NmwjI0AAAAAJ&hl=zh-CN), Yuhang He, Chenming Zhang, Long Chen.

## News 
- **[2024/11]** Paper released on [arXiv](https://arxiv.org/abs/2411.13112).

## Overall
![vis](radar.jpg)

## Getting Started

### 0. Prepare Dataset

We are using the Hugging Face dataset [MLLM_eval_dataset](https://huggingface.co/datasets/bonbon-rj/MLLM_eval_dataset) for evaluation. The images are sourced from the `CAM_FRONT` in the validation set of [nuScenes](https://www.nuscenes.org/). We have provided a `metadata.jsonl` file for all images, allowing users to easily access properties such as `location2D`.


### 1. [Setup Environment](docs/EnvironmentSetup.md)

### 2. Inference

Run inference according to your requirements:
- For GPT API calls:

```shell
export OPENAI_API_KEY=your_api_key

python inference/get_MLLM_output.py \
--model_type gpt \
--model gpt-4o \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir prompt/prompts \
--save_dir inference/mllm_outputs
```

- For Gemini API calls:

```shell
export GOOGLE_API_KEY=your_api_key

python inference/get_MLLM_output.py \
--model_type gemini \
--model models/gemini-1.5-flash \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir prompt/prompts \
--save_dir inference/mllm_outputs
```

- For Local LLaVA-Next inference:
```shell
python inference/get_MLLM_output.py \
--model_type llava \
--model lmms-lab/llava-onevision-qwen2-7b-si \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir prompt/prompts \
--save_dir inference/mllm_outputs
```

- For Local QWen2-VL inference:
```shell
python inference/get_MLLM_output.py \
--model_type qwen \
--model Qwen/Qwen2-VL-7B-Instruct \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir prompt/prompts \
--save_dir inference/mllm_outputs
```



Run script to get the random output for the prompts:

```shell
python inference/get_random_output.py \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir prompt/prompts \
--save_dir inference/mllm_outputs
```



After executing the script, the results will be saved in the directory: `{save_dir}/{model_type}/{model}`.



### 3. Evaluation

You can execute the script below to evaluate all results located in `eval_root_dir`:

```shell
python evaluation/eval_from_json.py \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--eval_root_dir inference/mllm_outputs \
--save_dir evaluation/eval_result \
--eval_model_path all 
```

Alternatively, you can also run the following script to evaluate a specific result under `eval_root_dir` by specifying a model `eval_model_path`:

```shell
python evaluation/eval_from_json.py \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--eval_root_dir inference/mllm_outputs \
--save_dir evaluation/eval_result \
--eval_model_path gemini/gemini-1.5-flash
```

After running the scripts, the evaluation results will be stored in the directory: `{save_dir}`.
## Citation
```
@article{DriveMLLM,
        title={DriveMLLM: A Benchmark for Spatial Understanding with Multimodal Large Language Models in Autonomous Driving},
        author={Guo, Xianda and Zhang Ruijun and Duan Yiqun and He Yuhang and Zhang, Chenming and Chen, Long},
        journal={arXiv preprint arXiv:2411.13112},
        year={2024}
}
```
