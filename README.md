# Benchmark-For-MLLM


## Dataset

We are using the Hugging Face dataset [MLLM_eval_dataset](https://huggingface.co/datasets/bonbon-rj/MLLM_eval_dataset) for evaluation. The images are sourced from the `CAM_FRONT` in the validation set of [nuScenes](https://www.nuscenes.org/). We have provided a `metadata.jsonl` file for all images, allowing users to easily access properties such as `location2D`.



## Getting Started

### Setup Environment

To get started, clone the repository and set up a conda environment with the required packages:

```shell
git clone https://github.com/XiandaGuo/Benchmark-For-MLLM.git
cd Benchmark-For-MLLM

conda create -n mllm_benchmark python=3.10
source activate mllm_benchmark
pip install -r requirements.txt
```

### Additional Environment Setup
Depending on your needs, set up the following environments for API calls or local model inference:
- For GPT API calls:

```shell
pip install openai==1.42.0
```

- For Gemini API calls:
```shell
pip install google-generativeai==0.7.2
```

- For Local LLaVA-Next inference:

```shell
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT/
pip install --upgrade pip  
pip install -e ".[train]" 
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git 
cd ..

## flash atten (optional)
conda install -c "nvidia/label/cuda-12.1.0" cuda
pip install flash-attn --no-build-isolation --no-cache-dir
```

- For Local QWen2-VL inference:

```shell
git clone https://github.com/QwenLM/Qwen2-VL.git
cd Qwen2-VL
pip install -r requirements_web_demo.txt
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils[decord]
cd ..

## flash atten (optional)
conda install -c "nvidia/label/cuda-12.1.0" cuda
pip install flash-attn --no-build-isolation --no-cache-dir
```


**Reference Links**:

- [OpenAI API Quick Start](https://platform.openai.com/docs/quickstart?language-preference=python)
- [Gemini API Quick Start](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)
- [LLaVA-NeXT Official Gitgub Website](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Qwen2-VL Official Github Website](https://github.com/QwenLM/Qwen2-VL)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Cuda Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases)



## Get MLLM Output

Run inference according to your requirements:
- For GPT API calls:

```shell
export OPENAI_API_KEY=your_api_key

python get_MLLM_output.py \
--model_type gpt \
--model gpt-4o \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir ./prompts \
--save_dir ./mllm_outputs
```

- For Gemini API calls:

```shell
export GOOGLE_API_KEY=your_api_key

python get_MLLM_output.py \
--model_type gemini \
--model models/gemini-1.5-flash \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir ./prompts \
--save_dir ./mllm_outputs
```

- For Local LLaVA-Next inference:
```shell
# run LLaVA-OneVision
python get_MLLM_output.py \
--model_type llava \
--model lmms-lab/llava-onevision-qwen2-7b-si \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir ./prompts \
--save_dir ./mllm_outputs
```

- For Local QWen2-VL inference:
```shell
python get_MLLM_output.py \
--model_type qwen \
--model Qwen/Qwen2-VL-7B-Instruct \
--hf_dataset bonbon-rj/MLLM_eval_dataset \
--prompts_dir ./prompts \
--save_dir ./mllm_outputs
```

After executing the script, the results will be saved in the directory: `{save_dir}/{model_type}`.


