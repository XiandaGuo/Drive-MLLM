# SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models

## Update
We have changed the title from "DriveMLLM: A Benchmark for Spatial Understanding with Multimodal Large Language Models in Autonomous Driving" to "SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models". If you use the data from the first version of DriveMLLM, you can use the v1 branch.

## Dataset

We extracted and processed data from the [nuScenes](https://www.nuscenes.org/) dataset to create our own [SURDS](https://huggingface.co/datasets/bonbon-rj/SURDS) dataset for training and evaluation purposes.  Due to the large size of the training data, we also provide a separate evaluation-only version: [SURDS_eval](https://huggingface.co/datasets/bonbon-rj/SURDS_eval).  A `metadata.jsonl` file is included for all images, allowing users to conveniently access properties such as `xy2Ds`.



## Getting Started

### Environment Setup

To get started, follow the steps below to set up the environment:

```shell
# Clone the repository and add it to PYTHONPATH
git clone https://github.com/XiandaGuo/Drive-MLLM.git
cd Drive-MLLM
echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

# Create a Conda environment and install core dependencies
conda create -n surds python=3.10 
source activate surds
pip install -r requirements.txt

# Set up the Qwen2-VL environment
git clone https://github.com/QwenLM/Qwen2-VL.git
cd Qwen2-VL
pip install -r requirements_web_demo.txt
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils[decord]
pip install flash-attn --no-build-isolation --no-cache-dir  # (Recommended) 
pip install transformers==4.50.0 # Stable version for this project
cd ..

# Install SGLang with acceleration support
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]==0.4.4.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python # Different versions of SGLang may adopt varying acceleration strategies
```



**Reference Links**:

- [Qwen2-VL Official Github Website](https://github.com/QwenLM/Qwen2-VL)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [SGLang installation](https://docs.sglang.ai/start/install.html)



### VQAs Generation

To generate Visual Question-Answering (VQA) examples for evaluation, run the script below. It downloads the dataset from Hugging Face, applies the prompts provided in `<prompt_dir>`, and stores the generated VQAs in the `<vqas_save_dir>` directory.

```shell
python hfdata_to_eval_vqa.py \
--hf_dataset bonbon-rj/SURDS_eval \
--prompt_dir prompt/prompts_reasoning \
--vqas_save_dir eval_vqas_reasoning
```



### Inference

#### Running Inference with SGLang

To perform inference on the `vqas_dir` prompts using [SGLang](https://github.com/sgl-project/sglang), execute the script below. 

The example below demonstrates inference with the `Qwen/Qwen2.5-VL-3B-Instruct` model on 8 × 80 GB GPUs:

```shell
python inference/get_vlm_output_sglang.py \
--save_dir inference/vlm_outputs \
--save_sub_dir qwen \
--vqas_dir eval_vqas_reasoning \
--bs_per_req 1850 \
--sglang_model "Qwen/Qwen2.5-VL-3B-Instruct" \
--sglang_tpl qwen2-vl \
--sglang_dtype bfloat16 \
--sglang_mem 0.9 \
--sglang_maxreq 64 \
--sglang_dp 8 \
--sglang_tp 1
```

The results will be saved to the directory: `<save_dir>/<save_sub_dir>/<sglang_model>`.



#### Generating Random Outputs

To obtain random outputs on the `<vqas_dir>` prompts, run:

```shell
python inference/get_vlm_output_random.py \
--save_dir inference/vlm_outputs \
--vqas_dir eval_vqas_reasoning 
```

The results will be saved to the directory: `<save_dir>/random/random`.



#### Adapting Unsupported Models

If your target model is not yet supported by SGLang, you can use `get_vlm_output_random.py` as a template and replace the `generate_random_output` function with your model’s inference implementation.



### Evaluation

To evaluate all model outputs stored in `<eval_root_dir>`, you can run the following script:

```shell
python evaluation/eval_from_json.py \
--vqas_dir eval_vqas_reasoning \
--eval_root_dir inference/vlm_outputs \
--eval_model_path all \
--save_dir evaluation/eval_result 
```

Alternatively, to evaluate a specific model's output under `<eval_root_dir>`, specify the desired `<eval_model_path>`:

```shell
python evaluation/eval_from_json.py \
--vqas_dir eval_vqas_reasoning \
--eval_root_dir inference/vlm_outputs \
--eval_model_path qwen/Qwen2.5-VL-3B-Instruct \
--save_dir evaluation/eval_result 
```

After running the scripts, the evaluation results will be stored in the directory: `<save_dir>`.



### Training

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning (SFT), and adopt the [VLM-R1](https://github.com/om-ai-lab/VLM-R1) framework to train the model using Group Relative Policy Optimization (GRPO).

To prepare SFT data with chain-of-thought (CoT) reasoning, use the provided scripts: `summarize_rules.py` and `gen_cot.py`.

For reinforcement learning, the GRPO implementation is available in `grpo.py`.
