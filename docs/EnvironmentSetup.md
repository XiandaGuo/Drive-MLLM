### Environment Setup
To get started, clone the repository and set up a conda environment with the required packages:

```shell
git clone https://github.com/XiandaGuo/Drive-MLLM.git
cd Drive-MLLM

conda create -n drive_mllm python=3.10
source activate drive_mllm
pip install -r requirements.txt

# setup PYTHONPATH
echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
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
