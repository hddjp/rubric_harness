# RaR 实验运行指南

本文说明依赖安装、模型下载、judge 服务启动和六组训练实验的运行流程。

六组实验由以下组合构成：

| reward 脚本 | 数据集 |
| --- | --- |
| `run_baseline.sh` | medicine、science |
| `run_shipped.sh` | medicine、science |
| `run_agentic.sh` | medicine、science |

## 1. 创建环境并安装依赖

在项目根目录执行：

```bash
conda create -n ssttyy python=3.12 -y
conda activate ssttyy

python -m pip install -r rl/simple.txt

# nb.txt 中包含需要源码构建的依赖，关闭 build isolation。
# Apex 需要这两个环境变量来编译 C++/CUDA 扩展。
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
python -m pip install --no-build-isolation -r rl/nb.txt
```

如果当前 shell 尚未初始化 Conda，先执行对应的 `conda init`，重新打开 shell 后再激活环境。

## 2. 下载模型

在项目根目录执行：

```bash
conda activate ssttyy

python rl/src/download_hf_model.py Qwen/Qwen3.8-27B
python rl/src/download_hf_model.py Qwen/Qwen3.5-4B
```

脚本默认使用 Hugging Face 的缓存目录，并支持断点续传。也可以指定缓存位置：

```bash
python rl/src/download_hf_model.py Qwen/Qwen3.8-27B \
  --cache-dir /path/to/huggingface-cache
```

如果模型仓库需要权限，增加 `--token <HF_TOKEN>`，或提前配置 Hugging Face 登录信息。

## 3. 启动 judge 服务

`serve_rar_judge.sh` 默认使用 4 张 GPU，监听 `0.0.0.0:8000`：

```bash
conda activate ssttyy
CUDA_VISIBLE_DEVICES=0,1,2,3 ./rl/serve_rar_judge.sh
```

脚本会启动 `Qwen/Qwen3.8-27B` 的 OpenAI 兼容接口。需要调整服务并发量时，可在启动前覆盖环境变量，例如：

```bash
# 24 GB × 4：保持默认参数
CUDA_VISIBLE_DEVICES=0,1,2,3 ./rl/serve_rar_judge.sh

# 48 GB × 4：提高并发序列数
RAR_JUDGE_MAX_NUM_SEQS=96 \
CUDA_VISIBLE_DEVICES=0,1,2,3 ./rl/serve_rar_judge.sh
```

如果训练和服务不在同一台机器，将 `0.0.0.0` 对应机器的可访问 IP 和端口记下来，例如：

```text
http://10.0.0.12:8000/v1
```

## 4. 配置 judge 服务地址

把上一步的地址填入 [src/shipped_reward.py](src/shipped_reward.py) 的 `JUDGE_BASE_URL`：

```python
JUDGE_BASE_URL = "http://10.0.0.12:8000/v1"
```

如果服务和训练在同一台机器，使用：

```python
JUDGE_BASE_URL = "http://127.0.0.1:8000/v1"
```

`baseline_reward.py` 和 `agentic_reward.py` 复用 `shipped_reward.py` 中的 judge 实现，因此也使用这个地址。

## 5. 启动训练实验

三个脚本都需要传入数据集参数：

```bash
./rl/run_baseline.sh medicine
./rl/run_baseline.sh science

./rl/run_shipped.sh medicine
./rl/run_shipped.sh science

./rl/run_agentic.sh medicine
./rl/run_agentic.sh science
```

也可以使用完整路径从任意工作目录启动。参数不区分大小写，合法值只有 `medicine` 和 `science`。

每个训练脚本默认使用 4 张 GPU。judge 服务也默认占用 4 张 GPU，因此应为服务和训练任务分配互不冲突的 GPU；并行运行两个训练任务时，也要确保有足够的剩余 GPU 和显存。

建议根据显存安排 judge 服务与训练任务：

- 24 GB × 4：使用默认的 `RAR_JUDGE_MAX_NUM_SEQS=32`，一个 judge 服务对应一个训练任务更稳妥。
- 48 GB × 4：可以使用 `RAR_JUDGE_MAX_NUM_SEQS=96`，一个 judge 服务可尝试对应两个训练任务。

训练脚本会自动根据自身位置定位 `data` 和 `src`，不要求从 `rl/` 目录启动。Python 默认使用当前激活环境中的 `python`；如需指定其他环境，可以覆盖：

```bash
PYTHON_BIN=/path/to/python ./rl/run_agentic.sh medicine
```

训练日志和 checkpoint 的具体位置由 VERL 配置及运行参数决定；启动前请确认训练 GPU、judge GPU 和模型缓存空间均已正确分配。
