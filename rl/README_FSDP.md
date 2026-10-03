# RaR rubric RL：FSDP 版运行手册

这份手册说明怎样在一台 8×80GB GPU 的机器上跑完六组实验：三种 rubric（agentic、baseline、shipped）乘以两个领域（science、medicine）。它替代 [README.md](README.md) 里的 megatron 流程，那套流程是为 24GB 卡设计的。所有改动和原因都在第 9 节。

实验目标是比较 agentic rubric 和同一生成模型（GLM-5.3）单次合成的 baseline rubric 作为 GRPO reward 的效果。shipped 是论文自带的 rubric，作为参考点。

## 0. 机器布局

| GPU | 用途 |
|---|---|
| 0,1 | judge 实例一，Qwen3.8-27B，TP=2，端口 8000 |
| 2,3 | judge 实例二，Qwen3.8-27B，TP=2，端口 8001 |
| 4,5 | 训练通道 A：agentic×science，然后 agentic×medicine，然后 shipped×science |
| 6,7 | 训练通道 B：baseline×science，然后 baseline×medicine，然后 shipped×medicine |

同一领域的 agentic 和 baseline 同时跑、共享 judge，judge 负载对两组一样。

在 47rc 上，两组并发时每步约 14 到 15 分钟。300 步一轮约 3 天，三轮约 9 天。

## 1. 环境

需要 Python 3.12 和 CUDA 12.8 的 PyTorch 轮子。关键版本如下，完整锁定清单见 `rl/requirements_fsdp.lock.txt`。

| 包 | 版本 | 说明 |
|---|---|---|
| torch | 2.10.0 | |
| vllm | 0.17.0 | |
| verl | 0.7.1 | PyPI 原版，没有打补丁 |
| transformers | 5.13.0 | qwen3_5 的模型定义只在 transformers 5 里 |
| flashinfer-python | 0.6.12 | vLLM 的 GDN 线性注意力 kernel |
| flash-linear-attention | 0.5.2 | 训练端 Qwen3.5 线性注意力层的 kernel，没有它反向传播极慢 |
| tilelang | 0.1.15 | 必装，原因见下 |
| flash_attn | 2.8.3 | |

```bash
conda create -n ssttyy python=3.12 -y
conda activate ssttyy
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r rl/requirements_fsdp.lock.txt
# flash_attn 如果需要编译：pip install flash_attn==2.8.3 --no-build-isolation
```

装完先检查这两项：

```bash
python -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f; print('fla', f())"
python -c "import tilelang, triton; print(tilelang.__version__, triton.__version__)"
```

**为什么要 tilelang。** fla 检测到 Hopper GPU 上 Triton 版本低于 3.7.1 时，会因为已知的 GDN 反向传播正确性 bug 直接报错。升级 Triton 会破坏 torch 和 vLLM 的版本绑定，装上 tilelang 后 fla 会改走 tilelang 的 kernel。我们在 47rc 上对比过 PyTorch 参考实现，前向和梯度的相对误差约 0.4%，属于 bf16 正常范围。

## 2. 模型和数据

```bash
python rl/src/download_hf_model.py Qwen/Qwen3.5-4B
python rl/src/download_hf_model.py Qwen/Qwen3.8-27B
```

训练脚本设置了 `HF_HUB_OFFLINE=1`，所以模型必须提前下载好。

训练数据和 train split 的 rubric 都在 HF 数据集 `soyorinlove/rubric_harness` 上，git 仓库里没有。一定要用它自带的 `verl-newnew`：这是作者筛过的训练集（medicine 15,983 题，science 12,400 题），每题都有 agentic 和 baseline 两种 rubric。自己用 `prepare_verl_data.py` 和 `dedupe_rar_datasets.py` 生成的版本题目更多，多出来的题查不到 rubric，会被静默记为 0 分。

```bash
python -c "from huggingface_hub import snapshot_download as d; d('soyorinlove/rubric_harness', repo_type='dataset', local_dir='data/hf_soyorinlove')"
mkdir -p rl/data/rubrics
ln -s ../../data/hf_soyorinlove/verl-newnew rl/data/verl-newnew
for f in data/hf_soyorinlove/rubrics/*.jsonl; do ln -s ../../../$f rl/data/rubrics/; done
```

held-out 验证集用下面两条命令生成。GPQA 不随仓库分发，需要在本地构建：

```bash
python rl/eval/download_eval_data.py --output-dir rl/eval/data --skip-healthbench
python rl/src/build_heldout.py
```

| 文件 | 内容 | 打分方式 |
|---|---|---|
| `rl/data/heldout/RaR-Science_val500.parquet` | RaR val 固定抽 500 题，种子 0 | shipped rubric 加 judge |
| `rl/data/heldout/RaR-Medicine_val500.parquet` | 同上 | 同上 |
| `rl/data/heldout/gpqa_diamond.parquet` | GPQA-Diamond 198 题，每题用固定种子打乱选项 | 规则判分，比对 `\boxed{字母}`，不经过 judge |

GPQA 原始数据里正确答案都排在第一个选项，所以必须打乱。science 的两组同时用 RaR val 和 GPQA 做验证，medicine 只用 RaR val。

## 3. 启动 judge

```bash
PYTHON_BIN=$(which python) JUDGE_GPU_GROUPS="0,1 2,3" setsid nohup ./rl/serve_judge_dp.sh >/dev/null 2>&1 &
for p in 8000 8001; do curl -s -m 3 -o /dev/null -w "%{http_code} " localhost:$p/v1/models; done   # 两个 200 就绪
```

每个实例都在自己的重启循环里，崩溃后约 20 秒自动拉起。日志在 `logs/judge_<端口>.log` 和 `logs/judge_supervise.log`。服务只监听 127.0.0.1。如果 judge 和训练不在同一台机器，要改 `serve_judge_dp.sh` 里的 `--host`，并设置 `JUDGE_BASE_URLS`。

## 4. 冒烟测试（正式训练前必做）

先跑 2 步，覆盖训练、验证和存 checkpoint 三条路径，约 30 到 40 分钟：

```bash
export PYTHON_BIN=$(which python) VLLM_WORKER_MULTIPROC_METHOD=spawn NGPU=2
CUDA_VISIBLE_DEVICES=4,5 TOTAL_STEPS=2 SAVE_FREQ=2 TEST_FREQ=2 VAL_BEFORE_TRAIN=False \
  VAL_MAX_SAMPLES=32 CKPT_DIR=/tmp/smoke_ckpt \
  ./rl/run_fsdp.sh agentic science trainer.project_name=smoke 2>&1 | tee logs/smoke.log
grep -oE "step:[12] .*" logs/smoke.log | tr ' ' '\n' | grep -E "timing_s/step|critic/score/mean|val-core|judge_failed"
```

通过的标准是两步都打出指标，`val-core` 有数值，`/tmp/smoke_ckpt/global_step_2` 存在，`judge_failed` 接近 0。

## 5. 正式训练

```bash
export PYTHON_BIN=$(which python) VLLM_WORKER_MULTIPROC_METHOD=spawn NGPU=2
CUDA_VISIBLE_DEVICES=4,5 setsid nohup ./rl/queue_lane.sh agentic:science agentic:medicine shipped:science >/dev/null 2>&1 &
# 等第一组打出 "dataset len" 再起第二条，避免两个 Ray 集群同时初始化
CUDA_VISIBLE_DEVICES=6,7 setsid nohup ./rl/queue_lane.sh baseline:science baseline:medicine shipped:medicine >/dev/null 2>&1 &
setsid nohup ./rl/strip_optim.sh >/dev/null 2>&1 &
```

调用链是 `queue_lane.sh`，然后是 `supervise_fsdp.sh`，最后是 `run_fsdp.sh`。

- **自动续跑。** 训练崩溃后会从最近的 checkpoint 自动续跑，最多重试 5 次。每 10 步存一次 checkpoint。
- **checkpoint 位置**在 `rl/checkpoints/<组名>/rar_<领域>/`。日志在 `logs/train_<组名>_<领域>.attempt<N>.log`。

`run_fsdp.sh` 的超参和原 `run_*.sh` 一致，只有学习率不同。可以用环境变量覆盖的参数：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LR` | 1e-6 | 原脚本是 5e-6，原因见第 9 节 |
| `TOTAL_STEPS` | 300 | 每步 32 题，每题 8 个回答 |
| `SAVE_FREQ` / `TEST_FREQ` | 10 / 25 | |
| `VAL_BEFORE_TRAIN` | True | 训练前先测一次，作为起点 |
| `VAL_MAX_SAMPLES` | -1 | 验证集全量 |
| `NGPU` | 2 | 每组训练用的卡数 |
| `CKPT_DIR` | `rl/checkpoints/<组名>/rar_<领域>` | |
| `JUDGE_BASE_URLS` | 8000 和 8001 两个端口 | 逗号分隔 |
| `JUDGE_MAX_CONCURRENT` | 16 | 每个 reward worker 的并发上限，每组共 8 个 worker |
| `JUDGE_OUTAGE_SECONDS` | 900 | judge 不可用时最长重试时间 |

## 6. 监控

```bash
python rl/progress.py            # 进度条、held-out 验证表、最近 10 步的训练健康度
python rl/build_dashboard.py     # 生成 logs/dashboard.html，带曲线，可在浏览器打开
```

- **要看的指标。** 组间比较只看 held-out 验证，也就是 `val-core/gpqa_diamond` 和 `val-core/rar_*`。训练 reward 用的是各组自己的 rubric，不能跨组比。
- **GPQA 的噪声。** 同一个模型重复测一次，GPQA 就可能差 2 到 3 个点。训练初期它接近 0，因为思考模式在 3584 token 内大多没写完。
- **失稳信号。** 和参考模型的 KL（`actor/kl_loss`）超过 1，或者 entropy 快速坍缩，都说明训练在失稳。lr=1e-6 时前 14 步 KL 约 0.002，entropy 约 0.7。

## 7. 磁盘

每个 checkpoint 约 51GB，其中 20GB 是模型权重，32GB 是 optimizer 状态。`strip_optim.sh` 每 10 分钟清理一次：

- **训练中的组。** 只给比最新 checkpoint 更旧的那些删掉 optimizer 状态。
- **训练结束的组。** 所有 checkpoint 都删掉 optimizer 状态。

权重始终保留。六组全部跑完时，checkpoint 总共约 240GB。

## 8. 停止和恢复

停止时先杀守护脚本，再杀训练进程。否则守护脚本会立刻把训练重新拉起来。

```bash
kill $(ps -u $(id -u) -o pid,args | grep -E "queue_lane.sh|supervise_fsdp.sh|rl/run_fsdp.sh" | grep -v grep | awk '{print $1}')
for g in 4 5 6 7; do for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; done
```

不要用 `ray stop`，它会去碰机器上别人的 Ray 集群。重新执行第 5 节的命令即可恢复，`resume_mode=auto` 会从最新的 checkpoint 接着跑。

## 9. 相对原 README 的改动和原因

1. **后端从 megatron 改为 FSDP。** megatron、TransformerEngine 和 apex 需要用 CUDA 12.4 的 nvcc 去编译 cu128 的 torch，风险很大。在 80GB 卡上训 4B 模型也不需要 offload。
2. **用固定 micro-batch，关掉动态 batch。** verl 0.7.1 的 `dp_actor` 调用 `prepare_dynamic_batch` 时没有传 data-parallel 组，两个 rank 可能切出不同数量的 micro-batch，FSDP 通信对不上就会死锁，通常发生在第 29 到 40 步，表现是 NCCL watchdog 超时。现在的设置是 `ppo_micro_batch_size_per_gpu=4`，数学上和动态 batch 等价。
3. **学习率从 5e-6 改为 1e-6。** 在 fp32 主权重下，5e-6 跑到约第 100 步时完全失稳：和参考模型的 KL 从 0.05 涨到 2.5 到 4.6，entropy 从 0.67 塌到 0.19，验证分数在两组都下降。原 megatron 配置把主权重存成 bf16，很可能把大部分 5e-6 的更新舍入掉了，所以那边看不出问题。这一点没有经过验证。
4. **权重同步 bucket 调到 4GB。** FSDP 下 fp32 的 embedding 有 2.5GB，超过默认 2GB 的上限。
5. **reward 返回值改为数值字段。** 原代码成功时返回 `rating: None`，verl 汇总验证指标时会因此崩溃，按原脚本会在第 25 步崩。现在改为数值型的 `judge_failed`（0 或 1），日志里还能看到 judge 失败率。
6. **验证集统一用 shipped rubric 打分。** 大部分 val 题没有 agentic 或 baseline 的 rubric，原代码会把它们记 0 分。现在三组的验证都用 shipped rubric，GPQA 用规则判分。训练时的 reward 不受影响。
7. **judge 改成 2 个 TP=2 的独立实例**，替代原来一个 TP=4 的服务。47rc 上 TP=4 的服务在机器内存压力大时发生过 worker 卡死，整个服务随之退出。单卡实例也试过，但装下 27B 后 KV cache 只够同时处理约 27 个请求，每步拖到 26 分钟。
8. **judge 故障时重试，不再记 0 分。** 遇到连接失败、5xx 或超时，reward 端会轮换实例重试，最长 15 分钟。原逻辑重试 2 次就记 0 分，judge 一挂，整批 reward 就悄悄变成 0。4xx 错误（回答太长、超出 judge 上下文）仍然直接记失败。
9. **每个 reward worker 的 judge 并发从 8 提到 16。** 只影响速度，不影响打分结果。
10. **其它。** checkpoint 间隔从 25 步改为 10 步；加了训练前验证；验证集改为固定的 held-out 文件。

## 10. 47rc 上的进度

到 2026-10-03，还没有可用的 lr=1e-6 结果。

- **lr=5e-6 的旧结果不可用。** 两组 science 用 lr=5e-6 跑到了第 98 步，但训练失稳，结果不可用，第 9 节第 3 条有详细说明。
- **lr=1e-6 只拿到了起点。** 新配置跑到第 14 步时 judge 卡死，修好后重启，又被一次系统级故障卡住：机器上大量 SWE-bench 容器创建虚拟网卡，占住了内核的网络锁，所有新的分布式初始化都挂在那里。

目前可用的只有训练前那一次验证，作为参考：

| 验证集 | agentic 起点 | baseline 起点 |
|---|---:|---:|
| RaR science val500 | 0.673 | 0.681 |
| GPQA-Diamond | 0.046 | 0.051 |

如果新机器上也跑着大量 docker 容器，注意观察 `cat /proc/pressure/memory`，以及 `ps -eo stat | grep -c ^D` 统计的 D 状态进程数。
