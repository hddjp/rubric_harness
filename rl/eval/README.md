# Downstream evaluation

This directory contains only evaluation code for the Medicine and Science
policies trained with harness-generated rubrics.

Install the data dependency:

```bash
pip install datasets huggingface_hub
```

Download and normalize the public datasets:

```bash
cd /data/sty/rubric_harness/rl
python eval/download_eval_data.py --output-dir eval/data
```

This downloads `openai/healthbench` and the `gpqa_diamond` configuration from
`Idavidrein/gpqa` (if accessible). If the original GPQA repository is gated,
the downloader automatically falls back to the public
`extraordinarylab/gpqa-diamond` Parquet mirror. The paper's HealthBench-1k is a specified 1,000-example
subset. If its identifier file is available, pass it for exact selection:

```bash
python eval/download_eval_data.py \
  --healthbench-ids path/to/healthbench_1k_ids.txt \
  --output-dir eval/data
```

Without that file, the downloader uses a reproducible `prompt_id`-sorted
1,000-row fallback and writes the selection method to
`eval/data/healthbench_selection.json`.

Serve the trained checkpoint through an OpenAI-compatible endpoint, then run:

```bash
python eval/eval_model.py \
  --task medicine \
  --data-dir eval/data \
  --results-dir eval/results/medicine \
  --model-base-url http://127.0.0.1:8001/v1 \
  --judge-base-url http://10.176.56.53:8000/v1

python eval/eval_model.py \
  --task science \
  --data-dir eval/data \
  --results-dir eval/results/science \
  --model-base-url http://127.0.0.1:8001/v1
```

Medicine computes HealthBench `overall_score` using physician-written rubric
criteria. Science performs four greedy trials with per-trial choice
permutation and reports mean GPQA-Diamond accuracy.
