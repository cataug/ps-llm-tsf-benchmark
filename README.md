````md
# PS-LLM TSF Benchmark

Pattern- and Semantics-Augmented LLM Benchmark for multivariate time series forecasting.

This repository contains the experimental code, launch scripts, lightweight logs, and result summaries for evaluating LLM-based time series forecasting models with explicit temporal pattern extraction and semantic enrichment.

## Overview

The project studies multivariate time series forecasting with frozen pretrained language-model backbones augmented by:

- multi-scale temporal pattern extraction;
- semantic time-series descriptors;
- pattern-semantic alignment;
- gated fusion;
- LLM-based forecasting heads;
- controlled comparisons against classical and Transformer-based baselines.

The main proposed model is referred to as **PS-LLM**.

## Repository Structure

```text
.
├── run_icdm_psllm.py
├── launch_icdm_parallel.py
├── launch_icdm_parallel_v2.py
├── dynamic_continue_from_job0021.py
├── check_llm_structure.py
├── Datasets.ipynb
├── runs_parallel_logs/
│   ├── jobs_manifest.json
│   ├── launcher_args.json
│   ├── *.cmd
│   ├── *.log
│   └── *.done
├── runs_icdm_psllm_parallel/
│   └── **/result.json
│   └── **/result.csv
│   └── **/history.csv
│   └── **/config.json
└── llm_structure_report.txt
````

## Main Files

| File                                    | Description                                                               |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `run_icdm_psllm.py`                     | Main experiment runner for single models and benchmark plans.             |
| `launch_icdm_parallel.py`               | Initial GPU launcher.                                                     |
| `launch_icdm_parallel_v2.py`            | Extended GPU launcher with skip logic.                                    |
| `dynamic_continue_from_job0021.py`      | Dynamic continuation script from the existing manifest.                   |
| `check_llm_structure.py`                | Utility for checking project files, logs, outputs, and dataset structure. |
| `runs_parallel_logs/jobs_manifest.json` | Full job manifest used by the launcher.                                   |
| `runs_parallel_logs/*.cmd`              | Exact command used for each launched job.                                 |
| `runs_parallel_logs/*.log`              | Lightweight logs for launched jobs.                                       |
| `runs_parallel_logs/*.done`             | Completion flags.                                                         |

## Excluded Files

The following are intentionally excluded from Git:

```text
data_icdm_tsf/
hf_models/
*.pt
*.pth
*.ckpt
*.safetensors
```

Datasets, pretrained HuggingFace models, and model checkpoints are too large for standard Git tracking.

## Datasets

The benchmark uses standard long-horizon forecasting datasets:

```text
ETTh1
ETTh2
ETTm1
ETTm2
Weather
Traffic
Electricity
Exchange
ILI
M4
```

Expected local layout:

```text
data_icdm_tsf/
├── ETTh1/ETTh1.csv
├── ETTh2/ETTh2.csv
├── ETTm1/ETTm1.csv
├── ETTm2/ETTm2.csv
├── Weather/Weather.csv
├── Traffic/Traffic.csv
├── Electricity/Electricity.csv
├── Exchange/Exchange.csv
├── ILI/ILI.csv
└── M4/
```

## Local Model Layout

The experiments assume local HuggingFace models, for example:

```text
/home/tahiti/Malashin_Projects/hf_models/
├── qwen2p5_1p5b/
├── qwen2p5_0p5b_instruct/
├── qwen2p5_1p5b_instruct/
├── opt_350m/
├── opt_1p3b/
├── pythia_410m/
├── pythia_1b/
├── smollm2_1p7b/
├── tinyllama_1p1b_chat/
├── bert_tiny/
├── bert_mini/
├── distilbert_base_uncased/
└── electra_small_discriminator/
```

## Environment

Activate the existing environment:

```bash
cd ~/LLM
source /home/tahiti/Malashin_Projects/.venv_a100/bin/activate
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Run a Single Experiment

Example smoke test:

```bash
python run_icdm_psllm.py \
  --plan single \
  --data_root data_icdm_tsf \
  --hf_root /home/tahiti/Malashin_Projects/hf_models \
  --dataset ETTh1 \
  --model psllm \
  --llm_name qwen2p5_0p5b_instruct \
  --semantic_encoder mlp \
  --mode M \
  --seq_len 96 \
  --pred_len 96 \
  --epochs 2 \
  --batch_size 4 \
  --d_model 64 \
  --n_layers 1 \
  --rank 32 \
  --max_train_batches 20 \
  --max_eval_batches 10
```

## Continue Existing Experiments Dynamically

To continue from the existing launcher manifest and skip completed jobs:

```bash
python dynamic_continue_from_job0021.py \
  --start_job_no 21 \
  --gpus auto \
  --max_per_gpu 4 \
  --min_free_mb 6000 \
  --max_util 98
```

If only one GPU is visible:

```bash
python dynamic_continue_from_job0021.py \
  --start_job_no 21 \
  --gpus 0 \
  --max_per_gpu 2 \
  --min_free_mb 6000 \
  --max_util 98
```

If interrupted, rerun the same command. Completed jobs with `result.json` or `.done` files will be skipped.

## Check Project Structure

```bash
python check_llm_structure.py --max_depth 3 --show_files 120 > llm_structure_report.txt
```

## Git Notes

Only lightweight outputs are tracked:

```text
config.json
history.csv
result.csv
result.json
*.log
*.cmd
*.done
jobs_manifest.json
launcher_args.json
```

Heavy artifacts such as model checkpoints are excluded.

## Main Experiment Groups

The benchmark includes:

| Group                  | Description                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| Main forecasting       | Last, DLinear, NLinear, PatchTST, iTransformer, LLM-direct, PS-LLM                            |
| Ablations              | Full PS-LLM, without pattern, without semantics, without alignment, without gate, without LLM |
| Backbone sensitivity   | OPT, Pythia, Qwen, SmolLM2, TinyLlama                                                         |
| Semantic encoder study | MLP statistics, BERT-tiny, BERT-mini, DistilBERT, ELECTRA                                     |
| Robustness             | Noise perturbation experiments                                                                |
| Few-shot               | Reduced training fractions                                                                    |

## Citation / Paper Context

This code supports experiments for an ICDM-style benchmark on pattern- and semantics-augmented LLMs for multivariate time series forecasting.

```
```
