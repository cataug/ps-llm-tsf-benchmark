# -*- coding: utf-8 -*-
"""
Flexible multi-GPU launcher for run_icdm_psllm.py

Usage examples:

1) Debug:
python launch_icdm_parallel.py --preset debug --gpus 0 --max_per_gpu 1

2) Fill all visible GPUs:
python launch_icdm_parallel.py --preset main96 --gpus auto --max_per_gpu 1

3) Two jobs per GPU:
python launch_icdm_parallel.py --preset ablations --gpus 0,1,2,3 --max_per_gpu 2

4) Run everything in stages:
python launch_icdm_parallel.py --preset all --gpus auto --max_per_gpu 1

The launcher creates:
runs_parallel_logs/
  job_0001_....log
  job_0001_....cmd
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


SCRIPT = "run_icdm_psllm.py"

DEFAULT_DATA_ROOT = "data_icdm_tsf"
DEFAULT_HF_ROOT = "/home/tahiti/Malashin_Projects/hf_models"
DEFAULT_OUT_ROOT = "runs_icdm_psllm_parallel"
LOG_ROOT = Path("runs_parallel_logs")


MAIN_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Weather", "Traffic", "Electricity", "Exchange", "ILI"
]

SMALL_DATASETS = ["ETTh1", "Weather", "Exchange"]
ABLATION_DATASETS = ["ETTh1", "Weather", "Electricity", "Exchange", "ILI"]
BACKBONE_DATASETS = ["ETTh1", "Weather", "Exchange", "ILI"]
NOISE_DATASETS = ["ETTh1", "Weather", "Traffic", "Exchange"]
FEWSHOT_DATASETS = ["ETTh1", "Weather", "Exchange", "ILI"]

DEFAULT_BACKBONES = [
    "opt_350m",
    "opt_1p3b",
    "pythia_410m",
    "pythia_1b",
    "qwen2p5_0p5b_instruct",
    "qwen2p5_1p5b",
    "qwen2p5_1p5b_instruct",
    "smollm2_1p7b",
    "tinyllama_1p1b_chat",
]

BIG_BACKBONES = [
    "qwen2p5_3b",
    "phi3p5_mini_instruct",
]

SEMANTIC_ENCODERS = [
    "mlp",
    "bert_tiny",
    "bert_mini",
    "distilbert_base_uncased",
    "electra_small_discriminator",
]


def shlex_join(parts):
    import shlex
    return " ".join(shlex.quote(str(x)) for x in parts)


def detect_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        gpus = [x.strip() for x in out.splitlines() if x.strip()]
        return gpus
    except Exception:
        return ["0"]


def base_cmd(args):
    return [
        sys.executable,
        SCRIPT,
        "--data_root", args.data_root,
        "--hf_root", args.hf_root,
        "--mode", args.mode,
        "--seq_len", str(args.seq_len),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--d_model", str(args.d_model),
        "--n_layers", str(args.n_layers),
        "--n_heads", str(args.n_heads),
        "--rank", str(args.rank),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--amp", str(args.amp),
        "--num_workers", str(args.num_workers),
    ]


def cmd_single(
    args,
    out_dir,
    dataset,
    model,
    pred_len,
    llm_name=None,
    semantic_encoder=None,
    extra=None,
):
    cmd = base_cmd(args)
    cmd += [
        "--plan", "single",
        "--out_dir", out_dir,
        "--dataset", dataset,
        "--model", model,
        "--pred_len", str(pred_len),
    ]

    if llm_name is not None:
        cmd += ["--llm_name", llm_name]

    if semantic_encoder is not None:
        cmd += ["--semantic_encoder", semantic_encoder]

    if extra:
        cmd += extra

    return cmd


def add_debug_jobs(args, jobs):
    for dataset in ["ETTh1", "Weather", "Exchange"]:
        for model in ["last", "dlinear", "patchtst", "psllm"]:
            jobs.append({
                "name": f"debug_{dataset}_{model}_H96",
                "cmd": cmd_single(
                    args,
                    out_dir=f"{args.out_root}/debug",
                    dataset=dataset,
                    model=model,
                    pred_len=96,
                    llm_name="qwen2p5_0p5b_instruct",
                    semantic_encoder="mlp",
                    extra=[
                        "--max_train_batches", "20",
                        "--max_eval_batches", "10",
                        "--epochs", "2",
                        "--batch_size", str(min(args.batch_size, 4)),
                        "--d_model", "64",
                        "--n_layers", "1",
                        "--rank", "32",
                    ],
                )
            })


def add_main96_jobs(args, jobs):
    models = ["last", "dlinear", "nlinear", "patchtst", "itransformer", "llm_direct", "psllm"]
    for dataset in MAIN_DATASETS:
        for model in models:
            jobs.append({
                "name": f"main96_{dataset}_{model}",
                "cmd": cmd_single(
                    args,
                    out_dir=f"{args.out_root}/main96",
                    dataset=dataset,
                    model=model,
                    pred_len=96 if dataset != "ILI" else 24,
                    llm_name=args.main_llm,
                    semantic_encoder="mlp",
                )
            })


def add_main_full_jobs(args, jobs):
    models = ["last", "dlinear", "nlinear", "patchtst", "itransformer", "llm_direct", "psllm"]
    for dataset in MAIN_DATASETS:
        horizons = [24, 36, 48, 60] if dataset == "ILI" else [96, 192, 336, 720]
        for h in horizons:
            for model in models:
                jobs.append({
                    "name": f"mainfull_{dataset}_{model}_H{h}",
                    "cmd": cmd_single(
                        args,
                        out_dir=f"{args.out_root}/main_full",
                        dataset=dataset,
                        model=model,
                        pred_len=h,
                        llm_name=args.main_llm,
                        semantic_encoder="mlp",
                    )
                })


def add_ablation_jobs(args, jobs):
    # use built-in ablation plan per dataset+horizon, easier and less error-prone
    for dataset in ABLATION_DATASETS:
        horizons = [24, 48] if dataset == "ILI" else [96, 336]
        for h in horizons:
            jobs.append({
                "name": f"ablation_{dataset}_H{h}",
                "cmd": base_cmd(args) + [
                    "--plan", "ablations",
                    "--out_dir", f"{args.out_root}/ablation",
                    "--datasets", dataset,
                    "--horizons", str(h),
                    "--llm_name", args.main_llm,
                    "--semantic_encoder", "mlp",
                ]
            })


def add_backbone_jobs(args, jobs):
    for dataset in BACKBONE_DATASETS:
        for backbone in DEFAULT_BACKBONES:
            jobs.append({
                "name": f"backbone_{dataset}_{backbone}",
                "cmd": cmd_single(
                    args,
                    out_dir=f"{args.out_root}/backbones",
                    dataset=dataset,
                    model="psllm",
                    pred_len=96 if dataset != "ILI" else 24,
                    llm_name=backbone,
                    semantic_encoder="mlp",
                    extra=[
                        "--epochs", str(args.backbone_epochs),
                        "--batch_size", str(args.backbone_batch_size),
                    ],
                )
            })


def add_big_backbone_jobs(args, jobs):
    for dataset in ["ETTh1", "Weather", "Exchange"]:
        for backbone in BIG_BACKBONES:
            jobs.append({
                "name": f"bigbackbone_{dataset}_{backbone}",
                "cmd": cmd_single(
                    args,
                    out_dir=f"{args.out_root}/backbones_big",
                    dataset=dataset,
                    model="psllm",
                    pred_len=96,
                    llm_name=backbone,
                    semantic_encoder="mlp",
                    extra=[
                        "--epochs", str(args.big_epochs),
                        "--batch_size", str(args.big_batch_size),
                        "--d_model", "128",
                        "--rank", "64",
                    ],
                )
            })


def add_semantic_jobs(args, jobs):
    for dataset in BACKBONE_DATASETS:
        for enc in SEMANTIC_ENCODERS:
            jobs.append({
                "name": f"semantic_{dataset}_{enc}",
                "cmd": cmd_single(
                    args,
                    out_dir=f"{args.out_root}/semantic",
                    dataset=dataset,
                    model="psllm",
                    pred_len=96 if dataset != "ILI" else 24,
                    llm_name=args.main_llm,
                    semantic_encoder=enc,
                    extra=[
                        "--epochs", str(args.semantic_epochs),
                        "--batch_size", str(args.semantic_batch_size),
                    ],
                )
            })


def add_noise_jobs(args, jobs):
    models = ["patchtst", "itransformer", "llm_direct", "psllm"]
    for dataset in NOISE_DATASETS:
        for noise in [0.0, 0.05, 0.1, 0.2]:
            for model in models:
                jobs.append({
                    "name": f"noise_{dataset}_{model}_N{noise}",
                    "cmd": cmd_single(
                        args,
                        out_dir=f"{args.out_root}/noise",
                        dataset=dataset,
                        model=model,
                        pred_len=96,
                        llm_name=args.main_llm,
                        semantic_encoder="mlp",
                        extra=[
                            "--noise_std", str(noise),
                            "--epochs", str(args.robust_epochs),
                        ],
                    )
                })


def add_fewshot_jobs(args, jobs):
    models = ["dlinear", "patchtst", "itransformer", "llm_direct", "psllm"]
    for dataset in FEWSHOT_DATASETS:
        for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
            for model in models:
                jobs.append({
                    "name": f"fewshot_{dataset}_{model}_F{frac}",
                    "cmd": cmd_single(
                        args,
                        out_dir=f"{args.out_root}/fewshot",
                        dataset=dataset,
                        model=model,
                        pred_len=96 if dataset != "ILI" else 24,
                        llm_name=args.main_llm,
                        semantic_encoder="mlp",
                        extra=[
                            "--fewshot_ratio", str(frac),
                            "--epochs", str(args.robust_epochs),
                        ],
                    )
                })


def add_m4_jobs(args, jobs):
    models = ["last", "dlinear", "nlinear", "patchtst", "llm_direct", "psllm"]
    for freq in ["Hourly", "Daily", "Weekly", "Monthly"]:
        for model in models:
            jobs.append({
                "name": f"m4_{freq}_{model}",
                "cmd": base_cmd(args) + [
                    "--plan", "single",
                    "--task", "m4",
                    "--dataset", "M4",
                    "--mode", "S",
                    "--out_dir", f"{args.out_root}/m4",
                    "--m4_freq", freq,
                    "--pred_len", "-1",
                    "--model", model,
                    "--llm_name", args.debug_llm,
                    "--semantic_encoder", "mlp",
                    "--epochs", str(args.robust_epochs),
                    "--batch_size", str(args.m4_batch_size),
                    "--d_model", "64",
                    "--rank", "32",
                ]
            })


def build_jobs(args):
    jobs = []

    presets = [args.preset] if args.preset != "all" else [
        "main96",
        "ablations",
        "backbones",
        "semantic",
        "noise",
        "fewshot",
    ]

    for preset in presets:
        if preset == "debug":
            add_debug_jobs(args, jobs)
        elif preset == "main96":
            add_main96_jobs(args, jobs)
        elif preset == "main_full":
            add_main_full_jobs(args, jobs)
        elif preset == "ablations":
            add_ablation_jobs(args, jobs)
        elif preset == "backbones":
            add_backbone_jobs(args, jobs)
        elif preset == "big_backbones":
            add_big_backbone_jobs(args, jobs)
        elif preset == "semantic":
            add_semantic_jobs(args, jobs)
        elif preset == "noise":
            add_noise_jobs(args, jobs)
        elif preset == "fewshot":
            add_fewshot_jobs(args, jobs)
        elif preset == "m4":
            add_m4_jobs(args, jobs)
        else:
            raise ValueError(f"Unknown preset: {preset}")

    if args.limit > 0:
        jobs = jobs[:args.limit]

    return jobs


def is_done(job, out_root):
    # Conservative skip: checks if any matching result.json exists under out_root.
    # Since exact run_dir naming is long, use log-level skip only when command file marked success.
    done_flag = LOG_ROOT / f"{job['name']}.done"
    return done_flag.exists()


def launch_job(job, gpu, job_id, args):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    safe_name = job["name"].replace("/", "_").replace(":", "_")
    log_path = LOG_ROOT / f"job_{job_id:04d}_{safe_name}_gpu{gpu}.log"
    cmd_path = LOG_ROOT / f"job_{job_id:04d}_{safe_name}_gpu{gpu}.cmd"
    done_path = LOG_ROOT / f"{job['name']}.done"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = job["cmd"]

    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(shlex_join(cmd) + "\n")

    log_f = open(log_path, "w", encoding="utf-8")

    header = (
        f"JOB_ID={job_id}\n"
        f"JOB_NAME={job['name']}\n"
        f"GPU={gpu}\n"
        f"START={datetime.now().isoformat()}\n"
        f"CMD={shlex_join(cmd)}\n"
        f"{'=' * 120}\n"
    )
    log_f.write(header)
    log_f.flush()

    p = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.getcwd(),
        text=True,
    )

    return {
        "process": p,
        "gpu": gpu,
        "job": job,
        "job_id": job_id,
        "log_file": log_f,
        "log_path": log_path,
        "done_path": done_path,
        "start": time.time(),
    }


def scheduler(args):
    if args.gpus == "auto":
        gpus = detect_gpus()
    else:
        gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]

    if not gpus:
        raise RuntimeError("No GPUs selected")

    jobs = build_jobs(args)

    if args.shuffle:
        import random
        random.shuffle(jobs)

    print("=" * 120)
    print("PARALLEL ICDM LAUNCHER")
    print("GPUs:", gpus)
    print("max_per_gpu:", args.max_per_gpu)
    print("preset:", args.preset)
    print("jobs:", len(jobs))
    print("out_root:", args.out_root)
    print("log_root:", LOG_ROOT.resolve())
    print("=" * 120)

    with open(LOG_ROOT / "jobs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    queue = []
    skipped = 0
    for j in jobs:
        if args.skip_done and is_done(j, args.out_root):
            skipped += 1
        else:
            queue.append(j)

    print(f"Queued jobs: {len(queue)} | skipped existing done flags: {skipped}")

    active = []
    next_job_id = 1
    completed = 0
    failed = 0

    gpu_slots = {gpu: 0 for gpu in gpus}

    while queue or active:
        # update active processes
        still_active = []
        for item in active:
            ret = item["process"].poll()
            if ret is None:
                still_active.append(item)
                continue

            item["log_file"].write("\n" + "=" * 120 + "\n")
            item["log_file"].write(f"END={datetime.now().isoformat()}\n")
            item["log_file"].write(f"RETURN_CODE={ret}\n")
            item["log_file"].close()

            gpu_slots[item["gpu"]] -= 1

            if ret == 0:
                completed += 1
                item["done_path"].write_text(
                    f"done {datetime.now().isoformat()}\nlog={item['log_path']}\n",
                    encoding="utf-8",
                )
                print(f"[DONE] job {item['job_id']:04d} {item['job']['name']} gpu={item['gpu']}")
            else:
                failed += 1
                print(f"[FAIL] job {item['job_id']:04d} {item['job']['name']} gpu={item['gpu']} log={item['log_path']}")

                if args.stop_on_fail:
                    print("stop_on_fail=True, terminating remaining jobs...")
                    for a in still_active:
                        a["process"].terminate()
                    raise SystemExit(1)

        active = still_active

        # launch new jobs into free slots
        launched_any = False
        for gpu in gpus:
            while queue and gpu_slots[gpu] < args.max_per_gpu:
                job = queue.pop(0)
                item = launch_job(job, gpu, next_job_id, args)
                active.append(item)
                gpu_slots[gpu] += 1

                print(
                    f"[LAUNCH] job {next_job_id:04d} gpu={gpu} "
                    f"active={len(active)} queue={len(queue)} name={job['name']} "
                    f"log={item['log_path']}"
                )

                next_job_id += 1
                launched_any = True

                if args.launch_delay > 0:
                    time.sleep(args.launch_delay)

        if not launched_any:
            time.sleep(args.poll_interval)

    print("=" * 120)
    print("ALL JOBS FINISHED")
    print("completed:", completed)
    print("failed:", failed)
    print("logs:", LOG_ROOT.resolve())
    print("=" * 120)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--preset",
        type=str,
        default="debug",
        choices=[
            "debug",
            "main96",
            "main_full",
            "ablations",
            "backbones",
            "big_backbones",
            "semantic",
            "noise",
            "fewshot",
            "m4",
            "all",
        ],
    )

    p.add_argument("--gpus", type=str, default="auto")
    p.add_argument("--max_per_gpu", type=int, default=1)
    p.add_argument("--skip_done", type=int, default=1)
    p.add_argument("--stop_on_fail", type=int, default=0)
    p.add_argument("--shuffle", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1)

    p.add_argument("--poll_interval", type=float, default=10.0)
    p.add_argument("--launch_delay", type=float, default=2.0)

    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--hf_root", type=str, default=DEFAULT_HF_ROOT)
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)

    p.add_argument("--mode", type=str, default="M")
    p.add_argument("--seq_len", type=int, default=96)

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--backbone_batch_size", type=int, default=2)
    p.add_argument("--semantic_batch_size", type=int, default=2)
    p.add_argument("--big_batch_size", type=int, default=1)
    p.add_argument("--m4_batch_size", type=int, default=8)

    p.add_argument("--backbone_epochs", type=int, default=4)
    p.add_argument("--semantic_epochs", type=int, default=4)
    p.add_argument("--big_epochs", type=int, default=3)
    p.add_argument("--robust_epochs", type=int, default=4)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--rank", type=int, default=64)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--main_llm", type=str, default="qwen2p5_1p5b")
    p.add_argument("--debug_llm", type=str, default="qwen2p5_0p5b_instruct")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    scheduler(args)