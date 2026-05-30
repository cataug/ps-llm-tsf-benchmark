# -*- coding: utf-8 -*-
"""
Flexible multi-GPU scheduler for ICDM PS-LLM experiments.

This launcher DOES NOT change computation logic.
It only schedules independent calls to run_icdm_psllm.py.

Features:
  - automatic GPU detection via nvidia-smi;
  - configurable number of concurrent jobs per GPU;
  - exact skip of already completed jobs using result.json;
  - .done flags for completed launcher jobs;
  - logs and command files for every job;
  - optional retry once after failure;
  - presets:
      debug
      main96
      main_full
      ablations
      backbones
      big_backbones
      semantic
      noise
      fewshot
      m4
      all

Expected:
  ~/LLM/run_icdm_psllm.py
  ~/LLM/data_icdm_tsf/
  /home/tahiti/Malashin_Projects/hf_models/
"""

import os
import sys
import json
import time
import shlex
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


SCRIPT = "run_icdm_psllm.py"

DEFAULT_DATA_ROOT = "data_icdm_tsf"
DEFAULT_HF_ROOT = "/home/tahiti/Malashin_Projects/hf_models"
DEFAULT_OUT_ROOT = "runs_icdm_psllm_parallel"
LOG_ROOT = Path("runs_parallel_logs")


MAIN_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Weather", "Traffic", "Electricity", "Exchange", "ILI"
]

MAIN_MODELS = [
    "last",
    "dlinear",
    "nlinear",
    "patchtst",
    "itransformer",
    "llm_direct",
    "psllm",
]

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

M4_FREQS = ["Hourly", "Daily", "Weekly", "Monthly"]

M4_HORIZONS = {
    "Hourly": 48,
    "Daily": 14,
    "Weekly": 13,
    "Monthly": 18,
    "Quarterly": 8,
    "Yearly": 6,
}


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def detect_gpus() -> List[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        gpus = [x.strip() for x in out.splitlines() if x.strip()]
        return gpus
    except Exception:
        return ["0"]


def query_gpu_free_memory() -> Dict[str, int]:
    """
    Returns free GPU memory in MiB.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        result = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result[parts[0]] = int(float(parts[1]))
        return result
    except Exception:
        return {}


def pred_for_main(dataset: str, horizon: int) -> int:
    if dataset == "ILI":
        return 24 if horizon == 96 else horizon
    return horizon


def horizons_for_dataset(dataset: str, full: bool) -> List[int]:
    if dataset == "ILI":
        return [24, 36, 48, 60] if full else [24]
    return [96, 192, 336, 720] if full else [96]


def ablation_horizons(dataset: str) -> List[int]:
    if dataset == "ILI":
        return [24, 48]
    return [96, 336]


def sanitize_name(s: str) -> str:
    return (
        s.replace("/", "_")
         .replace(":", "_")
         .replace(" ", "_")
         .replace(".", "p")
    )


def exact_run_name(job: Dict[str, Any]) -> str:
    """
    Mirrors run_icdm_psllm.py naming:
    f"{task}_{dataset if task == 'main' else 'M4-' + freq}"
    f"_model-{model}"
    f"_llm-{llm_name}"
    f"_sem-{semantic_encoder}"
    f"_mode-{mode}"
    f"_L{seq_len}_H{pred_len}"
    f"_pat{use_pattern}_sem{use_semantic}_align{use_align}_gate{use_gate}_llm{use_llm}"
    f"_fs{fewshot_ratio}_noise{noise_std}_seed{seed}"
    """
    task = job["task"]
    ds_part = job["dataset"] if task == "main" else f"M4-{job['m4_freq']}"
    return (
        f"{task}_{ds_part}"
        f"_model-{job['model']}"
        f"_llm-{job['llm_name']}"
        f"_sem-{job['semantic_encoder']}"
        f"_mode-{job['mode']}"
        f"_L{job['seq_len']}_H{job['pred_len']}"
        f"_pat{job['use_pattern']}_sem{job['use_semantic']}_align{job['use_align']}_gate{job['use_gate']}_llm{job['use_llm']}"
        f"_fs{job['fewshot_ratio']}_noise{job['noise_std']}_seed{job['seed']}"
    )


def expected_result_path(job: Dict[str, Any]) -> Path:
    return Path(job["out_dir"]) / exact_run_name(job) / "result.json"


def done_flag_path(job: Dict[str, Any]) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    safe = sanitize_name(job["name"])
    return LOG_ROOT / f"{safe}.done"


def is_job_done(job: Dict[str, Any]) -> bool:
    result_path = expected_result_path(job)
    done_path = done_flag_path(job)

    if result_path.exists() and result_path.stat().st_size > 0:
        return True

    if done_path.exists():
        return True

    return False


def base_job(args) -> Dict[str, Any]:
    return {
        "data_root": args.data_root,
        "hf_root": args.hf_root,
        "mode": args.mode,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "rank": args.rank,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": "auto",
        "semantic_encoder": "mlp",
        "llm_name": args.main_llm,
        "fewshot_ratio": 1.0,
        "noise_std": 0.0,
        "use_pattern": 1,
        "use_semantic": 1,
        "use_align": 1,
        "use_gate": 1,
        "use_llm": 1,
        "m4_freq": "Hourly",
        "m4_max_series": -1,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
    }


def make_job(
    args,
    *,
    name: str,
    out_dir: str,
    task: str = "main",
    dataset: str = "ETTh1",
    model: str = "psllm",
    pred_len: int = 96,
    llm_name: Optional[str] = None,
    semantic_encoder: Optional[str] = None,
    mode: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    d_model: Optional[int] = None,
    n_layers: Optional[int] = None,
    rank: Optional[int] = None,
    fewshot_ratio: Optional[float] = None,
    noise_std: Optional[float] = None,
    use_pattern: Optional[int] = None,
    use_semantic: Optional[int] = None,
    use_align: Optional[int] = None,
    use_gate: Optional[int] = None,
    use_llm: Optional[int] = None,
    m4_freq: str = "Hourly",
    m4_max_series: int = -1,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    j = base_job(args)
    j.update({
        "name": name,
        "out_dir": out_dir,
        "task": task,
        "dataset": dataset,
        "model": model,
        "pred_len": pred_len,
        "m4_freq": m4_freq,
        "m4_max_series": m4_max_series,
    })

    if llm_name is not None:
        j["llm_name"] = llm_name
    if semantic_encoder is not None:
        j["semantic_encoder"] = semantic_encoder
    if mode is not None:
        j["mode"] = mode
    if batch_size is not None:
        j["batch_size"] = batch_size
    if epochs is not None:
        j["epochs"] = epochs
    if d_model is not None:
        j["d_model"] = d_model
    if n_layers is not None:
        j["n_layers"] = n_layers
    if rank is not None:
        j["rank"] = rank
    if fewshot_ratio is not None:
        j["fewshot_ratio"] = fewshot_ratio
    if noise_std is not None:
        j["noise_std"] = noise_std
    if use_pattern is not None:
        j["use_pattern"] = use_pattern
    if use_semantic is not None:
        j["use_semantic"] = use_semantic
    if use_align is not None:
        j["use_align"] = use_align
    if use_gate is not None:
        j["use_gate"] = use_gate
    if use_llm is not None:
        j["use_llm"] = use_llm

    if extra_args is None:
        extra_args = []
    j["extra_args"] = extra_args

    # Baselines do not use LLM modules.
    if model not in ["psllm", "llm_direct"]:
        j["use_pattern"] = 0
        j["use_semantic"] = 0
        j["use_align"] = 0
        j["use_gate"] = 0
        j["use_llm"] = 0

    # LLM-direct is explicitly raw projection -> frozen LLM -> head.
    if model == "llm_direct":
        j["use_pattern"] = 0
        j["use_semantic"] = 0
        j["use_align"] = 0
        j["use_gate"] = 0
        j["use_llm"] = 1

    return j


def job_to_cmd(job: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        SCRIPT,
        "--plan", "single",
        "--data_root", job["data_root"],
        "--hf_root", job["hf_root"],
        "--out_dir", job["out_dir"],
        "--task", job["task"],
        "--dataset", job["dataset"],
        "--mode", job["mode"],
        "--model", job["model"],
        "--llm_name", job["llm_name"],
        "--semantic_encoder", job["semantic_encoder"],
        "--seq_len", str(job["seq_len"]),
        "--pred_len", str(job["pred_len"]),
        "--batch_size", str(job["batch_size"]),
        "--epochs", str(job["epochs"]),
        "--patience", str(job["patience"]),
        "--d_model", str(job["d_model"]),
        "--n_layers", str(job["n_layers"]),
        "--n_heads", str(job["n_heads"]),
        "--rank", str(job["rank"]),
        "--lr", str(job["lr"]),
        "--weight_decay", str(job["weight_decay"]),
        "--amp", str(job["amp"]),
        "--num_workers", str(job["num_workers"]),
        "--seed", str(job["seed"]),
        "--device", job["device"],
        "--fewshot_ratio", str(job["fewshot_ratio"]),
        "--noise_std", str(job["noise_std"]),
        "--use_pattern", str(job["use_pattern"]),
        "--use_semantic", str(job["use_semantic"]),
        "--use_align", str(job["use_align"]),
        "--use_gate", str(job["use_gate"]),
        "--use_llm", str(job["use_llm"]),
        "--max_train_batches", str(job["max_train_batches"]),
        "--max_eval_batches", str(job["max_eval_batches"]),
    ]

    if job["task"] == "m4":
        cmd += [
            "--m4_freq", job["m4_freq"],
            "--m4_max_series", str(job["m4_max_series"]),
        ]

    cmd += job.get("extra_args", [])
    return cmd


def add_debug(args, jobs: List[Dict[str, Any]]):
    for dataset in ["ETTh1", "Weather", "Exchange"]:
        h = 96
        for model in ["last", "dlinear", "patchtst", "psllm"]:
            jobs.append(make_job(
                args,
                name=f"debug_{dataset}_{model}_H{h}",
                out_dir=f"{args.out_root}/debug",
                dataset=dataset,
                model=model,
                pred_len=h,
                llm_name=args.debug_llm,
                batch_size=min(args.batch_size, 4),
                epochs=2,
                d_model=64,
                n_layers=1,
                rank=32,
                extra_args=[
                    "--max_train_batches", "20",
                    "--max_eval_batches", "10",
                ],
            ))


def add_main96(args, jobs: List[Dict[str, Any]]):
    for dataset in MAIN_DATASETS:
        h = 24 if dataset == "ILI" else 96
        for model in MAIN_MODELS:
            jobs.append(make_job(
                args,
                name=f"main96_{dataset}_{model}_H{h}",
                out_dir=f"{args.out_root}/main96",
                dataset=dataset,
                model=model,
                pred_len=h,
                llm_name=args.main_llm,
                semantic_encoder="mlp",
            ))


def add_main_full(args, jobs: List[Dict[str, Any]]):
    for dataset in MAIN_DATASETS:
        horizons = horizons_for_dataset(dataset, full=True)
        for h in horizons:
            for model in MAIN_MODELS:
                jobs.append(make_job(
                    args,
                    name=f"mainfull_{dataset}_{model}_H{h}",
                    out_dir=f"{args.out_root}/main_full",
                    dataset=dataset,
                    model=model,
                    pred_len=h,
                    llm_name=args.main_llm,
                    semantic_encoder="mlp",
                ))


def add_ablations(args, jobs: List[Dict[str, Any]]):
    variants = [
        ("full",       "psllm",      1, 1, 1, 1, 1),
        ("wo_pattern", "psllm",      0, 1, 0, 0, 1),
        ("wo_semantic","psllm",      1, 0, 0, 0, 1),
        ("wo_align",   "psllm",      1, 1, 0, 1, 1),
        ("wo_gate",    "psllm",      1, 1, 1, 0, 1),
        ("wo_llm",     "psllm",      1, 1, 1, 1, 0),
        ("llm_direct", "llm_direct", 0, 0, 0, 0, 1),
    ]

    for dataset in ABLATION_DATASETS:
        for h in ablation_horizons(dataset):
            for vname, model, pat, sem, align, gate, llm in variants:
                jobs.append(make_job(
                    args,
                    name=f"ablation_{dataset}_{vname}_H{h}",
                    out_dir=f"{args.out_root}/ablation/{vname}",
                    dataset=dataset,
                    model=model,
                    pred_len=h,
                    llm_name=args.main_llm,
                    semantic_encoder="mlp",
                    use_pattern=pat,
                    use_semantic=sem,
                    use_align=align,
                    use_gate=gate,
                    use_llm=llm,
                ))


def add_backbones(args, jobs: List[Dict[str, Any]]):
    backbones = args.backbones.split(",") if args.backbones else DEFAULT_BACKBONES
    backbones = [b.strip() for b in backbones if b.strip()]

    for dataset in BACKBONE_DATASETS:
        h = 24 if dataset == "ILI" else 96
        for backbone in backbones:
            jobs.append(make_job(
                args,
                name=f"backbone_{dataset}_{sanitize_name(backbone)}_H{h}",
                out_dir=f"{args.out_root}/backbones",
                dataset=dataset,
                model="psllm",
                pred_len=h,
                llm_name=backbone,
                semantic_encoder="mlp",
                batch_size=args.backbone_batch_size,
                epochs=args.backbone_epochs,
            ))


def add_big_backbones(args, jobs: List[Dict[str, Any]]):
    big = args.big_backbones.split(",") if args.big_backbones else BIG_BACKBONES
    big = [b.strip() for b in big if b.strip()]

    for dataset in ["ETTh1", "Weather", "Exchange"]:
        for backbone in big:
            jobs.append(make_job(
                args,
                name=f"bigbackbone_{dataset}_{sanitize_name(backbone)}",
                out_dir=f"{args.out_root}/backbones_big",
                dataset=dataset,
                model="psllm",
                pred_len=96,
                llm_name=backbone,
                semantic_encoder="mlp",
                batch_size=args.big_batch_size,
                epochs=args.big_epochs,
                d_model=128,
                rank=64,
            ))


def add_semantic(args, jobs: List[Dict[str, Any]]):
    encoders = args.semantic_encoders.split(",") if args.semantic_encoders else SEMANTIC_ENCODERS
    encoders = [e.strip() for e in encoders if e.strip()]

    for dataset in BACKBONE_DATASETS:
        h = 24 if dataset == "ILI" else 96
        for enc in encoders:
            jobs.append(make_job(
                args,
                name=f"semantic_{dataset}_{enc}_H{h}",
                out_dir=f"{args.out_root}/semantic",
                dataset=dataset,
                model="psllm",
                pred_len=h,
                llm_name=args.main_llm,
                semantic_encoder=enc,
                batch_size=args.semantic_batch_size,
                epochs=args.semantic_epochs,
            ))


def add_noise(args, jobs: List[Dict[str, Any]]):
    models = ["patchtst", "itransformer", "llm_direct", "psllm"]
    noise_levels = [0.0, 0.05, 0.1, 0.2]

    for dataset in NOISE_DATASETS:
        for noise in noise_levels:
            for model in models:
                jobs.append(make_job(
                    args,
                    name=f"noise_{dataset}_{model}_N{noise}",
                    out_dir=f"{args.out_root}/noise",
                    dataset=dataset,
                    model=model,
                    pred_len=96,
                    llm_name=args.main_llm,
                    semantic_encoder="mlp",
                    noise_std=noise,
                    epochs=args.robust_epochs,
                ))


def add_fewshot(args, jobs: List[Dict[str, Any]]):
    models = ["dlinear", "patchtst", "itransformer", "llm_direct", "psllm"]
    fractions = [0.05, 0.1, 0.25, 0.5, 1.0]

    for dataset in FEWSHOT_DATASETS:
        h = 24 if dataset == "ILI" else 96
        for frac in fractions:
            for model in models:
                jobs.append(make_job(
                    args,
                    name=f"fewshot_{dataset}_{model}_F{frac}",
                    out_dir=f"{args.out_root}/fewshot",
                    dataset=dataset,
                    model=model,
                    pred_len=h,
                    llm_name=args.main_llm,
                    semantic_encoder="mlp",
                    fewshot_ratio=frac,
                    epochs=args.robust_epochs,
                ))


def add_m4(args, jobs: List[Dict[str, Any]]):
    models = ["last", "dlinear", "nlinear", "patchtst", "llm_direct", "psllm"]

    for freq in M4_FREQS:
        h = M4_HORIZONS[freq]
        for model in models:
            jobs.append(make_job(
                args,
                name=f"m4_{freq}_{model}_H{h}",
                out_dir=f"{args.out_root}/m4",
                task="m4",
                dataset="M4",
                mode="S",
                model=model,
                pred_len=h,
                llm_name=args.debug_llm,
                semantic_encoder="mlp",
                batch_size=args.m4_batch_size,
                epochs=args.robust_epochs,
                d_model=64,
                n_layers=1,
                rank=32,
                m4_freq=freq,
                m4_max_series=args.m4_max_series,
            ))


def build_jobs(args) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    if args.preset == "all":
        presets = [
            "main96",
            "ablations",
            "backbones",
            "semantic",
            "noise",
            "fewshot",
        ]
    else:
        presets = [args.preset]

    for preset in presets:
        if preset == "debug":
            add_debug(args, jobs)
        elif preset == "main96":
            add_main96(args, jobs)
        elif preset == "main_full":
            add_main_full(args, jobs)
        elif preset == "ablations":
            add_ablations(args, jobs)
        elif preset == "backbones":
            add_backbones(args, jobs)
        elif preset == "big_backbones":
            add_big_backbones(args, jobs)
        elif preset == "semantic":
            add_semantic(args, jobs)
        elif preset == "noise":
            add_noise(args, jobs)
        elif preset == "fewshot":
            add_fewshot(args, jobs)
        elif preset == "m4":
            add_m4(args, jobs)
        else:
            raise ValueError(f"Unknown preset: {preset}")

    if args.limit > 0:
        jobs = jobs[:args.limit]

    if args.shuffle:
        import random
        random.shuffle(jobs)

    return jobs


def write_manifest(jobs: List[Dict[str, Any]], args):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, job in enumerate(jobs, 1):
        manifest.append({
            "idx": i,
            "name": job["name"],
            "expected_result": str(expected_result_path(job)),
            "cmd": shell_join(job_to_cmd(job)),
            "done": is_job_done(job),
        })

    with open(LOG_ROOT / "jobs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(LOG_ROOT / "launcher_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


def launch_process(job: Dict[str, Any], gpu: str, job_id: int, args):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_name(job["name"])
    log_path = LOG_ROOT / f"job_{job_id:05d}_{safe_name}_gpu{gpu}.log"
    cmd_path = LOG_ROOT / f"job_{job_id:05d}_{safe_name}_gpu{gpu}.cmd"

    cmd = job_to_cmd(job)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(shell_join(cmd) + "\n")

    log_f = open(log_path, "w", encoding="utf-8")

    header = {
        "job_id": job_id,
        "job_name": job["name"],
        "gpu": gpu,
        "start": datetime.now().isoformat(),
        "expected_result": str(expected_result_path(job)),
        "cmd": shell_join(cmd),
    }
    log_f.write(json.dumps(header, indent=2) + "\n")
    log_f.write("=" * 120 + "\n")
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
        "job": job,
        "job_id": job_id,
        "gpu": gpu,
        "log_path": log_path,
        "cmd_path": cmd_path,
        "log_file": log_f,
        "start_time": time.time(),
        "retries": 0,
    }


def mark_done(job: Dict[str, Any], log_path: Path):
    p = done_flag_path(job)
    p.write_text(
        f"done={datetime.now().isoformat()}\n"
        f"result={expected_result_path(job)}\n"
        f"log={log_path}\n",
        encoding="utf-8",
    )


def scheduler(args):
    if args.gpus == "auto":
        gpus = detect_gpus()
    else:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if not gpus:
        raise RuntimeError("No GPUs selected")

    jobs_all = build_jobs(args)
    write_manifest(jobs_all, args)

    queue = []
    skipped = 0

    for job in jobs_all:
        if args.skip_existing and is_job_done(job):
            skipped += 1
        else:
            queue.append(job)

    gpu_slots = {gpu: 0 for gpu in gpus}
    active = []

    next_job_id = 1
    completed = 0
    failed = 0
    relaunched = 0

    print("=" * 120, flush=True)
    print("ICDM PS-LLM GPU SCHEDULER", flush=True)
    print("time:", datetime.now().isoformat(), flush=True)
    print("preset:", args.preset, flush=True)
    print("cwd:", os.getcwd(), flush=True)
    print("script:", SCRIPT, flush=True)
    print("gpus:", gpus, flush=True)
    print("max_per_gpu:", args.max_per_gpu, flush=True)
    print("total jobs:", len(jobs_all), flush=True)
    print("skipped existing:", skipped, flush=True)
    print("queued:", len(queue), flush=True)
    print("out_root:", args.out_root, flush=True)
    print("log_root:", LOG_ROOT.resolve(), flush=True)
    print("=" * 120, flush=True)

    while queue or active:
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

            job = item["job"]
            result_exists = expected_result_path(job).exists()

            if ret == 0 and result_exists:
                completed += 1
                mark_done(job, item["log_path"])
                print(
                    f"[DONE] job={item['job_id']:05d} gpu={item['gpu']} "
                    f"name={job['name']}",
                    flush=True,
                )

            elif ret == 0 and not result_exists:
                failed += 1
                print(
                    f"[FAIL:NO_RESULT] job={item['job_id']:05d} gpu={item['gpu']} "
                    f"name={job['name']} log={item['log_path']}",
                    flush=True,
                )

            else:
                can_retry = item["retries"] < args.retries
                if can_retry:
                    item["retries"] += 1
                    relaunched += 1
                    queue.append(job)
                    print(
                        f"[RETRY] job={item['job_id']:05d} gpu={item['gpu']} "
                        f"name={job['name']} ret={ret} log={item['log_path']}",
                        flush=True,
                    )
                else:
                    failed += 1
                    print(
                        f"[FAIL] job={item['job_id']:05d} gpu={item['gpu']} "
                        f"name={job['name']} ret={ret} log={item['log_path']}",
                        flush=True,
                    )

                    if args.stop_on_fail:
                        print("stop_on_fail=1, terminating active jobs.", flush=True)
                        for a in still_active:
                            try:
                                a["process"].terminate()
                            except Exception:
                                pass
                        raise SystemExit(1)

        active = still_active

        free_mem = query_gpu_free_memory() if args.min_free_mem_mb > 0 else {}

        launched = False

        for gpu in gpus:
            while queue and gpu_slots[gpu] < args.max_per_gpu:
                if args.min_free_mem_mb > 0:
                    mem = free_mem.get(gpu, None)
                    if mem is not None and mem < args.min_free_mem_mb:
                        break

                job = queue.pop(0)

                # Double-check skip at launch time.
                if args.skip_existing and is_job_done(job):
                    skipped += 1
                    print(
                        f"[SKIP] existing result name={job['name']} result={expected_result_path(job)}",
                        flush=True,
                    )
                    continue

                item = launch_process(job, gpu, next_job_id, args)
                active.append(item)
                gpu_slots[gpu] += 1

                print(
                    f"[LAUNCH] job={next_job_id:05d} gpu={gpu} "
                    f"active={len(active)} queue={len(queue)} name={job['name']} "
                    f"log={item['log_path']}",
                    flush=True,
                )

                next_job_id += 1
                launched = True

                if args.launch_delay > 0:
                    time.sleep(args.launch_delay)

        if not launched:
            time.sleep(args.poll_interval)

    print("=" * 120, flush=True)
    print("ALL JOBS FINISHED", flush=True)
    print("completed:", completed, flush=True)
    print("failed:", failed, flush=True)
    print("relaunched:", relaunched, flush=True)
    print("skipped:", skipped, flush=True)
    print("time:", datetime.now().isoformat(), flush=True)
    print("logs:", LOG_ROOT.resolve(), flush=True)
    print("=" * 120, flush=True)


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
    p.add_argument("--min_free_mem_mb", type=int, default=0)

    p.add_argument("--skip_existing", type=int, default=1)
    p.add_argument("--stop_on_fail", type=int, default=0)
    p.add_argument("--retries", type=int, default=0)

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
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--main_llm", type=str, default="qwen2p5_1p5b")
    p.add_argument("--debug_llm", type=str, default="qwen2p5_0p5b_instruct")

    p.add_argument(
        "--backbones",
        type=str,
        default=",".join(DEFAULT_BACKBONES),
    )

    p.add_argument(
        "--big_backbones",
        type=str,
        default=",".join(BIG_BACKBONES),
    )

    p.add_argument(
        "--semantic_encoders",
        type=str,
        default=",".join(SEMANTIC_ENCODERS),
    )

    p.add_argument("--m4_max_series", type=int, default=-1)

    p.add_argument("--max_train_batches", type=int, default=-1)
    p.add_argument("--max_eval_batches", type=int, default=-1)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    scheduler(args)