# -*- coding: utf-8 -*-
"""
Dynamic continuation from old manifest, starting at job_0021.

NO DELETE.
NO CLEANUP.
NO NOHUP REQUIRED.

Behavior:
  - reads runs_parallel_logs/jobs_manifest.json
  - starts from manifest job number 21 by default
  - skips completed jobs by result.json or .done
  - skips jobs already running
  - dynamically fills GPUs based on free memory and utilization
  - uses old log naming:
      runs_parallel_logs/job_0021_<name>_gpu0.log
  - appends to existing logs instead of overwriting
  - streams child output to console
"""

import os
import re
import sys
import json
import time
import shlex
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime


MANIFEST = Path("runs_parallel_logs/jobs_manifest.json")
LOG_ROOT = Path("runs_parallel_logs")


def sanitize(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("=", "-")
    )


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def load_manifest(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = []

    for i, item in enumerate(data):
        name = item.get("name") or f"job_{i + 1:04d}"
        cmd_str = item.get("cmd")
        expected_result = item.get("expected_result")

        if not cmd_str:
            continue

        jobs.append({
            "job_no": i + 1,
            "index0": i,
            "name": name,
            "cmd": shlex.split(cmd_str),
            "cmd_str": cmd_str,
            "expected_result": Path(expected_result) if expected_result else None,
        })

    return jobs


def detect_gpus():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as e:
        print(f"[GPU-DETECT-ERROR] {e}", flush=True)
        return [{"id": "0", "name": "unknown", "total": 0, "free": 0, "util": 0}]

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        gpus.append({
            "id": parts[0],
            "name": parts[1],
            "total": int(float(parts[2])),
            "free": int(float(parts[3])),
            "util": int(float(parts[4])),
        })

    return gpus


def get_gpu_stats(selected_ids):
    all_gpus = detect_gpus()
    selected = set(str(x) for x in selected_ids)
    return [g for g in all_gpus if g["id"] in selected]


def done_paths(job):
    safe = sanitize(job["name"])
    return [
        LOG_ROOT / f"{job['name']}.done",
        LOG_ROOT / f"{safe}.done",
    ]


def result_exists(job):
    p = job.get("expected_result")
    return p is not None and p.exists() and p.stat().st_size > 0


def done_exists(job):
    return any(p.exists() and p.stat().st_size > 0 for p in done_paths(job))


def is_completed(job):
    return result_exists(job) or done_exists(job)


def get_running_cmd_lines():
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True,
            errors="ignore",
        )
    except Exception:
        return []

    lines = []
    for line in out.splitlines():
        if "run_icdm_psllm.py" in line and "grep" not in line:
            lines.append(line)
    return lines


def is_running(job, running_lines):
    cmd = job["cmd"]

    key_parts = []
    for flag in [
        "--out_dir",
        "--dataset",
        "--model",
        "--pred_len",
        "--llm_name",
        "--semantic_encoder",
        "--task",
        "--m4_freq",
    ]:
        if flag in cmd:
            i = cmd.index(flag)
            if i + 1 < len(cmd):
                key_parts.append(flag)
                key_parts.append(cmd[i + 1])

    if not key_parts:
        return False

    for line in running_lines:
        if all(part in line for part in key_parts):
            return True

    return False


def mark_done(job, log_path):
    p = LOG_ROOT / f"{job['name']}.done"
    p.write_text(
        f"done={datetime.now().isoformat()}\n"
        f"name={job['name']}\n"
        f"expected_result={job.get('expected_result')}\n"
        f"log={log_path}\n",
        encoding="utf-8",
    )


def stream_output(proc, log_file, prefix):
    for line in proc.stdout:
        text = line.rstrip("\n")
        print(f"{prefix} {text}", flush=True)
        log_file.write(text + "\n")
        log_file.flush()


def launch(job, gpu_id):
    safe_name = sanitize(job["name"])

    log_path = LOG_ROOT / f"job_{job['job_no']:04d}_{safe_name}_gpu{gpu_id}.log"
    cmd_path = LOG_ROOT / f"job_{job['job_no']:04d}_{safe_name}_gpu{gpu_id}.cmd"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd_path.write_text(shell_join(job["cmd"]) + "\n", encoding="utf-8")

    # append mode: do not erase previous incomplete log
    log_file = open(log_path, "a", encoding="utf-8")

    header = [
        "",
        "=" * 120,
        f"RESTART_APPEND={datetime.now().isoformat()}",
        f"JOB_NO={job['job_no']:04d}",
        f"MANIFEST_INDEX={job['index0']}",
        f"NAME={job['name']}",
        f"GPU={gpu_id}",
        f"EXPECTED_RESULT={job.get('expected_result')}",
        f"CMD={shell_join(job['cmd'])}",
        "=" * 120,
    ]

    prefix = f"[job_{job['job_no']:04d}|gpu{gpu_id}|{job['name']}]"

    for h in header:
        print(f"{prefix} {h}", flush=True)
        log_file.write(h + "\n")
    log_file.flush()

    proc = subprocess.Popen(
        job["cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        env=env,
        text=True,
        bufsize=1,
    )

    thread = threading.Thread(
        target=stream_output,
        args=(proc, log_file, prefix),
        daemon=True,
    )
    thread.start()

    return {
        "process": proc,
        "thread": thread,
        "job": job,
        "gpu": str(gpu_id),
        "log_path": log_path,
        "cmd_path": cmd_path,
        "log_file": log_file,
        "start_time": time.time(),
    }


def print_gpu_stats(gpus, gpu_active):
    print("-" * 120, flush=True)
    print(f"GPU STATUS {datetime.now().isoformat()}", flush=True)
    for g in gpus:
        gid = g["id"]
        print(
            f"  gpu={gid} | name={g['name']} | free={g['free']} MiB / {g['total']} MiB | "
            f"util={g['util']}% | active_jobs={gpu_active.get(gid, 0)}",
            flush=True,
        )
    print("-" * 120, flush=True)


def choose_gpu(gpus, gpu_active, max_per_gpu, min_free_mb, max_util):
    """
    Pick GPU with most free memory that passes thresholds.
    """
    candidates = []

    for g in gpus:
        gid = g["id"]
        active = gpu_active.get(gid, 0)

        if active >= max_per_gpu:
            continue

        if g["free"] < min_free_mb:
            continue

        if g["util"] > max_util:
            continue

        candidates.append(g)

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["free"], -x["util"]), reverse=True)
    return candidates[0]["id"]


def scheduler(args):
    jobs = load_manifest(Path(args.manifest))

    if len(jobs) < args.start_job_no:
        raise RuntimeError(f"Manifest has only {len(jobs)} jobs. Cannot start from {args.start_job_no}")

    start_idx = args.start_job_no - 1
    raw_queue = jobs[start_idx:]

    if args.gpus == "auto":
        detected = detect_gpus()
        gpu_ids = [g["id"] for g in detected]
    else:
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]

    running_lines = get_running_cmd_lines()

    queue = []
    skipped_done = 0
    skipped_running = 0

    for job in raw_queue:
        if is_completed(job):
            skipped_done += 1
            continue
        if args.skip_running and is_running(job, running_lines):
            skipped_running += 1
            continue
        queue.append(job)

    if args.limit > 0:
        queue = queue[:args.limit]

    print("=" * 120, flush=True)
    print("DYNAMIC CONTINUE FROM OLD MANIFEST", flush=True)
    print("NO DELETE MODE", flush=True)
    print("manifest:", args.manifest, flush=True)
    print("start_job_no:", args.start_job_no, flush=True)
    print("start_manifest_name:", jobs[start_idx]["name"], flush=True)
    print("selected_gpus:", gpu_ids, flush=True)
    print("max_per_gpu:", args.max_per_gpu, flush=True)
    print("min_free_mb:", args.min_free_mb, flush=True)
    print("max_util:", args.max_util, flush=True)
    print("raw_tail_jobs:", len(raw_queue), flush=True)
    print("skipped_done:", skipped_done, flush=True)
    print("skipped_running:", skipped_running, flush=True)
    print("queued:", len(queue), flush=True)
    print("=" * 120, flush=True)

    active = []
    gpu_active = {gid: 0 for gid in gpu_ids}

    completed = 0
    failed = 0
    last_status = 0

    while queue or active:
        still = []

        for item in active:
            ret = item["process"].poll()

            if ret is None:
                still.append(item)
                continue

            item["thread"].join(timeout=5)

            item["log_file"].write("\n" + "=" * 120 + "\n")
            item["log_file"].write(f"END={datetime.now().isoformat()}\n")
            item["log_file"].write(f"RETURN_CODE={ret}\n")
            item["log_file"].close()

            gpu_active[item["gpu"]] -= 1

            job = item["job"]

            if ret == 0 and result_exists(job):
                completed += 1
                mark_done(job, item["log_path"])
                print(
                    f"[DONE] job_{job['job_no']:04d} gpu={item['gpu']} "
                    f"name={job['name']} completed={completed}",
                    flush=True,
                )
            else:
                failed += 1
                print(
                    f"[FAIL] job_{job['job_no']:04d} gpu={item['gpu']} "
                    f"name={job['name']} ret={ret} log={item['log_path']}",
                    flush=True,
                )
                if args.stop_on_fail:
                    for a in still:
                        try:
                            a["process"].terminate()
                        except Exception:
                            pass
                    raise SystemExit(1)

        active = still

        gpus = get_gpu_stats(gpu_ids)

        launched_any = False

        while queue:
            gpu_id = choose_gpu(
                gpus=gpus,
                gpu_active=gpu_active,
                max_per_gpu=args.max_per_gpu,
                min_free_mb=args.min_free_mb,
                max_util=args.max_util,
            )

            if gpu_id is None:
                break

            job = queue.pop(0)

            # late skip
            current_running = get_running_cmd_lines()
            if is_completed(job):
                print(f"[SKIP-LATE-DONE] job_{job['job_no']:04d} {job['name']}", flush=True)
                continue
            if args.skip_running and is_running(job, current_running):
                print(f"[SKIP-LATE-RUNNING] job_{job['job_no']:04d} {job['name']}", flush=True)
                continue

            item = launch(job, gpu_id)
            active.append(item)
            gpu_active[gpu_id] += 1

            print(
                f"[LAUNCH] job_{job['job_no']:04d} gpu={gpu_id} "
                f"active_total={len(active)} active_on_gpu={gpu_active[gpu_id]} "
                f"queue={len(queue)} name={job['name']} log={item['log_path']}",
                flush=True,
            )

            launched_any = True
            time.sleep(args.launch_delay)

            # refresh stats after each launch
            gpus = get_gpu_stats(gpu_ids)

        now = time.time()
        if now - last_status >= args.status_interval:
            print_gpu_stats(gpus, gpu_active)
            print(
                f"QUEUE STATUS active={len(active)} queue={len(queue)} "
                f"completed={completed} failed={failed}",
                flush=True,
            )
            for item in active:
                job = item["job"]
                elapsed = (time.time() - item["start_time"]) / 60.0
                print(
                    f"  ACTIVE job_{job['job_no']:04d} gpu={item['gpu']} "
                    f"elapsed={elapsed:.1f}min name={job['name']}",
                    flush=True,
                )
            last_status = now

        if not launched_any:
            time.sleep(args.poll_interval)

    print("=" * 120, flush=True)
    print("DYNAMIC CONTINUE FINISHED", flush=True)
    print("completed:", completed, flush=True)
    print("failed:", failed, flush=True)
    print("=" * 120, flush=True)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--manifest", type=str, default="runs_parallel_logs/jobs_manifest.json")
    p.add_argument("--start_job_no", type=int, default=21)

    p.add_argument("--gpus", type=str, default="auto")
    p.add_argument("--max_per_gpu", type=int, default=4)

    p.add_argument("--min_free_mb", type=int, default=6000)
    p.add_argument("--max_util", type=int, default=98)

    p.add_argument("--skip_running", type=int, default=1)
    p.add_argument("--stop_on_fail", type=int, default=0)

    p.add_argument("--limit", type=int, default=-1)

    p.add_argument("--poll_interval", type=float, default=10.0)
    p.add_argument("--status_interval", type=float, default=30.0)
    p.add_argument("--launch_delay", type=float, default=2.0)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    scheduler(args)
