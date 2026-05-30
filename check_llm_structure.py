# -*- coding: utf-8 -*-
"""
Check structure of ~/LLM project folder.

Run:
  python check_llm_structure.py

Optional:
  python check_llm_structure.py --root ~/LLM --max_depth 3 --show_files 30
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


IMPORTANT_DIRS = [
    "data_icdm_tsf",
    "runs_parallel_logs",
    "runs_icdm_psllm_parallel",
    "runs_icdm_psllm",
]

IMPORTANT_FILES = [
    "run_icdm_psllm.py",
    "launch_icdm_parallel.py",
    "launch_icdm_parallel_v2.py",
    "dynamic_continue_from_job0021.py",
    "jobs_manifest.json",
    "launcher_args.json",
]


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:,.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"


def file_mtime(path: Path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "NA"


def dir_size(path: Path):
    total = 0
    n_files = 0
    n_dirs = 0

    if not path.exists():
        return 0, 0, 0

    for root, dirs, files in os.walk(path):
        n_dirs += len(dirs)
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
                n_files += 1
            except Exception:
                pass

    return total, n_files, n_dirs


def print_header(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def print_top_level(root: Path):
    print_header("TOP LEVEL")

    items = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

    for p in items:
        try:
            st = p.stat()
            kind = "DIR " if p.is_dir() else "FILE"
            size = ""
            if p.is_file():
                size = sizeof_fmt(st.st_size)
            else:
                dsize, nf, nd = dir_size(p)
                size = f"{sizeof_fmt(dsize)} | files={nf} dirs={nd}"

            print(f"{kind:4} {p.name:45} {size:30} modified={file_mtime(p)}")
        except Exception as e:
            print(f"ERR  {p.name}: {e}")


def tree(root: Path, max_depth: int, show_files: int):
    print_header(f"TREE max_depth={max_depth}")

    printed_files = 0

    def rec(path: Path, depth: int, prefix: str):
        nonlocal printed_files

        if depth > max_depth:
            return

        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as e:
            print(prefix + f"[ERR] {path.name}: {e}")
            return

        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            line_prefix = prefix + connector

            if child.is_dir():
                dsize, nf, nd = dir_size(child)
                print(line_prefix + f"{child.name}/  [{sizeof_fmt(dsize)}, files={nf}, dirs={nd}]")
                rec(child, depth + 1, prefix + ("    " if i == len(children) - 1 else "│   "))
            else:
                if printed_files < show_files:
                    try:
                        print(line_prefix + f"{child.name}  [{sizeof_fmt(child.stat().st_size)}, {file_mtime(child)}]")
                    except Exception:
                        print(line_prefix + f"{child.name}")
                    printed_files += 1

    print(str(root))
    rec(root, 1, "")

    if printed_files >= show_files:
        print(f"\n[NOTE] File display limited to {show_files}. Increase --show_files if needed.")


def check_datasets(root: Path):
    print_header("DATASETS CHECK")

    data_root = root / "data_icdm_tsf"

    expected = {
        "ETTh1": "ETTh1/ETTh1.csv",
        "ETTh2": "ETTh2/ETTh2.csv",
        "ETTm1": "ETTm1/ETTm1.csv",
        "ETTm2": "ETTm2/ETTm2.csv",
        "Weather": "Weather/Weather.csv",
        "Traffic": "Traffic/Traffic.csv",
        "Electricity": "Electricity/Electricity.csv",
        "Exchange": "Exchange/Exchange.csv",
        "ILI": "ILI/ILI.csv",
    }

    if not data_root.exists():
        print("NOT FOUND:", data_root)
        return

    for name, rel in expected.items():
        p = data_root / rel
        if p.exists():
            try:
                size = sizeof_fmt(p.stat().st_size)
                # only header + first rows count-safe via pandas avoided for speed
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    header = f.readline().strip()
                n_cols = len(header.split(",")) if header else 0
                print(f"OK       {name:12} {str(p):55} size={size:12} cols≈{n_cols:4} modified={file_mtime(p)}")
            except Exception as e:
                print(f"OK/ERR   {name:12} {p} error={e}")
        else:
            print(f"MISSING  {name:12} {p}")

    m4 = data_root / "M4"
    if m4.exists():
        print("\nM4 FOUND:", m4)
        for p in sorted(m4.rglob("*.csv"))[:30]:
            print(f"  {p.relative_to(data_root)} | {sizeof_fmt(p.stat().st_size)} | {file_mtime(p)}")
    else:
        print("\nM4 not found:", m4)


def check_runs(root: Path):
    print_header("RUNS / LOGS CHECK")

    log_root = root / "runs_parallel_logs"
    out_root = root / "runs_icdm_psllm_parallel"

    for d in [log_root, out_root, root / "runs_icdm_psllm"]:
        if d.exists():
            size, nf, nd = dir_size(d)
            print(f"DIR {d.name:30} size={sizeof_fmt(size):12} files={nf:6} dirs={nd:6} modified={file_mtime(d)}")
        else:
            print(f"MISSING {d}")

    if log_root.exists():
        logs = sorted(log_root.glob("*.log"), key=lambda p: p.stat().st_mtime)
        cmds = sorted(log_root.glob("*.cmd"), key=lambda p: p.stat().st_mtime)
        dones = sorted(log_root.glob("*.done"), key=lambda p: p.stat().st_mtime)

        print("\nruns_parallel_logs counts:")
        print(f"  logs : {len(logs)}")
        print(f"  cmds : {len(cmds)}")
        print(f"  done : {len(dones)}")

        print("\nLast 15 logs:")
        for p in logs[-15:]:
            print(f"  {p.name:65} {sizeof_fmt(p.stat().st_size):10} {file_mtime(p)}")

        print("\nLast 15 done:")
        for p in dones[-15:]:
            print(f"  {p.name:65} {sizeof_fmt(p.stat().st_size):10} {file_mtime(p)}")

        manifest = log_root / "jobs_manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                print(f"\nManifest: {manifest}")
                print(f"  jobs: {len(data)}")
                print("  first 5:")
                for item in data[:5]:
                    print("   ", item.get("idx"), item.get("name"), item.get("expected_result"))
                print("  last 5:")
                for item in data[-5:]:
                    print("   ", item.get("idx"), item.get("name"), item.get("expected_result"))
            except Exception as e:
                print("Manifest read error:", e)

    if out_root.exists():
        results = sorted(out_root.rglob("result.json"), key=lambda p: p.stat().st_mtime)
        histories = sorted(out_root.rglob("history.csv"), key=lambda p: p.stat().st_mtime)
        bests = sorted(out_root.rglob("best_model.pt"), key=lambda p: p.stat().st_mtime)

        print("\nruns_icdm_psllm_parallel outputs:")
        print(f"  result.json  : {len(results)}")
        print(f"  history.csv  : {len(histories)}")
        print(f"  best_model.pt: {len(bests)}")

        print("\nLast 15 result.json:")
        for p in results[-15:]:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                print(
                    f"  {p.parent.name[:70]:70} "
                    f"mse={obj.get('test_mse')} mae={obj.get('test_mae')} "
                    f"modified={file_mtime(p)}"
                )
            except Exception:
                print(f"  {p} {file_mtime(p)}")


def check_scripts(root: Path):
    print_header("SCRIPTS CHECK")

    scripts = [
        "run_icdm_psllm.py",
        "launch_icdm_parallel.py",
        "launch_icdm_parallel_v2.py",
        "dynamic_continue_from_job0021.py",
        "restart_from_job0021.py",
        "restart_from_job0021_verbose.py",
        "continue_after_last_done.py",
        "resume_icdm_from_manifest.py",
    ]

    for s in scripts:
        p = root / s
        if p.exists():
            print(f"OK      {s:40} {sizeof_fmt(p.stat().st_size):12} modified={file_mtime(p)}")
        else:
            print(f"MISSING {s}")


def check_processes():
    print_header("CURRENT PROCESSES")

    try:
        out = subprocess.check_output(
            ["bash", "-lc", "ps -eo pid,ppid,stat,etime,cmd | grep -E 'run_icdm_psllm|launch_icdm|dynamic_continue|restart_from_job|continue_after|resume_icdm|wait_' | grep -v grep"],
            text=True,
            errors="ignore",
        )
        print(out.strip() if out.strip() else "No matching processes.")
    except Exception:
        print("No matching processes.")

    print("\nGPU:")
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, errors="ignore")
        print(out)
    except Exception as e:
        print("nvidia-smi error:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--show_files", type=int, default=40)
    ap.add_argument("--no_tree", action="store_true")
    ap.add_argument("--no_process", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    print_header("PROJECT STRUCTURE REPORT")
    print("root:", root)
    print("time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print_top_level(root)
    check_scripts(root)
    check_datasets(root)
    check_runs(root)

    if not args.no_tree:
        tree(root, max_depth=args.max_depth, show_files=args.show_files)

    if not args.no_process:
        check_processes()


if __name__ == "__main__":
    main()