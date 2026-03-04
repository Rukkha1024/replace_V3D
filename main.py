from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Allow running package imports without installation.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

def _md5_of_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(cmd: list[str], *, step_name: str, on_error: str) -> bool:
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if result.stdout:
            print(f"[{step_name}] stdout:\n{result.stdout.strip()}")
        if result.stderr:
            print(f"[{step_name}] stderr:\n{result.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[{step_name}] failed (code={exc.returncode})")
        if exc.stdout:
            print(f"[{step_name}] stdout:\n{exc.stdout.strip()}")
        if exc.stderr:
            print(f"[{step_name}] stderr:\n{exc.stderr.strip()}")
        if on_error == "abort":
            raise RuntimeError(f"{step_name} failed") from exc
        return False


def _relative_reference_path(path: Path, output_root: Path) -> Path:
    try:
        return path.relative_to(output_root)
    except ValueError:
        return path.name


def _compare_md5(outputs: list[Path], reference_dir: Path, output_root: Path) -> None:
    if not outputs:
        print("[MD5] no outputs to compare.")
        return
    if not reference_dir.exists():
        print(f"[MD5] reference directory not found: {reference_dir}")
        return

    ok = 0
    mismatch = 0
    missing = 0

    for out in outputs:
        output_hash = _md5_of_file(out)
        rel = _relative_reference_path(out, output_root)
        ref = reference_dir / rel
        if not ref.exists():
            alt = reference_dir / out.name
            ref = alt
        if not ref.exists():
            print(f"[MD5] MISSING REF: {out.name}")
            missing += 1
            continue
        ref_hash = _md5_of_file(ref)
        if output_hash == ref_hash:
            print(f"[MD5] OK: {out.name}")
            ok += 1
        else:
            print(f"[MD5] DIFF: {out.name}")
            print(f"       output: {out}")
            print(f"       ref:    {ref}")
            mismatch += 1

    print(f"[MD5] result: ok={ok}, mismatch={mismatch}, missing={missing}")


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Batch export unified time series CSV (MOS/COM/xCOM/BOS + joint angles + ankle torque)."
    )
    p.add_argument("--c3d_dir", default=str(_REPO_ROOT / "data" / "all_data"))
    p.add_argument("--event_xlsm", default=str(_REPO_ROOT / "data" / "perturb_inform.xlsm"))
    p.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config.yaml"),
        help="Config YAML path (forceplate/plot options).",
    )
    p.add_argument("--out_dir", default=str(_REPO_ROOT / "output"))
    p.add_argument(
        "--out_csv",
        default=str(_REPO_ROOT / "output" / "all_trials_timeseries.csv"),
        help="Batch output CSV path (default: output/all_trials_timeseries.csv).",
    )
    p.add_argument("--pre_frames", type=int, default=100)
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--overwrite", action="store_true", help="Overwrite batch timeseries CSV.")
    p.add_argument("--skip_unmatched", action="store_true", help="Pass unmatched cases in batch export.")
    p.add_argument(
        "--meta_prefilter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply add_meta.ipynb-equivalent trial filter before batch computations "
            "(default: enabled)."
        ),
    )
    p.add_argument("--on_error", choices=["continue", "abort"], default="continue")
    p.add_argument("--md5_reference_dir", default=None, help="Optional reference directory for MD5 compare.")
    p.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit number of C3D files for batch export (for quick checks).",
    )
    return p


def main() -> None:
    args = _make_parser().parse_args()

    c3d_dir = Path(args.c3d_dir)
    event_xlsm = Path(args.event_xlsm)
    out_dir = Path(args.out_dir)
    out_csv = Path(args.out_csv)
    md5_reference_dir: Optional[Path] = Path(args.md5_reference_dir) if args.md5_reference_dir else None

    if not event_xlsm.exists():
        raise FileNotFoundError(f"event_xlsm not found: {event_xlsm}")

    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_root = _REPO_ROOT / "scripts"
    produced: list[Path] = []

    if not c3d_dir.exists():
        raise FileNotFoundError(f"C3D directory not found: {c3d_dir}")

    batch_cmd = [
        sys.executable,
        str(scripts_root / "run_batch_all_timeseries_csv.py"),
        "--c3d_dir",
        str(c3d_dir),
        "--event_xlsm",
        str(event_xlsm),
        "--config",
        str(args.config),
        "--out_csv",
        str(out_csv),
        "--pre_frames",
        str(args.pre_frames),
        "--encoding",
        str(args.encoding),
    ]
    if args.skip_unmatched:
        batch_cmd.append("--skip_unmatched")
    if args.overwrite:
        batch_cmd.append("--overwrite")
    if args.max_files is not None:
        batch_cmd.extend(["--max_files", str(int(args.max_files))])
    if args.meta_prefilter:
        batch_cmd.append("--meta_prefilter")
    else:
        batch_cmd.append("--no-meta_prefilter")

    print("[RUN] scripts/run_batch_all_timeseries_csv.py")
    if _run_command(batch_cmd, step_name="batch_all_timeseries", on_error=args.on_error):
        if out_csv.exists():
            produced.append(out_csv)

    if md5_reference_dir is not None:
        _compare_md5(produced, md5_reference_dir, out_dir)


if __name__ == "__main__":
    main()
