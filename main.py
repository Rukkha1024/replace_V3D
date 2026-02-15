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

from replace_v3d.events import (
    load_subject_leg_length_cm,
    parse_subject_velocity_trial_from_filename,
    resolve_subject_from_token,
)


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


def _collect_outputs(*, out_dir: Path, c3d_stem: str, steps: set[str]) -> list[Path]:
    out = []
    if "angles" in steps:
        out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep.csv")
        out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep.xlsx")
        out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep_anat.csv")
        out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep_ana0.csv")
    if "mos" in steps:
        out.append(out_dir / f"{c3d_stem}_MOS_preStep.xlsx")
    if "torque" in steps:
        out.append(out_dir / f"{c3d_stem}_ankle_torque.xlsx")
    return [p for p in out if p.exists()]


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
        description=(
            "Default: batch export unified time series CSV (MOS/COM/xCOM/BOS + joint angles + ankle torque).\n"
            "Single-file processing is available only via --c3d."
        )
    )
    p.add_argument("--c3d", default=None, help="Optional single C3D file. If provided, runs single-file pipelines.")
    p.add_argument(
        "--steps",
        choices=["all", "mos", "angles", "torque"],
        default="all",
        help="Single-file mode only: which pipeline(s) to run.",
    )
    p.add_argument("--c3d_dir", default=str(_REPO_ROOT / "data" / "all_data"))
    p.add_argument("--event_xlsm", default=str(_REPO_ROOT / "data" / "perturb_inform.xlsm"))
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
    if args.c3d:
        c3d_path = Path(args.c3d)
        if not c3d_path.exists():
            raise FileNotFoundError(f"C3D file not found: {c3d_path}")

        try:
            subject_token, velocity, trial = parse_subject_velocity_trial_from_filename(c3d_path.name)
            subject = resolve_subject_from_token(event_xlsm, subject_token)
        except Exception as exc:
            raise RuntimeError(f"Cannot resolve subject/velocity/trial from {c3d_path.name}: {exc}") from exc

        stem = c3d_path.stem
        if args.steps == "all":
            steps = {"mos", "angles", "torque"}
        else:
            steps = {str(args.steps)}

        common_args = [
            "--c3d",
            str(c3d_path),
            "--event_xlsm",
            str(event_xlsm),
            "--subject",
            subject,
            "--velocity",
            str(velocity),
            "--trial",
            str(trial),
            "--out_dir",
            str(out_dir),
        ]

        if "mos" in steps:
            leg_length_cm: Optional[float] = load_subject_leg_length_cm(event_xlsm, subject)
            if leg_length_cm is None:
                msg = f"[SKIP] {c3d_path.name}: leg_length_cm missing for subject {subject}"
                print(msg)
                if args.on_error == "abort":
                    raise RuntimeError(msg)
            else:
                cmd_mos = [
                    sys.executable,
                    str(scripts_root / "run_mos_pipeline.py"),
                    *common_args,
                    "--leg_length_cm",
                    str(leg_length_cm),
                ]
                print(f"[RUN] {c3d_path.name} => run_mos_pipeline.py")
                _run_command(cmd_mos, step_name=f"mos:{stem}", on_error=args.on_error)

        if "angles" in steps:
            cmd_joint = [
                sys.executable,
                str(scripts_root / "run_joint_angles_pipeline.py"),
                *common_args,
            ]
            print(f"[RUN] {c3d_path.name} => run_joint_angles_pipeline.py")
            _run_command(cmd_joint, step_name=f"joint:{stem}", on_error=args.on_error)

        if "torque" in steps:
            cmd_ankle = [
                sys.executable,
                str(scripts_root / "run_ankle_torque_pipeline.py"),
                *common_args,
            ]
            print(f"[RUN] {c3d_path.name} => run_ankle_torque_pipeline.py")
            _run_command(cmd_ankle, step_name=f"ankle:{stem}", on_error=args.on_error)

        produced = _collect_outputs(out_dir=out_dir, c3d_stem=stem, steps=steps)
        if produced:
            print("[OUTPUT] generated files:")
            for p in produced:
                print(f" - {p}")
        else:
            print("[OUTPUT] no outputs captured.")

        if md5_reference_dir is not None:
            _compare_md5(produced, md5_reference_dir, out_dir)
        return

    if not c3d_dir.exists():
        raise FileNotFoundError(f"C3D directory not found: {c3d_dir}")

    batch_cmd = [
        sys.executable,
        str(scripts_root / "run_batch_all_timeseries_csv.py"),
        "--c3d_dir",
        str(c3d_dir),
        "--event_xlsm",
        str(event_xlsm),
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

    print("[RUN] scripts/run_batch_all_timeseries_csv.py")
    if _run_command(batch_cmd, step_name="batch_all_timeseries", on_error=args.on_error):
        if out_csv.exists():
            produced.append(out_csv)

    if md5_reference_dir is not None:
        _compare_md5(produced, md5_reference_dir, out_dir)


if __name__ == "__main__":
    main()
