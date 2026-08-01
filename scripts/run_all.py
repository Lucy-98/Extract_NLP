#!/usr/bin/env python3
"""Single entrypoint for the full pipeline -- so a leaderboard-time-pressure
re-run doesn't depend on remembering the right order from README.md.

Stages:
  prepare   prepare_ner_dataset.py + build_terminology_index.py + augment_ner_dataset.py,
            then sync train_augmented.jsonl into kaggle_upload/dataset/train.jsonl.
            Local, stdlib only. Uses only ids that have both input/{id}.txt and output/{id}.json.
  train     Push kaggle_upload/kernel to Kaggle forcing a T4 GPU (see README.md for why
            T4 specifically), after versioning kaggle_upload/dataset unless skipped.
            Poll until it finishes, download the exported model into models/ner_model/.
            Needs the `kaggle` package and ~/.kaggle/kaggle.json. Consumes your Kaggle
            weekly GPU quota every time it runs -- don't loop this.
  infer     run_pipeline.py, then check_submission.py --pred output_model,
            then package output.zip from output_model/.
  package   Validate output_model/ and write output.zip without rerunning inference.
  submit    Offline submission path: infer -> validate -> package. It does not train.
  all       prepare -> train -> infer, in order, stopping on the first failure.

Usage:
  python scripts/run_all.py prepare --aug-multiplier 1 --assertion-docs 30 --train-all-labels
  python scripts/run_all.py train
  python scripts/run_all.py infer
  python scripts/run_all.py package
  python scripts/run_all.py submit --input input_turn2 --pred output_model_turn2 --out output_turn2.zip
  python scripts/run_all.py all --aug-multiplier 1 --assertion-docs 30 --train-all-labels
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
KERNEL_DIR = ROOT / "kaggle_upload" / "kernel"
KAGGLE_DATASET_DIR = ROOT / "kaggle_upload" / "dataset"
BUNDLE_DIR = ROOT / "kaggle_bundle"  # full dataset uploaded to Kaggle; see stage_dataset_version
KERNEL_SLUG = "lucylng/viettelrace-ner-assertion-train"
MODEL_DIR = ROOT / "models" / "ner_model"
INPUT_DIR = ROOT / "input"
PRED_DIR = ROOT / "output_model"
TRUTH_DIR = ROOT / "output"
EXPORT_FILES = ("config.json", "model.pt", "tokenizer.json", "tokenizer_config.json")
OPTIONAL_EXPORT_FILES = ("hf_config.json",)
MIN_MODEL_PT_BYTES = 100_000_000

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# The kaggle CLI's own stdout can contain Vietnamese text (e.g. kernel titles);
# on Windows the default console codepage crashes on that unless UTF-8 I/O is
# forced. See CLAUDE.md "Training on Kaggle".
BASE_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
KAGGLE_ENV = {**BASE_ENV}


def run(cmd: list[str], **kwargs) -> None:
    print(f"$ {' '.join(cmd)}")
    kwargs.setdefault("env", BASE_ENV)
    subprocess.run(cmd, check=True, cwd=ROOT, **kwargs)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def replace_existing(tmp: Path, path: Path) -> None:
    try:
        os.replace(tmp, path)
        return
    except PermissionError:
        if path.exists():
            try:
                path.chmod(0o666)
            except OSError:
                pass
            path.unlink()
        try:
            os.replace(tmp, path)
        except PermissionError:
            shutil.copy2(tmp, path)
            tmp.unlink(missing_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"viettelrace-{os.getpid()}-{dst.name}.tmp"
    shutil.copy2(src, tmp)
    replace_existing(tmp, dst)


def export_dir_issues(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.is_dir():
        return [f"{path} is not a directory"]
    for name in EXPORT_FILES:
        file_path = path / name
        if not file_path.is_file():
            issues.append(f"missing {file_path}")
            continue
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            issues.append(f"cannot stat {file_path}: {exc}")
            continue
        if size <= 0:
            issues.append(f"{file_path} is empty")
        elif name == "model.pt" and size < MIN_MODEL_PT_BYTES:
            issues.append(f"{file_path} is suspiciously small ({size} bytes)")
    return issues


def export_zip_issues(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path} is not a file"]
    try:
        with zipfile.ZipFile(path) as zf:
            entries = {Path(info.filename).name: info for info in zf.infolist() if not info.is_dir()}
    except zipfile.BadZipFile as exc:
        return [f"{path} is not a readable zip: {exc}"]

    issues: list[str] = []
    for name in EXPORT_FILES:
        info = entries.get(name)
        if info is None:
            issues.append(f"missing {name} in {path}")
            continue
        if info.file_size <= 0:
            issues.append(f"{name} in {path} is empty")
        elif name == "model.pt" and info.file_size < MIN_MODEL_PT_BYTES:
            issues.append(f"model.pt in {path} is suspiciously small ({info.file_size} bytes)")
    return issues


def valid_export_dir(path: Path) -> bool:
    return not export_dir_issues(path)


def valid_export_zip(path: Path) -> bool:
    return not export_zip_issues(path)


def find_valid_export_dir(root: Path) -> Path | None:
    if valid_export_dir(root):
        return root
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and valid_export_dir(child):
                return child
    return None


def find_kernel_export(download_dir: Path) -> Path | None:
    extracted_dir = download_dir / "ner_model_export"
    if valid_export_dir(extracted_dir):
        return extracted_dir

    export_zip = download_dir / "ner_model_export.zip"
    if valid_export_zip(export_zip):
        return export_zip

    return None


def download_kernel_output(download_dir: Path, retries: int = 3, retry_delay: int = 20) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "kernels",
        "output",
        KERNEL_SLUG,
        "-p",
        str(download_dir),
        "--force",
    ]
    for attempt in range(1, retries + 1):
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT, env=KAGGLE_ENV)
        if result.returncode == 0:
            return
        if find_kernel_export(download_dir) is not None:
            print("Kaggle output failed, but a complete model export is present locally; continuing.")
            return
        if attempt < retries:
            print(f"Kernel output download failed (attempt {attempt}/{retries}); retrying in {retry_delay}s")
            time.sleep(retry_delay)

    raise SystemExit(f"kaggle kernels output failed after {retries} attempts.")


def snapshot_current_model() -> None:
    """Copy models/ner_model aside before a new export overwrites it.

    Kaggle's kernel-output endpoint ignores a version suffix and always returns
    the latest run, so a model you overwrite is gone for good -- that is how the
    40.828-scoring v12 weights were lost (worklog 2026-07-21). One local copy is
    ~1.1GB, which is cheap next to a leaderboard regression you cannot roll back.
    """
    if not (MODEL_DIR / "model.pt").is_file():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = MODEL_DIR.parent / f"{MODEL_DIR.name}_backup_{stamp}"
    if backup.exists():
        return
    shutil.copytree(MODEL_DIR, backup)
    print(f"Snapshotted the model being replaced -> {backup}")


def install_export_from_dir(export_dir: Path) -> None:
    issues = export_dir_issues(export_dir)
    if issues:
        raise SystemExit("Invalid extracted model export:\n  " + "\n  ".join(issues))
    snapshot_current_model()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_FILES:
        copy_file(export_dir / name, MODEL_DIR / name)
    for name in OPTIONAL_EXPORT_FILES:
        optional_path = export_dir / name
        if optional_path.is_file():
            copy_file(optional_path, MODEL_DIR / name)
    print(f"Copied {export_dir} -> {MODEL_DIR}")


def install_export_from_zip(export_zip: Path, work_dir: Path) -> None:
    issues = export_zip_issues(export_zip)
    if issues:
        raise SystemExit("Invalid model export zip:\n  " + "\n  ".join(issues))

    tmp_dir = work_dir / "_extract_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(export_zip) as zf:
            zf.extractall(tmp_dir)
        export_dir = find_valid_export_dir(tmp_dir)
        if export_dir is None:
            raise SystemExit(f"{export_zip} extracted, but no complete model export was found.")
        install_export_from_dir(export_dir)
        print(f"Unzipped {export_zip} -> {MODEL_DIR}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def install_kernel_export(download_dir: Path) -> None:
    export = find_kernel_export(download_dir)
    if export is None:
        extracted_dir = download_dir / "ner_model_export"
        export_zip = download_dir / "ner_model_export.zip"
        issues = export_dir_issues(extracted_dir) + export_zip_issues(export_zip)
        raise SystemExit(
            "No complete kernel model export found after download:\n  "
            + "\n  ".join(issues)
            + "\nDid the export cell finish on Kaggle?"
        )
    if export.suffix.lower() == ".zip":
        install_export_from_zip(export, download_dir)
    else:
        install_export_from_dir(export)


def stage_prepare(aug_multiplier: int, assertion_docs: int, train_all_labels: bool = False) -> None:
    prepare_cmd = [sys.executable, str(SCRIPTS / "prepare_ner_dataset.py")]
    if train_all_labels:
        prepare_cmd.append("--train-all-labels")
    run(prepare_cmd)
    run([sys.executable, str(SCRIPTS / "build_terminology_index.py")])
    run([
        sys.executable,
        str(SCRIPTS / "augment_ner_dataset.py"),
        "--multiplier",
        str(aug_multiplier),
        "--assertion-docs",
        str(assertion_docs),
    ])
    sync_kaggle_dataset()


def sync_kaggle_dataset() -> None:
    """Check the prepared dataset files the Kaggle bundle will pick up.

    The notebook reads files literally named `train.jsonl` / `holdout.jsonl`,
    and `build_kaggle_bundle.py` renames `train_augmented.jsonl` to `train.jsonl`
    as it assembles `kaggle_bundle/`. It reads them straight out of
    `data/ner_dataset/`, so nothing needs copying here.

    This function used to also copy them into `kaggle_upload/dataset/`. Those
    copies were never read by anything, but they made that folder *look* like a
    complete dataset -- and it carries the same `dataset-metadata.json` slug as
    the real bundle. Anyone running `kaggle datasets version -p
    kaggle_upload/dataset` would then overwrite the published dataset with a
    two-file version, stripping `input_turn2/`, `scripts/`, `data/terminology/`
    and `output/`. The kernel's distillation cell is wrapped in try/except, so
    it would not crash -- it would silently skip distillation and train
    curated-only, i.e. score ~34.4 instead of ~36.3. Only the metadata file
    belongs in that folder now.
    """
    src_train = ROOT / "data" / "ner_dataset" / "train_augmented.jsonl"
    src_holdout = ROOT / "data" / "ner_dataset" / "holdout.jsonl"
    if not src_train.exists() or not src_holdout.exists():
        raise SystemExit("Missing prepared dataset files; run the prepare stage first.")
    for stale in ("train.jsonl", "holdout.jsonl"):
        path = KAGGLE_DATASET_DIR / stale
        if path.exists():
            path.unlink()
            print(f"Removed stale unused copy {path}")
    print(f"Prepared dataset ready for build_kaggle_bundle.py: {src_train.name}, {src_holdout.name}")


def stage_dataset_version() -> None:
    """Version the *full* Kaggle bundle, not just train/holdout jsonl.

    Since 2026-07-25 the kernel doesn't only train -- it also runs the real
    `run_pipeline.py` from the bundled repo to produce output.zip, so the dataset
    has to carry `scripts/`, `data/terminology/*`, `output/` and `input_turn2/`
    too. Versioning the old thin `kaggle_upload/dataset/` folder instead would
    upload train.jsonl alone and the kernel's submission half would find nothing
    -- burning a full GPU run to produce a broken (or silently degraded)
    submission, which is exactly how the 31.89 regression happened.
    `build_kaggle_bundle.py` assembles and verifies it.
    """
    run([sys.executable, str(SCRIPTS / "build_kaggle_bundle.py"), "--out", str(BUNDLE_DIR)])
    if not (BUNDLE_DIR / "dataset-metadata.json").exists():
        raise SystemExit(f"{BUNDLE_DIR / 'dataset-metadata.json'} not found after building the bundle.")
    run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(BUNDLE_DIR),
            "-m",
            "sync ViettelRace bundle (scripts + terminology + labels + turn-2 inputs)",
            # WITHOUT THIS the upload silently contains only the bundle's root
            # files: `kaggle datasets version` defaults to --dir-mode skip, i.e.
            # it *ignores every subdirectory*. That produced a 142KB dataset
            # (train.jsonl + holdout.jsonl + metadata) instead of 43.8MB on
            # 2026-07-28, and the kernel then failed with no scripts/, no
            # terminology and no input_turn2/ to work with. It fails as a kernel
            # ERROR minutes later, not at upload time, so nothing points back here.
            "--dir-mode",
            "zip",
        ],
        env=KAGGLE_ENV,
    )


def kaggle_status(retries: int = 5, retry_delay: int = 15) -> str:
    # The status check runs dozens of times over a long poll loop, so a single
    # transient network/rate-limit hiccup shouldn't kill an in-progress (quota-
    # consuming) training run -- retry a few times before giving up.
    last_error = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", KERNEL_SLUG],
            cwd=ROOT, env=KAGGLE_ENV, capture_output=True, text=True,
        )
        if result.returncode == 0:
            match = re.search(r"KernelWorkerStatus\.([A-Z]+)", result.stdout)
            if match:
                return match.group(1)
            last_error = f"could not parse kernel status from: {result.stdout!r}"
        else:
            last_error = result.stderr.strip() or result.stdout.strip()

        if attempt < retries:
            print(f"  status check failed (attempt {attempt}/{retries}): {last_error} -- retrying in {retry_delay}s")
            time.sleep(retry_delay)

    raise RuntimeError(f"kaggle kernels status failed after {retries} attempts: {last_error}")


def stage_train(
    accelerator: str,
    poll_interval: int,
    timeout: int,
    skip_push: bool = False,
    skip_dataset_version: bool = False,
) -> None:
    if skip_push:
        print(f"Skipping push -- resuming poll for the currently running version of {KERNEL_SLUG}.")
    else:
        if not skip_dataset_version:
            stage_dataset_version()
        run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", accelerator],
            env=KAGGLE_ENV,
        )

    print(f"Polling {KERNEL_SLUG} every {poll_interval}s (timeout {timeout}s)...")
    deadline = time.time() + timeout
    status = ""
    while time.time() < deadline:
        status = kaggle_status()
        print(f"  status: {status}")
        if status in {"COMPLETE", "ERROR"} or status.startswith("CANCEL"):
            break
        time.sleep(poll_interval)
    else:
        raise SystemExit(f"Timed out after {timeout}s waiting for kernel to finish (last status: {status}).")

    if status != "COMPLETE":
        raise SystemExit(f"Kernel finished with status {status}, not COMPLETE -- check the run log on Kaggle.")

    download_dir = ROOT / ".kaggle_download"
    download_kernel_output(download_dir)

    # Newer kaggle CLI versions (2.x) auto-extract a single-zip kernel output
    # instead of leaving it as ner_model_export.zip: the bundle itself is saved
    # as output.zip and its contents land pre-extracted in a same-named folder.
    # Handle both that layout and the older raw-zip one.
    install_kernel_export(download_dir)


def stage_infer(input_dir: Path, pred_dir: Path, out_zip: Path) -> None:
    run([
        sys.executable,
        str(SCRIPTS / "run_pipeline.py"),
        "--input",
        str(input_dir),
        "--pred",
        str(pred_dir),
        "--drop-short-noise",
        "--add-terminology-entities",
        "--add-public-phrase-entities",
    ])
    run_check_submission(input_dir, pred_dir)
    stage_package(input_dir, pred_dir, out_zip)


def get_numbered_input_ids(input_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in input_dir.glob("*.txt"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)


def get_numbered_truth_ids() -> list[int]:
    ids: list[int] = []
    for path in TRUTH_DIR.glob("*.json"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)


def has_truth_files(input_dir: Path) -> bool:
    try:
        if input_dir.resolve() != INPUT_DIR.resolve():
            return False
    except OSError:
        return False
    input_ids = set(get_numbered_input_ids(input_dir))
    truth_ids = set(get_numbered_truth_ids())
    return bool(input_ids) and input_ids <= truth_ids


def run_check_submission(input_dir: Path, pred_dir: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPTS / "check_submission.py"),
        "--pred", str(pred_dir), "--input", str(input_dir),
    ]
    if has_truth_files(input_dir):
        cmd.extend(["--truth", str(TRUTH_DIR)])
    else:
        print("No matching public truth for this input folder; validating submission format only.")
    run(cmd)


def stage_package(input_dir: Path, pred_dir: Path, out_zip: Path) -> None:
    run([
        sys.executable, str(SCRIPTS / "package_submission.py"),
        "--pred", str(pred_dir), "--input", str(input_dir), "--out", str(out_zip),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "train", "infer", "package", "submit", "all"])
    parser.add_argument("--input", type=Path, default=INPUT_DIR, help="Input folder for infer/submit/package.")
    parser.add_argument("--pred", type=Path, default=PRED_DIR, help="Prediction folder for infer/submit/package.")
    parser.add_argument("--out", type=Path, default=ROOT / "output.zip", help="Zip path for infer/submit/package.")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle GPU type for the train stage.")
    parser.add_argument("--poll-interval", type=int, default=20, help="Seconds between train-stage status checks.")
    parser.add_argument("--timeout", type=int, default=3600, help="Max seconds to wait for training to finish.")
    parser.add_argument(
        "--aug-multiplier",
        type=int,
        default=1,
        help="Synthetic substitution variants per real training document for the prepare/all stages.",
    )
    parser.add_argument(
        "--assertion-docs",
        type=int,
        default=30,
        help="Synthetic assertion-cue documents generated for the prepare/all stages.",
    )
    parser.add_argument(
        "--train-all-labels",
        action="store_true",
        help=(
            "For prepare/all: train on all 100 labeled public files. Use this for a final "
            "model build after validation hyperparameters have been chosen."
        ),
    )
    parser.add_argument(
        "--skip-push", action="store_true",
        help="For the train/all stages: don't push a new kernel version, just poll the currently "
             "running one and download when it finishes. Use to resume after a crashed poll loop "
             "without spending another weekly GPU quota run.",
    )
    parser.add_argument(
        "--skip-dataset-version",
        action="store_true",
        help="For train/all: do not publish kaggle_upload/dataset before pushing the kernel.",
    )
    args = parser.parse_args()
    args.input = project_path(args.input)
    args.pred = project_path(args.pred)
    args.out = project_path(args.out)

    stages = {
        "prepare": [lambda: stage_prepare(args.aug_multiplier, args.assertion_docs, args.train_all_labels)],
        "train": [
            lambda: stage_train(
                args.accelerator,
                args.poll_interval,
                args.timeout,
                args.skip_push,
                args.skip_dataset_version,
            )
        ],
        "infer": [lambda: stage_infer(args.input, args.pred, args.out)],
        "package": [lambda: stage_package(args.input, args.pred, args.out)],
        "submit": [lambda: stage_infer(args.input, args.pred, args.out)],
        "all": [
            lambda: stage_prepare(args.aug_multiplier, args.assertion_docs, args.train_all_labels),
            lambda: stage_train(
                args.accelerator,
                args.poll_interval,
                args.timeout,
                args.skip_push,
                args.skip_dataset_version,
            ),
            lambda: stage_infer(args.input, args.pred, args.out),
        ],
    }
    for fn in stages[args.stage]:
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
