#!/usr/bin/env python3
"""Run the paper's raw-product processing stages in a separate workspace.

Raw inputs must be acquired under provider terms first. No network downloads,
model checkpoints, or historical prediction files are needed by these stages.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
STAGES = ("gdhy", "weather", "calendar", "lai", "ndvi", "gpp")


def initialize(workspace):
    source = REPO / "workflow"
    # Never overwrite a user's workspace code or silently keep stale scripts.
    for file in sorted(source.rglob("*.py")):
        target = workspace / file.relative_to(source)
        if target.exists() and target.read_bytes() != file.read_bytes():
            raise ValueError(f"Workspace code differs; use a new workspace: {target}")
    for file in sorted(source.rglob("*.py")):
        target = workspace / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(file, target)


def commands(stage):
    python = sys.executable
    if stage == "gdhy":
        return [[python, "scripts/convert_gdhy_to_npy_lon180.py", "--source",
                 "Data/GDHY/gdhy_v1.2_v1.3_20190128", "--output",
                 "Data/GDHY/gdhy_v1.2_v1.3_20190128_npy_lon180"]]
    if stage == "weather":
        return [[python, "scripts/aggregate_era5land_monthly_to_gdhy_npy.py"],
                [python, "scripts/split_era5land_0p5deg_npy_by_variable.py"]]
    if stage == "calendar":
        return [[python, "scripts/build_crop_growing_season_dataset.py"]]
    if stage == "lai":
        return [[python, "scripts/process_glass_lai_avhrr_to_growing_season.py"]]
    if stage == "ndvi":
        return [[python, "scripts/prepare_pku_ndvi.py", "--process"]]
    if stage == "gpp":
        # Support auditing requires cohorts built later; raster processing does not.
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import prepare_reclue_monthly_gpp as m; "
            "paths=[(y, (m.SAMPLE if y==1982 else m.RAW)/f'{y}.zip') for y in m.YEARS]; "
            "missing=[str(p) for _,p in paths if not p.is_file()]; "
            "assert not missing, f'Missing GPP archives: {missing}'; "
            "[m.process(y,p,audit_cohort=False) for y,p in paths]"
        )
        return [[python, "-c", code]]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stage", choices=("init", "all", *STAGES), default="init")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if workspace == REPO or workspace == REPO / "workflow":
        parser.error("Use a separate workspace, not the release source directory")
    stages = STAGES if args.stage == "all" else (() if args.stage == "init" else (args.stage,))
    if args.dry_run:
        for stage in stages:
            print(stage, commands(stage))
        return
    workspace.mkdir(parents=True, exist_ok=True)
    initialize(workspace)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for stage in stages:
        print(f"Processing {stage} in {workspace}", flush=True)
        for command in commands(stage):
            subprocess.run(command, cwd=workspace, env=env, check=True)
        if stage == "gpp":
            subprocess.run([sys.executable, str(REPO / 'scripts/finalize_gpp.py'),
                            '--workspace', str(workspace)], cwd=workspace, env=env, check=True)


if __name__ == "__main__":
    main()
