#!/usr/bin/env python3
"""Build caches and run P0 multimodal crop-yield baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_baseline import (
    CACHE_ROOT,
    CROPS,
    MODES,
    RESULTS_ROOT,
    build_crop_cache,
    run_gru,
    run_hgb,
    run_historical_baselines,
    write_csv,
    write_summary,
)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_optional_limit(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("cache", "historical", "hgb", "gru", "smoke", "p0", "summary"),
        required=True,
    )
    parser.add_argument("--crops", default="maize,rice,soybean,wheat")
    parser.add_argument("--modes", default="C0,C1,C2,C3,C4,C5,C6,C7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--area-threshold", type=float, default=100.0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hgb-max-iter", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    crops = parse_csv(args.crops)
    modes = parse_csv(args.modes)
    invalid_crops = sorted(set(crops) - set(CROPS))
    invalid_modes = sorted(set(modes) - set(MODES))
    if invalid_crops:
        raise SystemExit(f"Unsupported crops: {invalid_crops}")
    if invalid_modes:
        raise SystemExit(f"Unsupported modes: {invalid_modes}")
    if args.output_root != RESULTS_ROOT:
        raise SystemExit(f"Custom output roots are not supported yet; expected {RESULTS_ROOT}")
    if args.area_threshold != 100.0 and args.stage != "cache":
        raise SystemExit("The current experiment cache loader expects the frozen 100 ha threshold.")

    max_train_samples = parse_optional_limit(args.max_train_samples)
    max_eval_samples = parse_optional_limit(args.max_eval_samples)
    print(f"Cache root: {CACHE_ROOT}", flush=True)
    print(f"Results root: {RESULTS_ROOT}", flush=True)

    if args.stage in {"cache", "smoke", "p0"}:
        for crop in crops:
            path = build_crop_cache(
                crop,
                area_threshold=args.area_threshold,
                force=args.force_cache,
            )
            print(f"[CACHE] crop={crop} path={path}", flush=True)
        if args.stage == "cache":
            return

    rows: list[dict] = []
    if args.stage in {"historical", "p0"}:
        for crop in crops:
            rows.extend(run_historical_baselines(crop))

    if args.stage == "hgb":
        for crop in crops:
            for mode in modes:
                rows.append(
                    run_hgb(
                        crop=crop,
                        mode=mode,
                        seed=args.seed,
                        max_iter=args.hgb_max_iter,
                        max_train_samples=max_train_samples,
                        max_eval_samples=max_eval_samples,
                    )
                )

    if args.stage == "gru":
        for crop in crops:
            for mode in modes:
                rows.append(
                    run_gru(
                        crop=crop,
                        mode=mode,
                        seed=args.seed,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        max_train_samples=max_train_samples,
                        max_eval_samples=max_eval_samples,
                        device_name=args.device,
                    )
                )

    if args.stage == "smoke":
        smoke_modes = [mode for mode in ("C2", "C3", "C4", "C7") if mode in modes]
        for crop in crops:
            for mode in smoke_modes:
                rows.append(
                    run_gru(
                        crop=crop,
                        mode=mode,
                        seed=args.seed,
                        max_epochs=min(args.max_epochs, 5),
                        patience=min(args.patience, 3),
                        batch_size=args.batch_size,
                        max_train_samples=max_train_samples or 20_000,
                        max_eval_samples=max_eval_samples or 5_000,
                        device_name=args.device,
                    )
                )

    if args.stage == "p0":
        hgb_modes = [mode for mode in ("C2", "C3", "C4", "C7") if mode in modes]
        for crop in crops:
            for mode in hgb_modes:
                rows.append(
                    run_hgb(
                        crop=crop,
                        mode=mode,
                        seed=args.seed,
                        max_iter=args.hgb_max_iter,
                        max_train_samples=max_train_samples,
                        max_eval_samples=max_eval_samples,
                    )
                )
            for mode in modes:
                rows.append(
                    run_gru(
                        crop=crop,
                        mode=mode,
                        seed=args.seed,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        max_train_samples=max_train_samples,
                        max_eval_samples=max_eval_samples,
                        device_name=args.device,
                    )
                )

    if rows:
        write_csv(rows, RESULTS_ROOT / "summary" / f"{args.stage}_latest.csv")
    summary_path = write_summary()
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
