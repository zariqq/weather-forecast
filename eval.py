"""
eval.py

Standalone evaluation script — loads a trained checkpoint and computes
RMSE per pressure level at 3h and 6h horizons for a caller-specified
date range. Also extracts embeddings from right before the forecast head.

This is the script the committee runs on the closed test set.
No training happens here.

Usage:
    python eval.py --db ./era5.sqlite --place "Lidar(Tomsk)" \
        --ckpt weather_jepa.pt \
        --eval-start 2022-01-01 --eval-end 2023-12-31

Output:
    eval_metrics.json  — per-level RMSE for both horizons
    embeddings.npz     — extracted z_hat vectors + timestamps (if --save-embeddings)
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dblayer.pylidlight import DB
from dataset import (
    Era5WindowDataset,
    HORIZON_HOURS,
    TARGET_CHANNEL_IDX,
    load_profile_grid,
)
from download_data import ensure_db
from jepa_model import WeatherJEPA, HORIZON_TO_IDX
from train import collate, evaluate


def to_ts(s: str) -> int:
    return int(
        datetime.datetime.fromisoformat(s)
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def extract_all_embeddings(model, loader, horizon_hours: int, device):
    """
    Extract z_hat (the embedding BEFORE the forecast head) for all
    samples in the loader. Returns (embeddings, timestamps).
    """
    model.eval()
    embeds, timestamps = [], []
    horizon_idx = torch.tensor([HORIZON_TO_IDX[horizon_hours]], dtype=torch.long)

    with torch.no_grad():
        for context, _, _, _ in loader:
            context = context.to(device)
            z_ctx = model.encode_context(context)
            idx = horizon_idx.expand(z_ctx.size(0)).to(device)
            z_hat = model.predictor(z_ctx, idx)
            embeds.append(z_hat.cpu().numpy())

    model.train()
    return np.concatenate(embeds, axis=0)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a trained WeatherJEPA model on a date range"
    )
    ap.add_argument("--db", required=True, help="Path to SQLite database")
    ap.add_argument("--place", required=True, help="Place name, e.g. 'Lidar(Tomsk)'")
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument(
        "--eval-start", required=True, help="Start date (ISO), e.g. 2022-01-01"
    )
    ap.add_argument(
        "--eval-end", required=True, help="End date (ISO), e.g. 2023-12-31"
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--metrics-out",
        default="eval_metrics.json",
        help="Where to save per-level RMSE JSON",
    )
    ap.add_argument(
        "--save-embeddings",
        default=None,
        help="If set, save embeddings to this .npz file",
    )
    ap.add_argument(
        "--embedding-horizon",
        type=int,
        default=3,
        choices=[3, 6],
        help="Which horizon to extract embeddings for",
    )
    args = ap.parse_args()

    # --- DB & checkpoint ---
    args.db = ensure_db(args.db)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    print(f"checkpoint embed_dim={ckpt_args['embed_dim']}, "
          f"ctx_len={ckpt_args['ctx_len']}, "
          f"epoch={ckpt.get('epoch', '?')}")

    # --- Build eval dataset ---
    with DB(args.db) as db:
        place = db.get_place(args.place)
        grid = load_profile_grid(db, place)

    eval_ds = Era5WindowDataset(
        grid,
        ctx_len=ckpt_args["ctx_len"],
        start_ts=to_ts(args.eval_start),
        end_ts=to_ts(args.eval_end),
        mean=ckpt["mean"],
        std=ckpt["std"],
    )

    if len(eval_ds) == 0:
        print(
            f"ERROR: 0 eval samples in [{args.eval_start}, {args.eval_end}]. "
            f"Check that the date range overlaps the grid's span "
            f"({datetime.datetime.fromtimestamp(grid.timestamps[0], tz=datetime.timezone.utc).date()} "
            f"to {datetime.datetime.fromtimestamp(grid.timestamps[-1], tz=datetime.timezone.utc).date()}) "
            f"and fits inside ctx_len={ckpt_args['ctx_len']} + horizons."
        )
        sys.exit(1)

    print(
        f"eval samples: {len(eval_ds)}, "
        f"pressure levels: {len(grid.pressure_levels)}"
    )

    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    # --- Build model & load weights ---
    n_channels = grid.values.shape[-1]
    n_levels = len(grid.pressure_levels)

    model = WeatherJEPA(
        n_channels=n_channels,
        n_levels=n_levels,
        embed_dim=ckpt_args["embed_dim"],
        n_heads=ckpt_args["n_heads"],
        n_layers=ckpt_args["n_layers"],
        max_ctx_len=ckpt_args["ctx_len"],
        ema_momentum=ckpt_args.get("ema_momentum", 0.996),
        ctx_mask_ratio=ckpt_args.get("ctx_mask_ratio", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Evaluate ---
    rmse = evaluate(model, eval_loader, ckpt["mean"], ckpt["std"], device)

    results = {
        "checkpoint": args.ckpt,
        "place": args.place,
        "eval_range": [args.eval_start, args.eval_end],
        "embed_dim": ckpt_args["embed_dim"],
        "pressure_levels_hPa": grid.pressure_levels.tolist(),
        "rmse_by_horizon": {str(h): rmse[h].tolist() for h in HORIZON_HOURS},
        "rmse_mean_by_horizon": {
            str(h): float(rmse[h].mean()) for h in HORIZON_HOURS
        },
    }

    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2)

    # --- Print summary ---
    print(f"\n=== Evaluation: {args.eval_start} → {args.eval_end} ===")
    print(f"Place: {args.place}  |  embed_dim: {ckpt_args['embed_dim']}")
    print(f"{'Level (hPa)':>12s}  {'RMSE_3h':>10s}  {'RMSE_6h':>10s}")
    for i, p in enumerate(grid.pressure_levels):
        print(f"{p:12.2f}  {rmse[3][i]:10.6f}  {rmse[6][i]:10.6f}")
    print(f"{'MEAN':>12s}  {rmse[3].mean():10.6f}  {rmse[6].mean():10.6f}")
    print(f"\nSaved metrics to {args.metrics_out}")

    # --- Extract embeddings (optional) ---
    if args.save_embeddings:
        print(f"Extracting embeddings for horizon={args.embedding_horizon}h ...")
        embeds = extract_all_embeddings(
            model, eval_loader, args.embedding_horizon, device
        )
        np.savez_compressed(
            args.save_embeddings,
            embeddings=embeds,
            timestamps=grid.timestamps[eval_ds.index[: len(embeds)]],
            pressure_levels=grid.pressure_levels,
        )
        print(
            f"Saved {len(embeds)} embeddings (dim={embeds.shape[1]}) "
            f"to {args.save_embeddings}"
        )


if __name__ == "__main__":
    main()
