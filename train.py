"""
train.py

Trains WeatherJEPA and reports RMSE per pressure level at the 3h and 6h
horizons over a caller-specified date range, e.g.:

    python train.py --db ./era5.sqlite --place "Lidar(Tomsk)" \
        --train-start 2009-01-01 --train-end 2021-12-31 \
        --eval-start  2022-01-01 --eval-end  2023-12-31 \
        --epochs 30 --embed-dim 64

The --eval-start/--eval-end pair is exactly the "temporal range" knob
the assignment asks for so the committee can re-point evaluation at
their closed test window without touching the code.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_datasets, HORIZON_HOURS, TARGET_CHANNEL_IDX
from download_data import ensure_db
from jepa_model import WeatherJEPA, jepa_and_forecast_loss


def collate(batch):
    context = torch.stack([b["context"] for b in batch])
    future_full = {
        h: torch.stack([b["future_full"][h] for b in batch]) for h in HORIZON_HOURS
    }
    targets = {h: torch.stack([b["targets"][h] for b in batch]) for h in HORIZON_HOURS}
    raw_targets = {
        h: torch.stack([b["raw_targets"][h] for b in batch]) for h in HORIZON_HOURS
    }
    return context, future_full, targets, raw_targets


@torch.no_grad()
def evaluate(model, loader, mean, std, device):
    """Returns RMSE per pressure level (physical units, kg/kg) for each horizon."""
    model.eval()
    sq_err = {h: None for h in HORIZON_HOURS}
    count = 0
    tgt_mean = mean[TARGET_CHANNEL_IDX]
    tgt_std = std[TARGET_CHANNEL_IDX]

    for context, future_full, _targets, raw_targets in loader:
        context = context.to(device)
        future_full = {h: v.to(device) for h, v in future_full.items()}
        out = model(context, future_full)

        for h in HORIZON_HOURS:
            pred_std = out["per_horizon"][h]["forecast"].cpu().numpy()
            pred_phys = pred_std * tgt_std + tgt_mean  # de-standardize
            true_phys = raw_targets[h].numpy()
            se = (pred_phys - true_phys) ** 2  # [B, L]
            sq_err[h] = (
                se.sum(axis=0) if sq_err[h] is None else sq_err[h] + se.sum(axis=0)
            )
        count += context.size(0)

    rmse = {h: np.sqrt(sq_err[h] / max(count, 1)) for h in HORIZON_HOURS}
    model.train()
    return rmse


def extract_embedding(
    model: WeatherJEPA, context: torch.Tensor, horizon_hours: int, device
):
    """
    Convenience function for the presentation's 'methodology' slide:
    returns the exact vector extracted right before the forecast head
    (the final prognostic layer) for a given horizon.
    """
    from jepa_model import HORIZON_TO_IDX

    model.eval()
    with torch.no_grad():
        context = context.to(device)
        z_ctx = model.encode_context(context)
        idx = torch.full(
            (z_ctx.size(0),),
            HORIZON_TO_IDX[horizon_hours],
            dtype=torch.long,
            device=device,
        )
        z_hat = model.predictor(
            z_ctx, idx
        )  # <-- this is the extracted hidden/context vector
    model.train()
    return z_hat.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--eval-start", required=True)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--ctx-len", type=int, default=8)  # 8*3h = 24h lookback
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--jepa-weight", type=float, default=1.0)
    ap.add_argument("--forecast-weight", type=float, default=1.0)
    ap.add_argument("--ckpt", default="weather_jepa.pt")
    ap.add_argument("--metrics-out", default="rmse_metrics.json")
    args = ap.parse_args()

    args.db = ensure_db(args.db)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU via Metal
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    train_ds, eval_ds, grid = build_datasets(
        args.db,
        args.place,
        args.train_start,
        args.train_end,
        args.eval_start,
        args.eval_end,
        ctx_len=args.ctx_len,
    )
    print(
        f"train samples: {len(train_ds)}, eval samples: {len(eval_ds)}, "
        f"levels: {len(grid.pressure_levels)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    n_channels = train_ds.grid.values.shape[-1]
    n_levels = len(grid.pressure_levels)

    model = WeatherJEPA(
        n_channels=n_channels,
        n_levels=n_levels,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_ctx_len=args.ctx_len,
    ).to(device)

    opt = torch.optim.AdamW(
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.forecast_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    loss_history = []
    for epoch in range(args.epochs):
        running = {"jepa": 0.0, "forecast": 0.0}
        n_batches = 0
        for context, future_full, targets, _raw in train_loader:
            context = context.to(device)
            future_full = {h: v.to(device) for h, v in future_full.items()}
            targets = {h: v.to(device) for h, v in targets.items()}

            out = model(context, future_full)
            loss, parts = jepa_and_forecast_loss(
                out, targets, args.jepa_weight, args.forecast_weight
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target_encoder()  # EMA step, after every optimizer step

            running["jepa"] += parts["jepa"].item()
            running["forecast"] += parts["forecast"].item()
            n_batches += 1

        avg_jepa = running["jepa"] / max(n_batches, 1)
        avg_fcst = running["forecast"] / max(n_batches, 1)
        loss_history.append(
            {"epoch": epoch, "jepa_loss": avg_jepa, "forecast_loss": avg_fcst}
        )
        print(
            f"epoch {epoch:03d}  jepa_loss={avg_jepa:.5f}  forecast_loss={avg_fcst:.5f}"
        )

    rmse = evaluate(model, eval_loader, eval_ds.mean, eval_ds.std, device)
    results = {
        "pressure_levels_hPa": grid.pressure_levels.tolist(),
        "embed_dim": args.embed_dim,
        "eval_range": [args.eval_start, args.eval_end],
        "rmse_by_horizon": {str(h): rmse[h].tolist() for h in HORIZON_HOURS},
        "rmse_mean_by_horizon": {str(h): float(rmse[h].mean()) for h in HORIZON_HOURS},
        "loss_history": loss_history,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": train_ds.mean,
            "std": train_ds.std,
            "pressure_levels": grid.pressure_levels,
            "args": vars(args),
        },
        args.ckpt,
    )

    print("Mean RMSE by horizon (kg/kg):", results["rmse_mean_by_horizon"])
    print(f"Saved checkpoint to {args.ckpt}, metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
