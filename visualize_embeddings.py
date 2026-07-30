"""
visualize_embeddings.py

Loads a trained checkpoint, extracts the context/predicted embedding
(z_hat, right before the forecast head) for every sample in an
evaluation window, and produces:

  1. A PCA scatter plot, colored by month-of-year (to visually check
     the embedding organizes by season rather than collapsing).
  2. A t-SNE scatter plot, same coloring.
  3. A cosine-similarity heatmap over a random subset of samples, to
     check embeddings aren't all pointing the same direction (i.e.
     the "constant dying layer" failure mode called out in the brief).

Usage:
    python visualize_embeddings.py --db ./era5.sqlite --place "Lidar(Tomsk)" \
        --ckpt weather_jepa.pt --eval-start 2022-01-01 --eval-end 2023-12-31
"""

from __future__ import annotations

import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset import DB, Era5WindowDataset, load_profile_grid
from download_data import ensure_db
from jepa_model import WeatherJEPA, HORIZON_TO_IDX


def to_ts(s):
    return int(
        datetime.datetime.fromisoformat(s)
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-start", required=True)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--horizon", type=int, default=3, choices=[3, 6])
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="Randomly subsample across full eval range")
    ap.add_argument("--no-shuffle", action="store_false", dest="shuffle",
                    help="Take first N samples (fast debug)")
    ap.add_argument("--out-prefix", default="embeddings")
    args = ap.parse_args()

    args.db = ensure_db(args.db)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    with DB(args.db) as db:
        place = db.get_place(args.place)
        grid = load_profile_grid(db, place)

    ds = Era5WindowDataset(
        grid,
        ctx_len=ckpt["args"]["ctx_len"],
        start_ts=to_ts(args.eval_start),
        end_ts=to_ts(args.eval_end),
        mean=ckpt["mean"],
        std=ckpt["std"],
    )
    n_channels = grid.values.shape[-1]
    n_levels = len(grid.pressure_levels)
    model = WeatherJEPA(
        n_channels=n_channels,
        n_levels=n_levels,
        embed_dim=ckpt["args"]["embed_dim"],
        n_heads=ckpt["args"]["n_heads"],
        n_layers=ckpt["args"]["n_layers"],
        max_ctx_len=ckpt["args"]["ctx_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Choose which samples to extract. If shuffle, randomly sub-sample
    # across the full eval range; otherwise take the first N (fast debug).
    n_total = len(ds)
    n_use = min(args.max_samples, n_total)
    if args.shuffle:
        rng = np.random.default_rng(42)
        sample_indices = sorted(rng.choice(n_total, size=n_use, replace=False))
    else:
        sample_indices = list(range(n_use))

    embeds, months = [], []
    horizon_idx_val = HORIZON_TO_IDX[args.horizon]

    with torch.no_grad():
        for idx in range(0, n_use, 64):
            batch_indices = sample_indices[idx: idx + 64]
            batch = [ds[i] for i in batch_indices]
            context = torch.stack([b["context"] for b in batch]).to(device)
            z_ctx = model.encode_context(context)
            z_hat = model.predictor(
                z_ctx,
                torch.full((len(batch_indices),), horizon_idx_val, dtype=torch.long, device=device),
            )
            embeds.append(z_hat.cpu().numpy())
            for i in batch_indices:
                t = ds.index[i]
                months.append(
                    datetime.datetime.fromtimestamp(
                        grid.timestamps[t + ds.horizon_steps[args.horizon]],
                        tz=datetime.timezone.utc,
                    ).month
                )

    embeds = np.concatenate(embeds, axis=0)[:n_use]
    months = np.array(months)

    # --- PCA ---
    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(embeds)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(pca_xy[:, 0], pca_xy[:, 1], c=months, cmap="twilight", s=8)
    plt.colorbar(sc, label="month")
    plt.title(f"PCA of context embeddings (horizon={args.horizon}h)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_pca.png", dpi=150)
    plt.close()

    # --- t-SNE ---
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
    tsne_xy = tsne.fit_transform(embeds)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(tsne_xy[:, 0], tsne_xy[:, 1], c=months, cmap="twilight", s=8)
    plt.colorbar(sc, label="month")
    plt.title(f"t-SNE of context embeddings (horizon={args.horizon}h)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_tsne.png", dpi=150)
    plt.close()

    # --- cosine similarity heatmap over a random subset ---
    rng = np.random.default_rng(123)
    heatmap_idx = rng.choice(len(embeds), size=min(200, len(embeds)), replace=False)
    subset = embeds[heatmap_idx]
    norm = subset / (np.linalg.norm(subset, axis=1, keepdims=True) + 1e-8)
    cos_sim = norm @ norm.T
    plt.figure(figsize=(6, 5))
    plt.imshow(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="cosine similarity")
    plt.title("Cosine similarity between embeddings\n(should NOT be ~1.0 everywhere)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_cosine_sim.png", dpi=150)
    plt.close()

    print(f"embedding dim: {embeds.shape[1]}, n samples: {len(embeds)}")
    print(
        f"mean off-diagonal cosine similarity: "
        f"{(cos_sim.sum() - np.trace(cos_sim)) / (cos_sim.size - len(cos_sim)):.4f}"
    )
    print(
        f"Saved: {args.out_prefix}_pca.png, {args.out_prefix}_tsne.png, {args.out_prefix}_cosine_sim.png"
    )


if __name__ == "__main__":
    main()
