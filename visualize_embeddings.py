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
from torch.utils.data import DataLoader

from dataset import DB, Era5WindowDataset, load_profile_grid
from download_data import ensure_db
from jepa_model import WeatherJEPA, HORIZON_TO_IDX
from train import collate


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
    ap.add_argument("--max-samples", type=int, default=1500)
    ap.add_argument("--out-prefix", default="embeddings")
    args = ap.parse_args()

    args.db = ensure_db(args.db)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)

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
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

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

    embeds, months = [], []
    horizon_idx_val = HORIZON_TO_IDX[args.horizon]

    with torch.no_grad():
        sample_i = 0
        for context, _future_full, _targets, _raw in loader:
            if sample_i >= args.max_samples:
                break
            context = context.to(device)
            z_ctx = model.encode_context(context)
            idx = torch.full(
                (z_ctx.size(0),), horizon_idx_val, dtype=torch.long, device=device
            )
            z_hat = model.predictor(z_ctx, idx)
            embeds.append(z_hat.cpu().numpy())
            sample_i += context.size(0)

    embeds = np.concatenate(embeds, axis=0)[: args.max_samples]

    # Month-of-year color coding, purely for the plot (not used by the model)
    sample_timestamps = grid.timestamps[ds.index][: len(embeds)]
    months = np.array(
        [
            datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).month
            for t in sample_timestamps
        ]
    )

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

    # --- cosine similarity heatmap over a subset ---
    subset = embeds[: min(200, len(embeds))]
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
