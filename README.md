# WeatherJEPA

Short-term ERA5 absolute humidity forecasting with a Joint-Embedding Predictive Architecture (JEPA). 3h and 6h horizons, 37 pressure levels.

## Files

| File                      | Purpose                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `train.py`                | Training loop + mid-training eval + best checkpoint saving                                                |
| `eval.py`                 | Standalone inference: RMSE per pressure level + embedding extraction (for the committee)                  |
| `jepa_model.py`           | Transformer-based JEPA: context encoder, EMA target encoder, horizon-conditioned predictor, forecast head |
| `dataset.py`              | ERA5 data loading: pressure grid discovery, windowed context/target samples, standardization              |
| `visualize_embeddings.py` | PCA / t-SNE / cosine-similarity plots of extracted embeddings                                             |
| `download_data.py`        | Downloads the SQLite DB from Google Drive                                                                 |
| `dblayer/`                | DB access layer (`consts.py`, `pylidlight.py` -- provided, not mine)                                       |

## Quick start

```bash
# Train
python train.py \
  --db ./era5.db --place "Lidar(Tomsk)" \
  --train-start 2009-01-01 --train-end 2021-12-31 \
  --eval-start  2022-01-01 --eval-end  2023-12-31 \
  --embed-dim 64 --epochs 30

# Evaluate on any date range (no training)
python eval.py \
  --db ./era5.db --place "Lidar(Tomsk)" \
  --ckpt weather_jepa.pt \
  --eval-start 2024-01-01 --eval-end 2024-12-31 \
  --save-embeddings embeddings.npz
```

## Architecture

Context (past 8h of T/RH/AH/wind profiles) -> Transformer encoder -> pooled embedding `z_ctx`. Predictor `g_phi` maps `z_ctx` -> `z_hat` (conditioned on horizon 3h/6h via learned embedding). Forecast head decodes `z_hat` -> absolute humidity profile.

Target encoder is an EMA copy of the context encoder (no gradients). JEPA loss: `||z_hat - stopgrad(z_target)||^2` in representation space. Forecast loss: MSE on the decoded humidity profile.

The extracted "hidden context vector" is `z_hat` -- right before the forecast head.

## Key CLI flags

| Flag                                  | What                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| `--embed-dim`                         | Embedding size. Smaller = better leaderboard. Sweep 16/32/64                  |
| `--jepa-weight` / `--forecast-weight` | Loss ratio. If embeddings collapse (cosine > 0.9), raise jepa, lower forecast |
| `--eval-every N`                      | Eval frequency during training                                                |
| `--ckpt-best`                         | Auto-saves when eval RMSE improves                                            |

## Data

ERA5 reanalysis 2009–2023, Lidar(Tomsk). 123,769 hourly timesteps, 37 pressure levels (0–458 hPa). Variables: temperature, relative humidity, absolute humidity, wind direction/speed. Target: absolute humidity.

Native cadence is auto-detected (hourly in this DB). Horizons: 3 steps (3h) and 6 steps (6h).

## Checkpoint format

Self-contained `.pt` file:

```
model_state, mean, std, pressure_levels, args, epoch, eval_rmse_mean
```

Everything needed for inference without access to the original training code.
