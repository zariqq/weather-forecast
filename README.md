# WeatherJEPA — short-term ERA5 profile forecasting with a JEPA backbone

Built on top of the existing `consts.py` / `pylidlight.py` DB layer. Copy
those two files into this folder (unchanged) before running anything here.

## Files

| File | Role |
|---|---|
| `dataset.py` | Pulls ERA5 profiles from `DB` onto a fixed pressure-level grid, builds windowed context/target samples (24h context by default, 3h & 6h targets), standardizes with train-only statistics. |
| `jepa_model.py` | `WeatherJEPA`: context encoder, EMA target encoder, horizon-conditioned predictor, forecast head. |
| `train.py` | Training loop + per-pressure-level RMSE evaluation over a caller-specified date range. |
| `visualize_embeddings.py` | PCA / t-SNE / cosine-similarity plots of extracted embeddings, for the latent-space analysis slide. |

## How this maps to the assignment

- **§1 Data prep** — `dataset.load_profile_grid` + `Era5WindowDataset` build
  the labeled time-series dataset from ERA5 reanalysis, on a shared
  pressure-level grid discovered from the archive itself.
- **§2 Architecture** — `WeatherJEPA` is a Transformer-based JEPA: a
  context encoder over the past 24h of profiles, an EMA target encoder
  (stop-gradient, avoids representation collapse) over the true future
  profile, and a horizon-conditioned predictor trained in *representation
  space* rather than raw value space. This is the "self-designed
  architecture with justified layer choice" the committee is grading —
  the writeup in the model docstring covers the *why*.
- **§3 Evaluation** — `train.evaluate` computes RMSE **per pressure
  level**, separately for the 3h and 6h horizons, in physical units
  (de-standardized), on whatever `--eval-start/--eval-end` range is
  passed at the CLI. That's the hook the committee needs to re-point
  evaluation at the closed test window.
- **§4 Embedding extraction** — `WeatherJEPA.forward` and
  `train.extract_embedding` both expose `z_hat`: the vector produced by
  the predictor, immediately before `ForecastHead` (the "final
  prognostic layer"). Its dimensionality is a single CLI flag
  (`--embed-dim`), so leaderboard tuning for the smallest usable
  embedding is just a sweep over that one number.

## Running it

```bash
python train.py \
  --db ./era5.sqlite --place "Lidar(Tomsk)" \
  --train-start 2009-01-01 --train-end 2021-12-31 \
  --eval-start  2022-01-01 --eval-end  2023-12-31 \
  --ctx-len 8 --embed-dim 64 --epochs 30

python visualize_embeddings.py \
  --db ./era5.sqlite --place "Lidar(Tomsk)" --ckpt weather_jepa.pt \
  --eval-start 2022-01-01 --eval-end 2023-12-31 --horizon 3
```

`train.py` writes `rmse_metrics.json` (per-level RMSE arrays for both
horizons, aligned with `pressure_levels_hPa`) and `weather_jepa.pt`
(model weights + normalization stats + the pressure grid, so the
checkpoint is self-contained for the committee's closed-set run).

## Assumptions you'll need to double check against the actual DB

1. **Place name.** `TZ` in `pylidlight.py` only lists `"Lidar(Tomsk)"`.
   If your ERA5 rows live under a different `Place.name`, either add it
   to `TZ` or ignore — ERA5 timestamps here are read straight off
   `Era5Data.timestamp` (UTC), not through the lidar-specific `TZ` map.
2. **Native cadence.** `STEP_SECONDS = 3*3600` assumes 3-hourly ERA5
   steps (standard for this kind of MERRA2/ERA5-style archive). If your
   archive is hourly, change `STEP_SECONDS` and `HORIZON_STEPS`
   accordingly (horizons should stay 3h/6h; step counts change).
3. **Pressure grid discovery.** The grid is read from the `PRESSURE`
   field of the *first* available timestep and reused for every other
   variable via `get_era5_measurements(..., altitude=list(grid))`
   (which interpolates onto it). If pressure levels are actually
   time-invariant in the archive (typical for reanalysis), this is
   exact rather than approximate.
4. **No network/DB in this sandbox**, so this code is syntax-checked
   (`python -m py_compile`) but not execution-tested against real data
   or a real `torch` install — sanity-check shapes on a small date
   range before a full training run.

## Tuning knobs likely worth sweeping for the leaderboard

- `--embed-dim`: leaderboard rewards small embeddings; watch the
  RMSE/dim tradeoff in `rmse_metrics.json` across a sweep (e.g. 16/32/64/128).
- `--jepa-weight` / `--forecast-weight`: too much forecast weight turns
  this into a plain supervised model (embedding becomes less
  interesting); too much JEPA weight can undertrain the decode head.
- `--ctx-len`: longer context helps the diurnal cycle but costs
  quadratic attention time; 8 steps (24h) is a reasonable default start.
