"""
dataset.py

Builds a windowed, multi-level, multi-variable ERA5 dataset out of the
existing `pylidlight.DB` accessor, for training a JEPA-style short-term
forecaster of absolute humidity profiles.

Design notes
------------
ERA5 in this DB is stored per `Place` as a series of `Era5Data` rows
(one per timestamp), each with per-altitude measurements reachable via
`DB.get_era5_measurements`. The competition asks for forecasting on
*pressure* levels, so we:

 1. Discover a fixed pressure-level grid once (from the PRESSURE field
    of the first available timestep).
 2. For every timestep and every variable of interest, pull values on
    that fixed grid (letting `get_era5_measurements` interpolate them
    onto it — this also means we get a consistent tensor shape even
    if the raw grid wiggles slightly between timesteps).
 3. Stack timesteps into an array of shape [T, L, C] (time, levels,
    channels) and slide a context window of length `ctx_len` (3-hourly
    steps) across it, pairing each context window with the profile
    1 step ahead (3h horizon) and 2 steps ahead (6h horizon).

Samples that straddle a data gap (missing timesteps, so the real time
delta isn't ~3h/6h within tolerance) are dropped rather than silently
treated as valid — ERA5-reanalysis-style data is regular, but archived
subsets can have holes.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from dblayer.consts import Measure
from dblayer.pylidlight import DB, Place

HORIZON_HOURS = (3, 6)
TOLERANCE_FRACTION = 0.25  # allowed jitter, as a fraction of the detected native step
TOLERANCE_FLOOR_SECONDS = 5 * 60  # ...but never tighter than this

# Variables fed into the model. Absolute humidity is both an input
# channel and the sole forecast target (per the assignment).
INPUT_MEASURES = [
    Measure.TEMPERATURE,
    Measure.REL_HUMIDITY,
    Measure.ABS_HUMIDITY,
    Measure.WIND_DIRECTION,
    Measure.WIND_SPEED,
]
TARGET_MEASURE = Measure.ABS_HUMIDITY
TARGET_CHANNEL_IDX = INPUT_MEASURES.index(TARGET_MEASURE)


@dataclass
class ProfileGrid:
    place: Place
    pressure_levels: np.ndarray  # [L] hPa, descending
    timestamps: np.ndarray  # [T] unix seconds
    values: np.ndarray  # [T, L, C]
    step_seconds: int  # native cadence, auto-detected from timestamps


def _discover_pressure_grid(db: DB, place: Place, era5_series) -> np.ndarray:
    """Use the PRESSURE field of the first timestep as the canonical
    vertical grid every other variable gets interpolated onto."""
    for d in era5_series:
        alts, pressures = db.get_era5_measurements(
            d, Measure.PRESSURE, altitude=1_000_000  # effectively "all"
        )
        if len(pressures) >= 4:
            order = np.argsort(np.asarray(pressures))[::-1]  # descending hPa
            return np.asarray(alts)[order]
    raise RuntimeError(
        "Could not discover a pressure/altitude grid for %s" % place.name
    )


def load_profile_grid(db: DB, place: Place) -> ProfileGrid:
    """Pull the full ERA5 series for `place` onto a fixed vertical grid."""
    era5_series = db.get_era5_data(place)
    if not era5_series:
        raise RuntimeError("No ERA5 data for place %s" % place.name)

    grid = _discover_pressure_grid(db, place, era5_series)
    L = len(grid)
    C = len(INPUT_MEASURES)

    timestamps = np.empty(len(era5_series), dtype=np.int64)
    values = np.full((len(era5_series), L, C), np.nan, dtype=np.float32)

    for t_idx, d in enumerate(era5_series):
        timestamps[t_idx] = d.timestamp
        for c_idx, measure in enumerate(INPUT_MEASURES):
            _, vals = db.get_era5_measurements(d, measure, altitude=list(grid))
            values[t_idx, :, c_idx] = np.asarray(vals, dtype=np.float32)

    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    step_seconds = int(np.median(diffs)) if len(diffs) else 3 * 3600
    gap_frac = float((diffs > step_seconds * 1.5).mean()) if len(diffs) else 0.0
    print(
        f"[dataset] {place.name}: {len(timestamps)} timesteps, "
        f"detected native cadence = {step_seconds}s ({step_seconds/3600:.2f}h), "
        f"{gap_frac:.1%} of intervals look like gaps"
    )

    return ProfileGrid(
        place=place,
        pressure_levels=grid,
        timestamps=timestamps,
        values=values,
        step_seconds=step_seconds,
    )


def _level_stats(values: np.ndarray):
    """Per-level, per-channel mean/std over time axis.

    Returns arrays of shape (L, C). Each pressure level is standardized
    independently so upper-atmosphere structure (where humidity is ~1e-6 kg/kg)
    carries equal weight in the loss to surface structure (~0.01 kg/kg).
    """
    mean = np.nanmean(values, axis=0)   # (L, C)
    std = np.nanstd(values, axis=0)     # (L, C)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class Era5WindowDataset(Dataset):
    """
    Each sample:
      context: [ctx_len, L, C] standardized, past `ctx_len` profiles
      targets: dict horizon_hours -> [L] standardized ABS_HUMIDITY profile
      raw_targets: dict horizon_hours -> [L] *unstandardized* ABS_HUMIDITY,
                   kept around so RMSE can be reported in physical units
      pressure_levels: [L]
    """

    def __init__(
        self,
        grid: ProfileGrid,
        ctx_len: int = 8,  # 8 * 3h = 24h of context
        start_ts: int | None = None,
        end_ts: int | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.grid = grid
        self.ctx_len = ctx_len
        self.mean, self.std = (
            (mean, std) if mean is not None else _level_stats(grid.values)
        )

        # Derive step counts for each horizon from THIS grid's own detected
        # cadence (hourly, 3-hourly, whatever the archive actually is) —
        # never assume a fixed cadence, since ERA5 is natively hourly while
        # the MERRA2 tables in this same DB are 3-hourly.
        self.horizon_steps: dict[int, int] = {}
        for hours in HORIZON_HOURS:
            exact_steps = hours * 3600 / grid.step_seconds
            steps = round(exact_steps)
            if steps < 1 or abs(exact_steps - steps) > 0.05:
                raise ValueError(
                    f"{hours}h horizon is not a clean multiple of the detected "
                    f"native cadence ({grid.step_seconds}s = {grid.step_seconds/3600:.2f}h); "
                    f"got {exact_steps:.3f} steps. Re-check cadence detection or "
                    f"add an explicit step_seconds override."
                )
            self.horizon_steps[hours] = steps

        tolerance = max(
            TOLERANCE_FLOOR_SECONDS, int(grid.step_seconds * TOLERANCE_FRACTION)
        )

        self.index = []  # list of context-end index `t` usable as a sample anchor
        T = len(grid.timestamps)
        max_step = max(self.horizon_steps.values())
        for t in range(ctx_len - 1, T - max_step):
            if start_ts is not None and grid.timestamps[t] < start_ts:
                continue
            if end_ts is not None and grid.timestamps[t] > end_ts:
                continue

            ctx_slice = grid.values[t - ctx_len + 1 : t + 1]
            if np.isnan(ctx_slice).any():
                continue

            ok = True
            for hours, step in self.horizon_steps.items():
                dt = grid.timestamps[t + step] - grid.timestamps[t]
                nominal = step * grid.step_seconds
                if abs(dt - nominal) > tolerance:
                    ok = False
                    break
                if np.isnan(grid.values[t + step, :, TARGET_CHANNEL_IDX]).any():
                    ok = False
                    break
            if ok:
                self.index.append(t)

        if len(self.index) == 0:
            span = (
                f"{datetime.datetime.fromtimestamp(grid.timestamps[0], tz=datetime.timezone.utc).date()} "
                f"to {datetime.datetime.fromtimestamp(grid.timestamps[-1], tz=datetime.timezone.utc).date()}"
            )
            print(
                f"[dataset] WARNING: 0 samples built. Grid spans {span} ({T} timesteps, "
                f"cadence {grid.step_seconds}s). Requested window: "
                f"{start_ts and datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc).date()} "
                f"to {end_ts and datetime.datetime.fromtimestamp(end_ts, tz=datetime.timezone.utc).date()}. "
                f"Check that this range overlaps the grid's actual span and that ctx_len={ctx_len} "
                f"plus the horizons fit inside it."
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        t = self.index[i]
        g = self.grid
        ctx = g.values[t - self.ctx_len + 1 : t + 1]  # [ctx_len, L, C]
        ctx_std = (ctx - self.mean) / self.std        # mean/std shape (L, C) broadcasts

        tgt_mean = self.mean[:, TARGET_CHANNEL_IDX]    # (L,)
        tgt_std  = self.std[:, TARGET_CHANNEL_IDX]     # (L,)
        targets, raw_targets = {}, {}
        for hours, step in self.horizon_steps.items():
            raw = g.values[t + step, :, TARGET_CHANNEL_IDX]  # [L]
            std = (raw - tgt_mean) / tgt_std
            targets[hours] = torch.from_numpy(std.astype(np.float32))
            raw_targets[hours] = torch.from_numpy(raw.astype(np.float32))

        # Also hand back the two full future profiles (all channels,
        # standardized) — this is what the JEPA *target encoder* embeds.
        future_full = {
            hours: torch.from_numpy(
                ((g.values[t + step] - self.mean) / self.std).astype(np.float32)
            )
            for hours, step in self.horizon_steps.items()
        }

        return {
            "context": torch.from_numpy(ctx_std.astype(np.float32)),
            "targets": targets,  # standardized ABS_HUMIDITY, for the forecast head loss
            "raw_targets": raw_targets,  # physical units, for RMSE reporting
            "future_full": future_full,  # standardized full profile, for the JEPA embedding loss
        }


def build_datasets(
    db_path: str,
    place_name: str,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    ctx_len: int = 8,
):
    """
    Dates are ISO strings, e.g. "2020-01-01". Splits are by wall-clock
    time range so the evaluation window matches whatever the committee
    passes in at test time (per the "код должен... указать временной
    диапазон" requirement).
    """

    def to_ts(s):
        return int(
            datetime.datetime.fromisoformat(s)
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )

    with DB(db_path) as db:
        place = db.get_place(place_name)
        grid = load_profile_grid(db, place)

    train_ds = Era5WindowDataset(grid, ctx_len, to_ts(train_start), to_ts(train_end))
    eval_ds = Era5WindowDataset(
        grid,
        ctx_len,
        to_ts(eval_start),
        to_ts(eval_end),
        mean=train_ds.mean,
        std=train_ds.std,  # normalize eval with TRAIN stats only
    )
    return train_ds, eval_ds, grid
