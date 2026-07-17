"""
download_data.py

Ensures the competition's ERA5 SQLite database exists locally, downloading
it from the shared Google Drive folder on first run if it doesn't.

Requires: pip install gdown

Usage as a library (this is how train.py / visualize_embeddings.py use it):

    from download_data import ensure_db
    db_path = ensure_db("./data/data-challenge.db")

Usage as a script:

    python download_data.py --db ./data/data-challenge.db
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1v68Qigz0bitHuyeTyXeKtCUxDhqWIj6e?usp=sharing"
DB_EXTENSIONS = (".db", ".sqlite", ".sqlite3")


def ensure_db(
    db_path: str, folder_url: str = DRIVE_FOLDER_URL, verbose: bool = True
) -> str:
    """
    If `db_path` already exists, does nothing and returns it unchanged.
    Otherwise downloads the whole shared Drive folder into a local cache
    dir next to `db_path`, finds the one .db/.sqlite file inside it, and
    copies it to `db_path`.

    The Drive link is a *folder* (not a direct file), so this pulls
    everything in it once and reuses the cache on subsequent calls —
    only the final `shutil.copy2` runs again if `db_path` itself was
    deleted but the cache wasn't.
    """
    db_path = Path(db_path)
    if db_path.exists():
        if verbose:
            print(f"[download_data] found existing DB at {db_path}, skipping download")
        return str(db_path)

    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            "gdown is required to auto-download the dataset: pip install gdown"
        ) from e

    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = db_path.parent / ".drive_cache"

    candidates = _find_db_files(cache_dir) if cache_dir.exists() else []
    if not candidates:
        if verbose:
            print(
                f"[download_data] {db_path} not found, downloading Drive folder "
                f"to {cache_dir} (this only needs to happen once)..."
            )
        cache_dir.mkdir(exist_ok=True)
        gdown.download_folder(
            url=folder_url, output=str(cache_dir), quiet=not verbose, use_cookies=False
        )
        candidates = _find_db_files(cache_dir)

    if not candidates:
        raise RuntimeError(
            f"Downloaded the Drive folder to {cache_dir} but found no "
            f"{DB_EXTENSIONS} file inside. Check the folder contents manually "
            f"and either rename the file to one of those extensions or pass "
            f"the correct path directly via --db."
        )
    if len(candidates) > 1 and verbose:
        print(
            f"[download_data] multiple db-like files found, using the first: "
            f"{[c.name for c in candidates]}"
        )

    chosen = candidates[0]
    shutil.copy2(chosen, db_path)
    if verbose:
        print(f"[download_data] copied {chosen} -> {db_path}")
    return str(db_path)


def _find_db_files(root: Path) -> list[Path]:
    return sorted(p for ext in DB_EXTENSIONS for p in root.rglob(f"*{ext}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="target local path for the .db file")
    ap.add_argument("--folder-url", default=DRIVE_FOLDER_URL)
    args = ap.parse_args()
    ensure_db(args.db, args.folder_url)


if __name__ == "__main__":
    main()
