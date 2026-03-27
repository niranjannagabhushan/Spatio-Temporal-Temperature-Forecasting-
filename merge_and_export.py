"""
merge_and_export.py
───────────────────
Standalone script that:
  1. Extracts the RAR archive containing weather measurements.
  2. Loads both the weather data and station metadata CSVs.
  3. Merges them on ``station_id`` (right-merge via existing helper).
  4. Writes the result to ``merged_weather_stations.csv`` in the project root.
  5. Prints a short summary of each step.

Run from the project root:
    python merge_and_export.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# ── rarfile import with graceful fallback ─────────────────────────────────────
try:
    import rarfile
except ImportError:
    print(
        "ERROR: The 'rarfile' package is not installed.\n"
        "Please install it with:\n\n"
        "    pip install rarfile>=4.0\n\n"
        "or add it to requirements.txt and run:\n\n"
        "    pip install -r requirements.txt\n"
    )
    sys.exit(1)

# ── Make sure the project root is on sys.path so src.* imports resolve ────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402 – must come after sys.path is patched

from src.data.loader import load_station_metadata, load_weather_data  # noqa: E402
from src.data.preprocessor import merge_station_data  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
RAR_PATH = PROJECT_ROOT / "weather_station_data_202406202154.rar"
OUTPUT_PATH = PROJECT_ROOT / "merged_weather_stations.csv"

# Read config.yaml to discover the exact CSV filename expected inside the archive
_config_path = PROJECT_ROOT / "config.yaml"
with open(_config_path, "r", encoding="utf-8") as _fh:
    _cfg = yaml.safe_load(_fh)

WEATHER_CSV_NAME: str = _cfg["data"]["weather_data"]        # e.g. "weather_station_data_202406202154.csv"
METADATA_CSV_NAME: str = _cfg["data"]["station_metadata"]   # e.g. "weather_stations_202406211148.csv"


def extract_weather_csv(rar_path: Path, csv_name: str) -> Path:
    """Extract *csv_name* from *rar_path* into a temporary directory.

    Returns the full path to the extracted CSV file.
    """
    tmp_dir = tempfile.mkdtemp(prefix="weather_rar_")
    with rarfile.RarFile(str(rar_path)) as rf:
        # Find the member whose filename (basename) matches csv_name
        members = rf.namelist()
        match = next(
            (m for m in members if Path(m).name == csv_name),
            None,
        )
        if match is None:
            available = ", ".join(members) or "<archive is empty>"
            raise FileNotFoundError(
                f"'{csv_name}' not found inside '{rar_path.name}'.\n"
                f"Available members: {available}"
            )
        rf.extract(match, path=tmp_dir)

    extracted_path = Path(tmp_dir) / match
    return extracted_path


def main() -> None:
    # ── Step 1: Extract RAR ───────────────────────────────────────────────────
    print(f"Extracting '{RAR_PATH.name}' …")
    weather_csv_path = extract_weather_csv(RAR_PATH, WEATHER_CSV_NAME)
    print(f"  → Extracted to: {weather_csv_path}\n")

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    metadata_path = PROJECT_ROOT / METADATA_CSV_NAME

    print(f"Loading weather data from:    {weather_csv_path}")
    weather_df = load_weather_data(str(weather_csv_path))
    print(f"  shape: {weather_df.shape}")

    print(f"Loading station metadata from: {metadata_path.name}")
    metadata_df = load_station_metadata(str(metadata_path))
    print(f"  shape: {metadata_df.shape}\n")

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    print("Merging on 'station_id' (right-merge) …")
    merged_df = merge_station_data(weather_df, metadata_df)
    print(f"  merged shape: {merged_df.shape}\n")

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    merged_df.to_csv(str(OUTPUT_PATH), index=False)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    print("─" * 55)
    print("Summary")
    print("─" * 55)
    print(f"  Weather data shape   : {weather_df.shape}")
    print(f"  Station metadata shape: {metadata_df.shape}")
    print(f"  Merged output shape  : {merged_df.shape}")
    print(f"  Columns              : {list(merged_df.columns)}")
    print(f"  Output file          : {OUTPUT_PATH}")
    print("─" * 55)


if __name__ == "__main__":
    main()
