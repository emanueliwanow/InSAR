#!/usr/bin/env bash
set -euo pipefail

# Default values
DEFAULT_PROJECT_NAME="Tocantins"
DEFAULT_DATA_DIR="/insar-data"

# Initialize variables with defaults
PROJECT_NAME="$DEFAULT_PROJECT_NAME"
DATA_DIR="$DEFAULT_DATA_DIR"

# Function to show usage
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Download orbits for Sentinel-1 SLC data.

OPTIONS:
    --data-dir DIR      Base data directory (default: $DEFAULT_DATA_DIR)
    --project-name NAME Project name (default: $DEFAULT_PROJECT_NAME)
    -h, --help          Show this help message

EXAMPLES:
    $(basename "$0")
    $(basename "$0") --data-dir /my/data --project-name MyProject
    $(basename "$0") --project-name Amazon
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            echo "Use --help for usage information." >&2
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$DATA_DIR" ]]; then
    echo "Error: DATA_DIR cannot be empty" >&2
    exit 1
fi

if [[ -z "$PROJECT_NAME" ]]; then
    echo "Error: PROJECT_NAME cannot be empty" >&2
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Data directory '$DATA_DIR' does not exist" >&2
    exit 1
fi

SLC_DIR="$DATA_DIR/$PROJECT_NAME/SLC"
OUT_ROOT="$DATA_DIR/$PROJECT_NAME/orbits"

# Check if SLC directory exists
if [[ ! -d "$SLC_DIR" ]]; then
    echo "Error: SLC directory '$SLC_DIR' does not exist" >&2
    exit 1
fi

echo "Using DATA_DIR: $DATA_DIR"
echo "Using PROJECT_NAME: $PROJECT_NAME"
echo "SLC directory: $SLC_DIR"
echo "Output directory: $OUT_ROOT"
echo

# find all SLCs (SAFE dirs or zip files)
shopt -s nullglob
mapfile -t slcs < <(find "$SLC_DIR" -maxdepth 1 -type d -name "S1*_IW_SLC__*.SAFE" -o -type f -name "S1*_IW_SLC__*.zip" | sort)

if (( ${#slcs[@]} == 0 )); then
  echo "No SLCs found in $SLC_DIR"
  exit 1
fi

echo "Found ${#slcs[@]} SLC files"

# Check if orbit files are already downloaded
mkdir -p "$OUT_ROOT"
shopt -s nullglob
mapfile -t existing_orbits < <(find "$OUT_ROOT" -type f \( -name "*.EOF" -o -name "*.eof" \) | sort)
num_existing_orbits=${#existing_orbits[@]}

echo "Found $num_existing_orbits existing orbit files in $OUT_ROOT"

# If we already have at least as many orbit files as SLC files, skip downloading
if (( num_existing_orbits >= ${#slcs[@]} )); then
  echo "Orbit files already downloaded (found $num_existing_orbits orbit files for ${#slcs[@]} SLC files)"
  echo "Skipping orbit download. Use 'rm $OUT_ROOT/*.EOF $OUT_ROOT/*.eof 2>/dev/null || true' to force re-download."
  exit 0
fi

echo "Downloading orbit files..."
echo

for slc in "${slcs[@]}"; do
  #base="$(basename "$slc")"
  # Extract acquisition date (YYYYMMDD) from the standard S1 name
  # e.g. S1A_IW_SLC__1SDV_20230727T075102_...
  #date_str="$(grep -oE '20[0-9]{6}T[0-9]{6}' <<< "$base" | head -n1 | cut -c1-8 || true)"
  # If we can’t parse a date for any reason, just dump into a generic folder
  #out_dir="$OUT_ROOT/${date_str:-misc}"
  #mkdir -p "$out_dir"
  #echo "→ $base  →  $out_dir"
  echo "Fetching orbits for $slc"
  fetchOrbit.py -i "$slc" -o "$OUT_ROOT" -u "emanueldearaujo123@gmail.com" -p "Ps__143993**" -t /insar-data/.copernicus_dataspace_token
done
