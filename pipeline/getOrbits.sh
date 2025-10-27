#!/usr/bin/env bash
set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: $0 <config_file>"
    echo ""
    echo "Arguments:"
    echo "  config_file    Path to configuration file (required)"
    echo ""
    echo "Config file format:"
    echo "  PROJECT_NAME=\"project_name\""
    echo "  DATA_DIR=\"/path/to/data\""
    echo ""
    echo "Example:"
    echo "  $0 /insar-data/RioNegro/parameters.cfg"
    exit 1
}

# Function to load config file
load_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        echo "Error: Config file '$config_file' not found."
        exit 1
    fi
    
    # Source the config file safely
    # First check if the config file has proper format
    if ! grep -E '^[A-Z_]+=.*$' "$config_file" > /dev/null; then
        echo "Warning: Config file may not have proper KEY=VALUE format"
    fi
    
    # Source the config file
    source "$config_file"
    
    echo "Loaded configuration from: $config_file"
}

# Check command line arguments
if [[ $# -ne 1 ]]; then
    echo "Error: Exactly one argument (config file path) is required."
    usage
fi

# Check for help option
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

CONFIG_FILE="$1"

# Load config file
load_config "$CONFIG_FILE"

# Validate required parameters
if [[ -z "$PROJECT_NAME" || -z "$DATA_DIR" ]]; then
    echo "Error: Required parameters (PROJECT_NAME, DATA_DIR) must be defined in the config file."
    echo "Please check your config file: $CONFIG_FILE"
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
# if (( num_existing_orbits >= ${#slcs[@]} )); then
#   echo "Orbit files already downloaded (found $num_existing_orbits orbit files for ${#slcs[@]} SLC files)"
#   echo "Skipping orbit download. Use 'rm $OUT_ROOT/*.EOF $OUT_ROOT/*.eof 2>/dev/null || true' to force re-download."
#   exit 0
# fi

echo "Downloading orbit files..."
echo

for slc in "${slcs[@]}"; do
  echo "Fetching orbits for $slc"
  fetchOrbit.py -i "$slc" -o "$OUT_ROOT" -u "emanueldearaujo123@gmail.com" -p "Ps__143993**" -t /insar-data/.copernicus_dataspace_token
done
