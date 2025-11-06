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
    echo "  SUBSET=\"lat_min:lat_max,lon_min:lon_max\""
    echo "  REFERENCE_POINT=\"lat,lon\""
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


python /insar-data/InSAR/old/generateMask.py \
  --geom $DATA_DIR/$PROJECT_NAME/miaplpy/inputs/geometryRadar.h5 \
  --polygon "$POLYGON_MASK" \
  -o $DATA_DIR/$PROJECT_NAME/bridge_mask.h5 \
  --in-mask $DATA_DIR/$PROJECT_NAME/miaplpy/shadow_mask.h5 