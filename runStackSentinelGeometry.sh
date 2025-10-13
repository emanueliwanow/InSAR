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
    echo "  POLYGON=\"coordinates\""
    echo "  BBOX=\"bbox_coordinates\""
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
if [[ -z "$PROJECT_NAME" || -z "$DATA_DIR" || -z "$POLYGON" || -z "$BBOX" ]]; then
    echo "Error: Required parameters (PROJECT_NAME, DATA_DIR, POLYGON, BBOX) must be defined in the config file."
    echo "Please check your config file: $CONFIG_FILE"
    exit 1
fi

# Setup CDS API configuration
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: bbf72fd2-453b-478c-b1e2-432b4b684556
EOF
chmod 600 ~/.cdsapirc

# Additional constants
CONTAINER_NAME="insar_container"



DEM_FILE=$(find $DATA_DIR/$PROJECT_NAME/DEM/ -name "*.dem" -type f | head -1)
echo "Using DEM file: $DEM_FILE"

cd $DATA_DIR/$PROJECT_NAME/


stackSentinel.py \
  --bbox "$BBOX" \
  --coregistration geometry \
  -W slc \
  --esd_coherence_threshold 0.72 \
  --snr_misreg_threshold 6 \
  --num_overlap_connections 6 \
  --azimuth_looks 1 --range_looks 1 \
  -s $DATA_DIR/$PROJECT_NAME/SLC/ \
  -d "$DEM_FILE" \
  -a $DATA_DIR/aux_cal \
  -o $DATA_DIR/$PROJECT_NAME/orbits  \
  --start_date $START_DATE \
  --stop_date $END_DATE 

#   --start_date STARTDATE
#                         Start date for stack processing. Acquisitions before start date are ignored. format should be YYYY-MM-DD e.g., 2015-01-23
#   --stop_date STOPDATE  Stop date for stack processing. Acquisitions after stop date are ignored. format should be YYYY-MM-DD e.g., 2017-02-26


run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_01_unpack_topo_reference
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_02_unpack_secondary_slc
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_03_average_baseline
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_04_fullBurst_geo2rdr
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_05_fullBurst_resample
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_06_extract_stack_valid_region
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_07_merge_reference_secondary_slc
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_08_grid_baseline
