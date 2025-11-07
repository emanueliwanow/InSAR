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

# Validate required parameters
if [[ -z "$PROJECT_NAME" || -z "$DATA_DIR" || -z "$SUBSET" || -z "$REFERENCE_POINT" ]]; then
    echo "Error: Required parameters (PROJECT_NAME, DATA_DIR, SUBSET, REFERENCE_POINT) must be defined in the config file."
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

cat > $DATA_DIR/$PROJECT_NAME/config.json << EOF
{
    general: {
        input_path: "$MIAPLPY_VERSION/inputs/",
        output_path: "sarvey/",
        num_cores: 4,
        num_patches: 1,
        apply_temporal_unwrapping: true,
        spatial_unwrapping_method: "puma",
        logging_level: "INFO",
        logfile_path: "sarvey_logfiles/",
    },
    phase_linking: {
        use_phase_linking_results: true,
        inverted_path: "$MIAPLPY_VERSION/inverted/",
        num_siblings: 20,
        mask_phase_linking_file: "",
        use_ps: true,
        mask_ps_file: "$MIAPLPY_VERSION/maskPS.h5",
    },
    preparation: {
        start_date: null,
        end_date: null,
        ifg_network_type: "sb",
        num_ifgs: 3,
        max_tbase: 100,
        filter_window_size: 9,
    },
    consistency_check: {
        coherence_p1: 0.8,
        grid_size: 60,
        mask_p1_file: "",
        num_nearest_neighbours: 30,
        max_arc_length: null,
        velocity_bound: 0.1,
        dem_error_bound: 100.0,
        num_optimization_samples: 100,
        arc_unwrapping_coherence: 0.4,
        min_num_arc: 3,
    },
    unwrapping: {
        use_arcs_from_temporal_unwrapping: true,
    },
    filtering: {
        coherence_p2: 0.7,
        apply_aps_filtering: true,
        interpolation_method: "kriging",
        grid_size: 1000,
        mask_p2_file: "",
        use_moving_points: true,
        max_temporal_autocorrelation: 0.3,
    },
    densification: {
        num_connections_to_p1: 5,
        max_distance_to_p1: 2000,
        velocity_bound: 0.15,
        dem_error_bound: 100.0,
        num_optimization_samples: 100,
        arc_unwrapping_coherence: 0.5,
    },
}
EOF


cd $DATA_DIR/$PROJECT_NAME/

sarvey -f config.json 0 4
sarvey_export sarvey/p2_coh70_ts.h5 -o sarvey/shp/p2_coh80.shp






