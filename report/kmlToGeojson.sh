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

CONFIG_FILE="$1"

# Load config file
load_config "$CONFIG_FILE"

# Convert KML to GeoJSON
ogr2ogr -f GeoJSON /insar-data/$PROJECT_NAME/${PROJECT_NAME}.geojson /insar-data/$PROJECT_NAME/${PROJECT_NAME}.kml