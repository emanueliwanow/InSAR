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
    echo "  POLYGON=\"coordinates\""
    echo "  START_DATE=\"YYYY-MM-DD\""
    echo "  END_DATE=\"YYYY-MM-DD\""
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
if [[ -z "$POLYGON" || -z "$START_DATE" || -z "$END_DATE" || -z "$PROJECT_NAME" || -z "$DATA_DIR" ]]; then
    echo "Error: All required parameters (POLYGON, START_DATE, END_DATE, PROJECT_NAME, DATA_DIR) must be defined in the config file."
    echo "Please check your config file: $CONFIG_FILE"
    exit 1
fi



# 1) Query SLC Bursts to get measurement TIFF paths
#    We use the OData Bursts endpoint, selecting only the S3Path field.
#    Each path ends in "/measurement/...tiff" :contentReference[oaicite:0]{index=0}
url="https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"
filter_base="OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(($POLYGON))') \
and ContentDate/Start ge ${START_DATE}T00:00:00.000Z \
and ContentDate/Start le ${END_DATE}T23:59:59.999Z \
and OperationalMode eq 'IW' \
and PolarisationChannels eq 'VV' \
and OrbitDirection eq 'DESCENDING' \
and PlatformSerialIdentifier eq 'A'"

# 0) Seed: find track+swath if not given
seed=$(curl -s -G "$url" \
  --data-urlencode "\$filter=$filter_base" \
  --data-urlencode "\$select=ParentProductName,RelativeOrbitNumber,SwathIdentifier,ContentDate" \
  --data-urlencode "\$orderby=ContentDate/Start desc" \
  --data-urlencode "\$top=1")

RELATIVE_ORBIT="${RELATIVE_ORBIT:-$(echo "$seed" | jq -r '.value[0].RelativeOrbitNumber')}"
SWATH="${SWATH:-$(echo "$seed" | jq -r '.value[0].SwathIdentifier')}"

echo "Using track (RelativeOrbitNumber): $RELATIVE_ORBIT"
#echo "Using subswath (SwathIdentifier): $SWATH"

# 1) Main query restricted to the same subswath + track
filter="$filter_base and RelativeOrbitNumber eq $RELATIVE_ORBIT"


response=$(curl -s -G "$url" \
  --data-urlencode "\$filter=$filter" \
  --data-urlencode "\$select=S3Path,ContentDate,PlatformSerialIdentifier" \
  --data-urlencode "\$orderby=ContentDate/Start desc" \
  --data-urlencode "\$count=true" \
  --data-urlencode "\$top=400")
#echo response: $response
count=$(echo "$response" | jq -r '.["@odata.count"]')
echo "Count: $count"

# Use jq to process each item in the response
# Initialize counter for alternating download pattern
counter=0
echo "$response" | jq -c '.value[]' | while read -r item; do
  counter=$((counter + 1))
  s3path=$(echo "$item" | jq -r '.S3Path')
  start_date=$(echo "$item" | jq -r '.ContentDate.Start')
  end_date=$(echo "$item" | jq -r '.ContentDate.End')
  platform_serial_identifier=$(echo "$item" | jq -r '.PlatformSerialIdentifier')
  echo "Product #$counter - Start Date: $start_date Satellite: $platform_serial_identifier"
  
  # Only download every other product (odd-numbered products)
  if [ $((counter % 1)) -eq 0 ]; then
    echo "  --> Downloading product #$counter"
    # Strip off the "/measurement/…" to get the SAFE prefix
    safe_prefix="${s3path%/measurement*}/"

    # # Derive the SAFE folder name (basename), e.g. "S1A_IW_SLC__1SDV_…SAFE"
    safe_folder=$(basename "$safe_prefix")

    # # Create a matching local directory
    local_dest="${DATA_DIR}/${PROJECT_NAME}/SLC/${safe_folder}"
    mkdir -p "$local_dest"

    echo "Downloading entire SAFE product into ${local_dest}/"
    aws s3 sync \
      --endpoint-url https://eodata.dataspace.copernicus.eu \
      "s3:/${safe_prefix}" \
      "$local_dest/" 
  else
    echo "  --> Skipping product #$counter (downloading every other product)"
  fi

  
done
