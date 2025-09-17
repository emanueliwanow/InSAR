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
if [[ -z "$POLYGON" || -z "$START_DATE" || -z "$END_DATE" ]]; then
    echo "Error: All parameters (--polygon, --startdate, --enddate) are required."
    usage
fi

# 1) Query SLC Bursts to get measurement TIFF paths
#    We use the OData Bursts endpoint, selecting only the S3Path field.
#    Each path ends in "/measurement/...tiff"
# Build a base filter (no track/swath yet)
url="https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"
filter_base="OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(($POLYGON))') \
and ContentDate/Start ge ${START_DATE}T00:00:00.000Z \
and ContentDate/Start le ${END_DATE}T23:59:59.999Z \
and OperationalMode eq 'IW' \
and PolarisationChannels eq 'VV' \
and OrbitDirection eq 'DESCENDING' \
and PlatformSerialIdentifier eq 'A' \
and SwathIdentifier eq '$SWATH'"

# 0) Seed: find track+swath if not given
seed=$(curl -s -G "$url" \
  --data-urlencode "\$filter=$filter_base" \
  --data-urlencode "\$select=ParentProductName,RelativeOrbitNumber,SwathIdentifier,ContentDate" \
  --data-urlencode "\$orderby=ContentDate/Start desc" \
  --data-urlencode "\$top=1")

RELATIVE_ORBIT="${RELATIVE_ORBIT:-$(echo "$seed" | jq -r '.value[0].RelativeOrbitNumber')}"
SWATH="${SWATH:-$(echo "$seed" | jq -r '.value[0].SwathIdentifier')}"

echo "Using track (RelativeOrbitNumber): $RELATIVE_ORBIT"
echo "Using subswath (SwathIdentifier): $SWATH"

# 1) Main query restricted to the same subswath + track
filter="$filter_base and RelativeOrbitNumber eq $RELATIVE_ORBIT and SwathIdentifier eq '$SWATH'"

response=$(curl -s -G "$url" \
  --data-urlencode "\$filter=$filter" \
  --data-urlencode "\$select=S3Path,ContentDate,PlatformSerialIdentifier,ParentProductName" \
  --data-urlencode "\$orderby=ContentDate/Start desc" \
  --data-urlencode "\$count=true" \
  --data-urlencode "\$top=200")

#echo response: $response
count=$(echo "$response" | jq -r '.["@odata.count"]')
echo "Found $count Sentinel-1 products in the given area and date range"

# Use jq to process each item in the response
echo "$response" | jq -c '.value[]' | while read -r item; do
  s3path=$(echo "$item" | jq -r '.S3Path')
  start_date=$(echo "$item" | jq -r '.ContentDate.Start')
  end_date=$(echo "$item" | jq -r '.ContentDate.End')
  platform_serial_identifier=$(echo "$item" | jq -r '.PlatformSerialIdentifier')
  echo "Product date: $start_date Satellite: $platform_serial_identifier"
 
done
