#!/usr/bin/env bash
set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: $0 --polygon=\"coords\" --startdate=\"YYYY-MM-DD\" --enddate=\"YYYY-MM-DD\" --project-name=\"project_name\" --data-dir=\"data_dir\""
    echo "Example: $0 --polygon=\"-47.46112 -6.56085,-47.45888 -6.56045,-47.45903 -6.55953,-47.46127 -6.55999,-47.46112 -6.56085\" --startdate=\"2019-01-01\" --enddate=\"2024-12-21\" --project-name=\"AutazMirim\" --data-dir=\"/insar-data\""
    exit 1
}

# Default values
POLYGON=""
START_DATE=""
END_DATE=""
DATA_DIR="/insar-data"
PROJECT_NAME="AutazMirim"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --polygon=*)
            POLYGON="${1#*=}"
            shift
            ;;
        --startdate=*)
            START_DATE="${1#*=}"
            shift
            ;;
        --enddate=*)
            END_DATE="${1#*=}"
            shift
            ;;
        --project-name=*)
            PROJECT_NAME="${1#*=}"
            shift
            ;;
        --data-dir=*)
            DATA_DIR="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option $1"
            usage
            ;;
        
    esac
done

# Validate required parameters
if [[ -z "$POLYGON" || -z "$START_DATE" || -z "$END_DATE" || -z "$PROJECT_NAME" || -z "$DATA_DIR" ]]; then
    echo "Error: All parameters (--polygon, --startdate, --enddate, --project-name, --data-dir) are required."
    usage
fi



# 1) Query SLC Bursts to get measurement TIFF paths
#    We use the OData Bursts endpoint, selecting only the S3Path field.
#    Each path ends in "/measurement/...tiff" :contentReference[oaicite:0]{index=0}
url="https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"
filter="OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(($POLYGON))') \
and ContentDate/Start ge ${START_DATE}T00:00:00.000Z \
and ContentDate/Start le ${END_DATE}T23:59:59.999Z \
and OperationalMode eq 'IW' \
and PolarisationChannels eq 'VV' \
and OrbitDirection eq 'DESCENDING' \
and PlatformSerialIdentifier eq 'A'"


response=$(curl -s -G "$url" \
  --data-urlencode "\$filter=$filter" \
  --data-urlencode "\$select=S3Path,ContentDate,PlatformSerialIdentifier" \
  --data-urlencode "\$orderby=ContentDate/Start desc" \
  --data-urlencode "\$count=true" \
  --data-urlencode "\$top=200")
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
  #if [ $((counter % 2)) -eq 1 ]; then
    echo "  --> Downloading product #$counter"
    # Strip off the "/measurement/…" to get the SAFE prefix
    safe_prefix="${s3path%/measurement*}/"

    # # Derive the SAFE folder name (basename), e.g. "S1A_IW_SLC__1SDV_…SAFE"
    safe_folder=$(basename "$safe_prefix")

    # # Create a matching local directory
    local_dest="${DATA_DIR}/${PROJECT_NAME}/SLC/${safe_folder}"
    mkdir -p "$local_dest"

    #echo "Downloading entire SAFE product into ${local_dest}/"
    aws s3 sync \
      --endpoint-url https://eodata.dataspace.copernicus.eu \
      "s3:/${safe_prefix}" \
      "$local_dest/" 
  #else
  #  echo "  --> Skipping product #$counter (downloading every other product)"
  #fi

  
done
