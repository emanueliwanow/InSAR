#!/usr/bin/env bash
set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: $0 --polygon=\"coords\" --startdate=\"YYYY-MM-DD\" --enddate=\"YYYY-MM-DD\""
    echo "Example: $0 --polygon=\"-47.46112 -6.56085,-47.45888 -6.56045,-47.45903 -6.55953,-47.46127 -6.55999,-47.46112 -6.56085\" --startdate=\"2019-01-01\" --enddate=\"2024-12-21\""
    exit 1
}

# Default values
POLYGON=""
START_DATE=""
END_DATE=""

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
if [[ -z "$POLYGON" || -z "$START_DATE" || -z "$END_DATE" ]]; then
    echo "Error: All parameters (--polygon, --startdate, --enddate) are required."
    usage
fi

# 1) Query SLC Bursts to get measurement TIFF paths
#    We use the OData Bursts endpoint, selecting only the S3Path field.
#    Each path ends in "/measurement/...tiff"
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
  --data-urlencode "\$top=400")
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
