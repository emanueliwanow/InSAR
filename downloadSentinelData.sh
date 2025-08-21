#!/usr/bin/env bash
set -euo pipefail

# 1) Query SLC Bursts to get measurement TIFF paths
#    We use the OData Bursts endpoint, selecting only the S3Path field.
#    Each path ends in "/measurement/...tiff" :contentReference[oaicite:0]{index=0}
url="https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"
filter="OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((-47.46112 -6.56085,-47.45888 -6.56045,-47.45903 -6.55953,-47.46127 -6.55999,-47.46112 -6.56085))') \
and ContentDate/Start ge 2019-01-01T00:00:00.000Z \
and ContentDate/Start le 2024-12-21T23:59:59.999Z \
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
echo "$response" | jq -c '.value[]' | while read -r item; do
  s3path=$(echo "$item" | jq -r '.S3Path')
  start_date=$(echo "$item" | jq -r '.ContentDate.Start')
  end_date=$(echo "$item" | jq -r '.ContentDate.End')
  platform_serial_identifier=$(echo "$item" | jq -r '.PlatformSerialIdentifier')
  echo "Start Date: $start_date Satellite: $platform_serial_identifier"
  # Strip off the "/measurement/…" to get the SAFE prefix
  safe_prefix="${s3path%/measurement*}/"

  # # Derive the SAFE folder name (basename), e.g. "S1A_IW_SLC__1SDV_…SAFE"
  safe_folder=$(basename "$safe_prefix")

  # # Create a matching local directory
  local_dest="/insar-data/Tocantins/SLC/${safe_folder}"
  mkdir -p "$local_dest"

  #echo "Downloading entire SAFE product into ${local_dest}/"
  aws s3 cp \
    --recursive \
    --endpoint-url https://eodata.dataspace.copernicus.eu \
    "s3:/${safe_prefix}" \
    "$local_dest/"

  
done
