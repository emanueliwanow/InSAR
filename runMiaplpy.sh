cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: bbf72fd2-453b-478c-b1e2-432b4b684556
EOF
chmod 600 ~/.cdsapirc
POLYGON="-47.46112 -6.56085,-47.45888 -6.56045,-47.45903 -6.55953,-47.46127 -6.55999,-47.46112 -6.56085"
PROJECT_NAME="Miaplpy_Tocantins"
DATA_DIR="/insar-data"
CONTAINER_NAME="insar_container"


# Get bbox from polygon: S N W E
# Parse polygon coordinates (format: "long lat,long lat,...")
# Split by comma and process each coordinate pair
IFS=',' read -ra coord_pairs <<< "$POLYGON"
min_lat=999
max_lat=-999
min_lon=999
max_lon=-999

for coord_pair in "${coord_pairs[@]}"; do
    # Trim whitespace and split lon lat
    coord_pair=$(echo "$coord_pair" | xargs)
    lon=$(echo "$coord_pair" | awk '{print $1}')
    lat=$(echo "$coord_pair" | awk '{print $2}')
    
    echo "Processing coordinate: lon=$lon, lat=$lat"
    
    # Compare latitudes
    if (( $(awk "BEGIN {print ($lat < $min_lat)}") )); then
        min_lat=$lat
    fi
    if (( $(awk "BEGIN {print ($lat > $max_lat)}") )); then
        max_lat=$lat
    fi
    
    # Compare longitudes
    if (( $(awk "BEGIN {print ($lon < $min_lon)}") )); then
        min_lon=$lon
    fi
    if (( $(awk "BEGIN {print ($lon > $max_lon)}") )); then
        max_lon=$lon
    fi
done


# Convert to numeric values to ensure proper formatting
Ssen=$(awk "BEGIN {printf \"%.3f\", $min_lat}")
Nsen=$(awk "BEGIN {printf \"%.3f\", $max_lat}")
Wsen=$(awk "BEGIN {printf \"%.3f\", $min_lon}")
Esen=$(awk "BEGIN {printf \"%.3f\", $max_lon}")

#mkdir -p $DATA_DIR/$PROJECT_NAME/mintpy/
#cd $DATA_DIR/$PROJECT_NAME/mintpy/

DEM_FILE=$(find $DATA_DIR/$PROJECT_NAME/DEM/ -name "*.dem" -type f | head -1)
echo "Using DEM file: $DEM_FILE"

cd $DATA_DIR/$PROJECT_NAME/
stackSentinel.py --bbox "$Ssen $Nsen $Wsen $Esen" --coregistration geometry -W slc --num_connections 3 --azimuth_looks 3 --range_looks 5 -s $DATA_DIR/$PROJECT_NAME/SLC/ -d "$DEM_FILE" -a $DATA_DIR/aux_cal -o $DATA_DIR/$PROJECT_NAME/orbits --start_date 2024-01-01
#smallbaselineApp.py $DATA_DIR/$PROJECT_NAME/mintpy/$PROJECT_NAME.txt 

run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_01_unpack_topo_reference
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_02_unpack_secondary_slc
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_03_average_baseline
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_04_fullBurst_geo2rdr
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_05_fullBurst_resample
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_06_extract_stack_valid_region
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_07_merge_reference_secondary_slc
run.py -i $DATA_DIR/$PROJECT_NAME/run_files/run_08_grid_baseline

