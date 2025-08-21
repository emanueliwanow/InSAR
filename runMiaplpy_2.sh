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
Ssen=$(awk "BEGIN {printf \"%.2f\", $min_lat}")
Nsen=$(awk "BEGIN {printf \"%.2f\", $max_lat}")
Wsen=$(awk "BEGIN {printf \"%.2f\", $min_lon}")
Esen=$(awk "BEGIN {printf \"%.2f\", $max_lon}")

#mkdir -p $DATA_DIR/$PROJECT_NAME/miaplpy/
#cd $DATA_DIR/$PROJECT_NAME/miaplpy/

cat > $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt << EOF
################
miaplpy.load.processor      = isce  #[isce,snap,gamma,roipac], auto for isceTops
miaplpy.load.updateMode     = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
miaplpy.load.compression    = gzip  #[gzip / lzf / no], auto for no.
miaplpy.load.autoPath       = yes    # [yes, no] auto for no
        
miaplpy.load.slcFile        = ../merged/SLC/*/*.slc.full  #[path2slc_file]
##---------for ISCE only:
miaplpy.load.metaFile       = ../reference/IW*.xml
miaplpy.load.baselineDir    = ../baselines
##---------geometry datasets:
miaplpy.load.demFile          = ../merged/geom_reference/hgt.rdr.full
miaplpy.load.lookupYFile      = ../merged/geom_reference/lat.rdr.full
miaplpy.load.lookupXFile      = ../merged/geom_reference/lon.rdr.full
miaplpy.load.incAngleFile     = ../merged/geom_reference/los.rdr.full
miaplpy.load.azAngleFile      = ../merged/geom_reference/los.rdr.full
miaplpy.load.shadowMaskFile   = ../merged/geom_reference/shadowMask.rdr.full
miaplpy.load.waterMaskFile    = None
##---------interferogram datasets:
miaplpy.load.unwFile        = ./inverted/interferograms_single_reference/*/*fine*.unw
miaplpy.load.corFile        = ./inverted/interferograms_single_reference/*/*fine*.cor
miaplpy.load.connCompFile   = ./inverted/interferograms_single_reference/*/*.unw.conncomp
        
##---------subset (optional):
## if both yx and lalo are specified, use lalo option unless a) no lookup file AND b) dataset is in radar coord
miaplpy.subset.lalo         = -6.59:-6.54,-47.49:-47.41

# MiaplPy options 
#miaplpy.multiprocessing.numProcessor   = 40
miaplpy.interferograms.type = single_reference

## Mintpy options
mintpy.compute.cluster     = local  # if dask is not available, set this option to no 
mintpy.compute.numWorker   = 8 

#mintpy.reference.lalo     = -0.1786, -78.5933
mintpy.troposphericDelay.method = no
EOF
#cd $DATA_DIR/$PROJECT_NAME/
#miaplpyApp $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt --dir ./miaplpy
