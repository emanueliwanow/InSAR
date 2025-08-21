cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: bbf72fd2-453b-478c-b1e2-432b4b684556
EOF
chmod 600 ~/.cdsapirc
POLYGON="-47.46112 -6.56085,-47.45888 -6.56045,-47.45903 -6.55953,-47.46127 -6.55999,-47.46112 -6.56085"
PROJECT_NAME="Tocantins"
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

mkdir -p $DATA_DIR/$PROJECT_NAME/mintpy/
cd $DATA_DIR/$PROJECT_NAME/mintpy/

cat > $DATA_DIR/$PROJECT_NAME/mintpy/$PROJECT_NAME.txt << EOF
# vim: set filetype=cfg:
##------------------------------- ISCE/topsStack ----------------------##
#ssaraopt = --platform=SENTINEL-1A,SENTINEL-1B -r 128 -f 589,590,591,592,593  -e 2017-07-01
#sentinelStack.boundingBox      = '-1 0.15 -91.6 -90.9'
#sentinelStack.subswath         = 1 2  # comment 
#sentinelStack.numConnections   = 5   # comment
#sentinelStack.azimuthLooks     = 5   # comment
#sentinelStack.rangeLooks       = 15  # comment
#sentinelStack.filtStrength     = 0.2  # comment
#sentinelStack.unwMethod        = snaphu  # comment
#sentinelStack.coregistration   = auto  # comment
#subset.y0:y1,x0:x1 = 1150:1600,1070:1670


##-------------------------------- MintPy -----------------------------##
########## 1. Load Data (--load to exit after this step)
## load_data.py -H to check more details and example inputs.
mintpy.load.processor        = isce
##---------for ISCE only:
mintpy.load.metaFile         = ../reference/IW*.xml
mintpy.load.baselineDir      = ../baselines
##---------interferogram datasets:
mintpy.load.unwFile          = ../merged/interferograms/*/filt_*.unw
mintpy.load.corFile          = ../merged/interferograms/*/filt_*.cor
mintpy.load.connCompFile     = ../merged/interferograms/*/filt_*.unw.conncomp
##---------geometry datasets:
mintpy.load.demFile          = ../merged/geom_reference/hgt.rdr
mintpy.load.lookupYFile      = ../merged/geom_reference/lat.rdr
mintpy.load.lookupXFile      = ../merged/geom_reference/lon.rdr
mintpy.load.incAngleFile     = ../merged/geom_reference/los.rdr
mintpy.load.azAngleFile      = ../merged/geom_reference/los.rdr
mintpy.load.shadowMaskFile   = ../merged/geom_reference/shadowMask.rdr
mintpy.load.waterMaskFile    = None

mintpy.subset.lalo =  [-6.59:-6.54,-47.49:-47.41]
#mintpy.subset.lalo =  [$Ssen:$Nsen,$Wsen:$Esen]
mintpy.reference.minCoherence = 0.5
# mintpy.reference.lalo        =   -6.558134,-47.450072
# mintpy.topographicResidual.stepFuncDate  = 20170910,20180613  #eruption dates
# mintpy.deramp                = linear
EOF

smallbaselineApp.py $DATA_DIR/$PROJECT_NAME/mintpy/$PROJECT_NAME.txt 
