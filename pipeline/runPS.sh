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

cat > $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt << EOF
# vim: set filetype=cfg:

############################
# 0) Loader (MiaplPy / MintPy)
############################
miaplpy.load.processor      = isce     #[isce,snap,gamma,roipac]; auto handles isceTOPS
miaplpy.load.updateMode     = yes      #[yes/no] re-use HDF5 if complete
miaplpy.load.compression    = gzip     #[gzip/lzf/no]
miaplpy.load.autoPath       = yes

# Coregistered SLCs (ISCE topsStack output)
miaplpy.load.slcFile        = ../merged/SLC/*/*.slc.full

## ISCE metadata/baselines
miaplpy.load.metaFile       = ../reference/IW*.xml
miaplpy.load.baselineDir    = ../baselines

## Geometry (ISCE geom_reference)
miaplpy.load.demFile        = ../merged/geom_reference/hgt.rdr.full
miaplpy.load.lookupYFile    = ../merged/geom_reference/lat.rdr.full
miaplpy.load.lookupXFile    = ../merged/geom_reference/lon.rdr.full
miaplpy.load.incAngleFile   = ../merged/geom_reference/los.rdr.full
miaplpy.load.azAngleFile    = ../merged/geom_reference/los.rdr.full
miaplpy.load.shadowMaskFile = ../merged/geom_reference/shadowMask.rdr.full
miaplpy.load.waterMaskFile  = None

## Interferograms (MiaplPy-generated dirs)
# Use mini_stacks to increase redundancy in low-coherence scenes
miaplpy.load.unwFile        = ./inverted/interferograms_delaunay_4/*/*fine*.unw
miaplpy.load.corFile        = ./inverted/interferograms_delaunay_4/*/*fine*.cor
miaplpy.load.connCompFile   = ./inverted/interferograms_delaunay_4/*/*.unw.conncomp

############################
# 1) Subset (tight around bridge)
############################
# Curaca bridge bbox (lat_min:lat_max, lon_min:lon_max)
miaplpy.subset.lalo         = $SUBSET

############################
# 2) Interferogram network
############################
# Increase redundancy and favor short baselines via mini_stacks
miaplpy.interferograms.networkType = delaunay   #[mini_stacks,single_reference,sequential,delaunay]

############################
# 3) Phase Linking / SHP (MiaplPy)
############################
# Patch-wise phase linking
miaplpy.inversion.patchSize           = auto     # auto ~200
miaplpy.inversion.ministackSize       = 10       # images per mini-stack
# SHP windows - keep small so bridge pixels don't mix with forest
miaplpy.inversion.rangeWindow         = 5
miaplpy.inversion.azimuthWindow       = 5
miaplpy.inversion.shpTest             = ks     #[ks,ad,ttest]; auto->ks
miaplpy.inversion.phaseLinkingMethod  = EMI  #[EVD,EMI,PTA,sequential_*]
miaplpy.inversion.sbw_connNum         = auto
miaplpy.inversion.PsNumShp            = 10        # neighbors used for PS candidate
#miaplpy.inversion.mask                = $DATA_DIR/$PROJECT_NAME/bridge_mask.h5

############################
# 4) Time series threshold (MiaplPy)
############################
# Keep a bit permissive while tuning; you can raise later
miaplpy.timeseries.minTempCoh         = 0.2

############################
# 5) MintPy options (run by MiaplPy steps)
############################

# Reference point (keep on stable man-made target near deck)
mintpy.reference.lalo                 = $REFERENCE_POINT
mintpy.reference.maskFile             = no

########## 5.1 Modify network (data-driven pruning)
# Coherence-/area-ratio-based pruning + minimum redundancy
# mintpy.network.coherenceBased         = yes      # enable coherence-based pruning
# mintpy.network.minCoherence           = 0.5      # start modest; 0.5–0.6 typical
# mintpy.network.areaRatioBased         = yes
# mintpy.network.minAreaRatio           = 0.6
# # Inversion redundancy (per-acquisition min # of ifgs)
# mintpy.networkInversion.minRedundancy = 1.5

########## 5.2 Unwrapping error correction
# Phase-closure based unwrap error correction (falls back to bridging if needed)
mintpy.correct_unwrap_error.method    = phase_closure

########## 5.3 Tropospheric delay correction
# Provide your weather dir (GACOS ztd grids or ERA5 via PyAPS)
mintpy.troposphericDelay.method       = auto    #[gacos/pyaps/no]
mintpy.troposphericDelay.weatherDir   = $DATA_DIR/$PROJECT_NAME/

########## 5.4 Inversion mask strictness
# After cleaning, you can tighten these (default auto=0.7 is strict)
mintpy.networkInversion.minTempCoh    = 0.6      #[0.0-1.0] temporal coherence mask

########## 5.5 Geocode
mintpy.geocode                        = auto
mintpy.geocode.SNWE                   = auto
mintpy.geocode.laloStep               = 0.000070189, 0.000084286
mintpy.geocode.interpMethod           = auto
mintpy.geocode.fillValue              = auto

mintpy.reference.date = 20170709   #[reference_date.txt / 20090214 / no], auto for reference_date.txt


###### Masking the bridge
#mintpy.network.maskFile = /insar-data/InSAR/old/road_roi_mask.h5
#miaplpy.inversion.mask = /insar-data/InSAR/old/road_roi_mask.h5

# mintpy.network.startDate       = 20240504
# mintpy.network.endDate         = 20241124
#mintpy.network.startDate       = 20161001
#mintpy.network.endDate         = 20190101

# ########## 4. correct_unwrap_error (optional)
# ## connected components (mintpy.load.connCompFile) are required for this step.
# ## SNAPHU (Chem & Zebker,2001) is currently the only unwrapper that provides connected components as far as we know.
# ## reference: Yunjun et al. (2019, section 3)
# ## supported methods:
# ## a. phase_closure          - suitable for highly redundant network
# ## b. bridging               - suitable for regions separated by narrow decorrelated features, e.g. rivers, narrow water bodies
# ## c. bridging+phase_closure - recommended when there is a small percentage of errors left after bridging
# mintpy.unwrapError.method          = bridging  #[bridging / phase_closure / bridging+phase_closure / no], auto for no
# mintpy.unwrapError.waterMaskFile   = auto  #[waterMask.h5 / no], auto for waterMask.h5 or no [if not found]
# mintpy.unwrapError.connCompMinArea = auto  #[1-inf], auto for 2.5e3, discard regions smaller than the min size in pixels

# ## phase_closure options:
# ## numSample - a region-based strategy is implemented to speedup L1-norm regularized least squares inversion.
# ##     Instead of inverting every pixel for the integer ambiguity, a common connected component mask is generated,
# ##     for each common conn. comp., numSample pixels are radomly selected for inversion, and the median value of the results
# ##     are used for all pixels within this common conn. comp.
# mintpy.unwrapError.numSample       = auto  #[int>1], auto for 100, number of samples to invert for common conn. comp.

# ## bridging options:
# ## ramp - a phase ramp could be estimated based on the largest reliable region, removed from the entire interferogram
# ##     before estimating the phase difference between reliable regions and added back after the correction.
# ## bridgePtsRadius - half size of the window used to calculate the median value of phase difference
# mintpy.unwrapError.ramp            = linear  #[linear / quadratic], auto for no; recommend linear for L-band data
# mintpy.unwrapError.bridgePtsRadius = auto  #[1-inf], auto for 50, half size of the window around end points

EOF


cd $DATA_DIR/$PROJECT_NAME/
#miaplpyApp $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt --dir ./miaplpy
miaplpyApp $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt --dostep load_data --dir ./miaplpy
miaplpyApp $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt --dostep phase_linking --dir ./miaplpy
miaplpyApp $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt --dostep concatenate_patches --dir ./miaplpy






