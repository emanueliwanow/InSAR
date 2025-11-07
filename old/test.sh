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


CSV_TOTAL="/insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_filterWindow5.csv"
CSV_90D="/insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_filterWindow3Last90Days.csv"
SAVE_HEATMAP="/insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_heatmap2.png"

# python /insar-data/InSAR/calibrateProbParams.py \
#   --csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_noFilter.csv \
#   --pillars $PILLARS \
#   --out-json /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_risk_calibration.json

# python /insar-data/InSAR/calculateProb.py \
#   --csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_noFilter.csv \
#   --pillars $PILLARS \
#   --out-csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk.csv \
#   --tau-v 4 --tau-a 4 --tau-sigma 1.2 --m0 4.7
#   #--save-heatmap /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk_heatmap.png \

# python /insar-data/InSAR/plotHeatmap.py \
#     --csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk.csv \
#     --n_sections $N_SECTIONS \
#     --save-prefix /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}

# save_kmz.py /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/velocity.h5 \
#     --g /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
#     --mask /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/maskPS.h5 \
#     --step 1 \
#     --output /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocity.kmz \
#     -u mm/year

save_kmz.py /insar-data/$PROJECT_NAME/sarvey/p2_coh70_ts.h5 \
    --g /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
    --step 1 \
    --output /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocity.kmz \
    -u mm/year

# view.py /insar-data/$PROJECT_NAME/miaplpy/network_delaunay_4/geo/geo_timeseries_ERA5_demErr.h5 \
#     -o /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_timeseries_ERA5_demErr.png \
#     --save --nodisplay \
#     --pts-yx 147 26 
# geocode.py /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/bridge_mask.h5 \
#            -l /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
#            --output /insar-data/$PROJECT_NAME/report/geo_bridge_mask.h5
# #--lalo 0.000070189 0.000084286 \
# save_kmz.py /insar-data/$PROJECT_NAME/report/geo_bridge_mask.h5 \
#     --step 1 \
#     --output /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_mask.kmz 

#--g /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \