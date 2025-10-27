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

python /insar-data/InSAR/calculateProb.py \
  --csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_noFilter.csv \
  --pillars $PILLARS \
  --out-csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk.csv \
  --tau-v 4 --tau-a 4 --tau-sigma 1.2 --m0 4.7
  #--save-heatmap /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk_heatmap.png \

python /insar-data/InSAR/plotHeatmap.py \
    --csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk.csv \
    --n_sections $N_SECTIONS \
    --save-prefix /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}
