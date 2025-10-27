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


python /insar-data/InSAR/report/generateCSVandPlot.py --project-name $PROJECT_NAME \
    --miaplpy-version $MIAPLPY_VERSION \
    --shrink-eps-m 1.5 \
    --coh-thr 0.0

# python /insar-data/InSAR/generateHeatmap2.py --csv-total $CSV_TOTAL \
#     --csv-90d $CSV_90D \
#     --n_sections $N_SECTIONS \
#     --save $SAVE_HEATMAP \
#     --pillars $PILLARS 

python /insar-data/InSAR/report/calculateProb.py \
  --csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_noFilter.csv \
  --pillars $PILLARS \
  --out-csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk.csv \
  --tau-v 4 --tau-a 4 --tau-sigma 1.2 --m0 4.7
  #--save-heatmap /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk_heatmap.png \

python /insar-data/InSAR/report/plotHeatmap.py \
    --csv /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_bridge_risk.csv \
    --n_sections $N_SECTIONS \
    --save-prefix /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_heatmap


# python /insar-data/InSAR/generateVelocityMap.py \
#       --velocity /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/velocity.h5 \
#       --polygons /insar-data/$PROJECT_NAME/$PROJECT_NAME.geojson \
#       --geometry /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
#       --shrink-eps-m 0 \
#       --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocity_sections.png \
#       --cmap jet --dpi 300 --figsize 12 4

# python /insar-data/InSAR/generateVelocityMap.py \
#       --velocity /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/velocity.h5 \
#       --polygons /insar-data/$PROJECT_NAME/$PROJECT_NAME.geojson \
#       --geometry /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
#       --shrink-eps-m 0 \
#       --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap.png \
#       --out-abs /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap_absolute.png \
#       --abs-velocity --green-red-cmap --dpi 300 --figsize 12 4 
#     #   --temporal-coherence /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/temporalCoherence.h5 \
#     #   --coh-thr 0.4

# python /insar-data/InSAR/generateVelocityMap2.py \
#       --velocity /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/velocity.h5 \
#       --polygons /insar-data/$PROJECT_NAME/$PROJECT_NAME.geojson \
#       --geometry /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
#       --shrink-eps-m 0 \
#       --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap2.png \
#       --out-abs /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap2_absolute.png --green-red-cmap \
#       --abs-velocity --dpi 300 --figsize 12 4 \
#       --tc /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/temporalCoherence.h5 
#      # --coh-thr 0.4 
#      # --exclude-tc-eq1 
#      # --points --point-size 10 --step 1

python /insar-data/InSAR/report/generateVelocityMap.py \
      --velocity /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/velocity.h5 \
      --polygons /insar-data/$PROJECT_NAME/$PROJECT_NAME.geojson \
      --geometry /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
      --shrink-eps-m 0 \
      --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap.png \
      --out-abs /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap_absolute.png --green-red-cmap \
      --abs-velocity --dpi 300 --figsize 12 4 \
      --tc /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/temporalCoherence.h5 \
      --shrink-eps-m 3 \
      --coh-thr 0.0
     # --exclude-tc-eq1 
     # --points --point-size 10 --step 1

python /insar-data/InSAR/report/generateVelAccPlot.py --project-name $PROJECT_NAME --csv-path /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_filterWindow5.csv

python /insar-data/InSAR/report/generateReport.py --project $PROJECT_NAME --title $PROJECT_TITLE --start $START_DATE --end $END_DATE --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_report.pdf