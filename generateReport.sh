

PROJECT_NAME="Tocantins24meses"
PROJECT_TITLE="Tocantins"
START_DATE="2022-12-01"
END_DATE="2024-12-01"
MIAPLPY_VERSION="miaplpy"
CSV_TOTAL="/insar-data/$PROJECT_NAME/${PROJECT_NAME}_ts_filterWindow5.csv"
CSV_90D="/insar-data/$PROJECT_NAME/${PROJECT_NAME}_ts_filterWindow3Last90Days.csv"
N_SECTIONS=18
SAVE_HEATMAP="/insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_heatmap2.png"
PILLARS="'S2,S3','S4,S5','S6,S7','S8,S9','S10,S11','S12,S13','S14,S15','S16,S17','S18,S19','S20,S21','S22,S23','S24,S25','S26,S27','S28,S29'"


python /insar-data/InSAR/generateCSVandPlot.py --project-name $PROJECT_NAME --miaplpy-version $MIAPLPY_VERSION

python /insar-data/InSAR/generateHeatmap2.py --csv-total $CSV_TOTAL \
    --csv-90d $CSV_90D \
    --n_sections $N_SECTIONS \
    --save $SAVE_HEATMAP \
    --pillars $PILLARS 

python /insar-data/InSAR/generateVelocityMap.py \
      --velocity /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/velocity.h5 \
      --polygons /insar-data/$PROJECT_NAME/$PROJECT_NAME.geojson \
      --geometry /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
      --shrink-eps-m 0 \
      --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocity_sections.png \
      --cmap jet --dpi 300 --figsize 12 4

python /insar-data/InSAR/generateVelocityMap.py \
      --velocity /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/velocity.h5 \
      --polygons /insar-data/$PROJECT_NAME/$PROJECT_NAME.geojson \
      --geometry /insar-data/$PROJECT_NAME/$MIAPLPY_VERSION/network_delaunay_4/inputs/geometryRadar.h5 \
      --shrink-eps-m 0 \
      --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap.png \
      --out-abs /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_velocityMap_absolute.png \
      --abs-velocity --green-red-cmap --dpi 300 --figsize 12 4

python /insar-data/InSAR/generateVelAccPlot.py --project-name $PROJECT_NAME --csv-path /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_ts_filterWindow5.csv

python /insar-data/InSAR/generateReport.py --project $PROJECT_NAME --title $PROJECT_TITLE --start $START_DATE --end $END_DATE --out /insar-data/$PROJECT_NAME/report/${PROJECT_NAME}_report.pdf