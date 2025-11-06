# Example bbox over São Paulo (adjust to your AOI)
W=-46.86337675909827; S=-23.50686182723791; E=-46.8565737491036; N=-23.50304501762568

gdal_translate \
  -projwin_srs EPSG:4326 \
  -projwin $W $N $E $S \
  /insar-data/PS_prediction/predicted_PS/ml_model_data_southamerica/merged_southamerica.tif /insar-data/PS_prediction/predicted_PS/ml_model_data_southamerica/lambda_sp_clip.tif

gdal_translate -of PNG -scale /insar-data/PS_prediction/predicted_PS/ml_model_data_southamerica/lambda_sp_clip.tif /insar-data/PS_prediction/predicted_PS/ml_model_data_southamerica/lambda_sp_preview.png



