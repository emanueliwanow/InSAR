import rasterio, matplotlib.pyplot as plt
from rasterio.windows import from_bounds

# open raster
src = rasterio.open("/insar-data/PS_prediction/predicted_PS/ml_model_data_southamerica/merged_southamerica.tif")

# define AOI bbox (W,S,E,N in the raster CRS, usually EPSG:4326)
# Tiete
W=-48
S=-26
E=-44
N=-22

# JK
# W=-47.83499163769471
# S=-15.83023653303048
# E=-47.82491182208675
# N=-15.8172351480543

# Tocantins
# W=-49
# S=-10
# E=-45
# N=-5
win = from_bounds(W, S, E, N, src.transform)

arr = src.read(1, window=win)
plt.imshow(arr, cmap='viridis')  # quick look
plt.title("Predicted PS per pixel (λ)")
plt.colorbar(label="PS expected per ~3\" pixel")
plt.savefig("/insar-data/PS_prediction/predicted_PS/ml_model_data_southamerica/lambda_sp_preview.png")
plt.close()

# close raster
src.close()
# -47.45708653503829,-6.558628035205263,0 -47.46292715952661,-6.561809858759957,0 