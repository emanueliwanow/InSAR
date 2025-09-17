cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: bbf72fd2-453b-478c-b1e2-432b4b684556
EOF
chmod 600 ~/.cdsapirc
PROJECT_NAME="Curaca"
DATA_DIR="/insar-data"
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
miaplpy.subset.lalo         = -3.277:-3.274,-59.856:-59.854

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
miaplpy.inversion.rangeWindow         = 10
miaplpy.inversion.azimuthWindow       = 10
miaplpy.inversion.shpTest             = auto     #[ks,ad,ttest]; auto->ks
miaplpy.inversion.phaseLinkingMethod  = sequential_EMI  #[EVD,EMI,PTA,sequential_*]
miaplpy.inversion.sbw_connNum         = auto
miaplpy.inversion.PsNumShp            = 10        # neighbors used for PS candidate
#miaplpy.inversion.mask                = /insar-data/InSAR/old/road_roi_mask.h5

############################
# 4) Time series threshold (MiaplPy)
############################
# Keep a bit permissive while tuning; you can raise later
miaplpy.timeseries.minTempCoh         = 0.2

############################
# 5) MintPy options (run by MiaplPy steps)
############################

# Reference point (keep on stable man-made target near deck)
#mintpy.reference.lalo                 = -3.275948021066561,-59.85507425475626
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
mintpy.networkInversion.minTempCoh    = 0.3      #[0.0-1.0] temporal coherence mask

########## 5.5 Geocode
mintpy.geocode                        = auto
mintpy.geocode.SNWE                   = auto
mintpy.geocode.laloStep               = 0.000070189, 0.000084286
mintpy.geocode.interpMethod           = auto
mintpy.geocode.fillValue              = auto


###### Masking the bridge
#mintpy.network.maskFile = /insar-data/Curaca/mask_bridge.h5
#miaplpy.inversion.mask = /insar-data/Curaca/mask_bridge.h5
EOF


cd $DATA_DIR/$PROJECT_NAME/
miaplpyApp $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt --dir ./miaplpy








#cat > $DATA_DIR/$PROJECT_NAME/$PROJECT_NAME.txt << EOF
# ################
# miaplpy.load.processor      = isce  #[isce,snap,gamma,roipac], auto for isceTops
# miaplpy.load.updateMode     = no  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
# miaplpy.load.compression    = gzip  #[gzip / lzf / no], auto for no.
# miaplpy.load.autoPath       = yes    # [yes, no] auto for no
        
# miaplpy.load.slcFile        = ../merged/SLC/*/*.slc.full  #[path2slc_file]
# ##---------for ISCE only:
# miaplpy.load.metaFile       = ../reference/IW*.xml
# miaplpy.load.baselineDir    = ../baselines
# ##---------geometry datasets:
# miaplpy.load.demFile          = ../merged/geom_reference/hgt.rdr.full
# miaplpy.load.lookupYFile      = ../merged/geom_reference/lat.rdr.full
# miaplpy.load.lookupXFile      = ../merged/geom_reference/lon.rdr.full
# miaplpy.load.incAngleFile     = ../merged/geom_reference/los.rdr.full
# miaplpy.load.azAngleFile      = ../merged/geom_reference/los.rdr.full
# miaplpy.load.shadowMaskFile   = ../merged/geom_reference/shadowMask.rdr.full
# miaplpy.load.waterMaskFile    = None
# ##---------interferogram datasets:
# #miaplpy.load.unwFile        = ./inverted/interferograms_single_reference/*/*fine*.unw
# #miaplpy.load.corFile        = ./inverted/interferograms_single_reference/*/*fine*.cor
# #miaplpy.load.connCompFile   = ./inverted/interferograms_single_reference/*/*.unw.conncomp
# #miaplpy.load.unwFile        = ./inverted/interferograms_mini_stacks/*/*fine*.unw
# #miaplpy.load.corFile        = ./inverted/interferograms_mini_stacks/*/*fine*.cor
# #miaplpy.load.connCompFile   = ./inverted/interferograms_mini_stacks/*/*.unw.conncomp
# miaplpy.load.unwFile        = ./inverted/interferograms_delaunay_4/*/*fine*.unw
# miaplpy.load.corFile        = ./inverted/interferograms_delaunay_4/*/*fine*.cor
# miaplpy.load.connCompFile   = ./inverted/interferograms_delaunay_4/*/*.unw.conncomp
        
# ##---------subset (optional):
# ## if both yx and lalo are specified, use lalo option unless a) no lookup file AND b) dataset is in radar coord
# #miaplpy.subset.lalo         = -6.561931864005232:-6.558154602203117,-47.46404884577386:-47.45525789019609
# # Tocantins
# #miaplpy.subset.lalo         = -6.560776280233487:-6.559429285987968,-47.4626109537542:-47.45722534955109


# # Autaz Mirim
# #miaplpy.subset.lalo         = -3.285728172223284:-3.282516054019862,-59.86503513622157:-59.86266781146101


# # Curaca
# miaplpy.subset.lalo         = -3.276568529410899:-3.274196318088118,-59.85551604445834:-59.85386491901124

# # MiaplPy options 
# #miaplpy.multiprocessing.numProcessor   = 40
# miaplpy.interferograms.networkType = delaunay #[mini_stacks, single_reference, sequential, delaunay]

# ## Mintpy options
# #mintpy.compute.cluster     = local  # if dask is not available, set this option to no 
# #mintpy.compute.numWorker   = 8 

# #mintpy.reference.lalo     = -6.560609867799984,-47.46213912721969

# #Autaz Mirim
# #mintpy.reference.lalo     = -3.282984576229724,-59.86302168471724

# # Curaca
# mintpy.reference.lalo     = -3.275948021066561,-59.85507425475626

# mintpy.reference.maskFile      = no #[filename / no], auto for maskConnComp.h5
# #mintpy.troposphericDelay.method = no

# ########## 2,3. Perform patch wise phase linking and concatenate patches
# ## window sizes are used in step 2, 3,
# miaplpy.inversion.patchSize                = auto   # patch size (n*n) to divide the image for parallel processing, auto for 200
# miaplpy.inversion.ministackSize            = auto   # number of images in each ministack, auto for 10
# miaplpy.inversion.rangeWindow              = 15   # range window size for searching SHPs, auto for 15
# miaplpy.inversion.azimuthWindow            = 15  # azimuth window size for searching SHPs, auto for 15
# miaplpy.inversion.shpTest                  = auto   # [ks, ad, ttest] auto for ks: kolmogorov-smirnov test
# miaplpy.inversion.phaseLinkingMethod       = auto   # [EVD, EMI, PTA, sequential_EVD, sequential_EMI, sequential_PTA, SBW], auto for sequential_EMI
# miaplpy.inversion.sbw_connNum              = auto   # auto for 10, number of consecutive interferograms
# miaplpy.inversion.PsNumShp                 = 10   # auto for 10, number of shps for ps candidates
# miaplpy.inversion.mask                     = auto   # mask file for phase inversion, auto for None

# miaplpy.timeseries.minTempCoh             = 0.3    # auto for 0.5

# ## geocode override
# mintpy.geocode              = auto  #[yes / no], auto for yes
# mintpy.geocode.SNWE         = -3.276568529410899,-3.274196318088118,-59.85551604445834,-59.85386491901124  #[-1.2,0.5,-92,-91 / none ], auto for none, output extent in degree
# mintpy.geocode.SNWE         = auto
# mintpy.geocode.laloStep     = 0.00002, 0.00002  #[-0.000555556,0.000555556 / None], auto for None, output resolution in degree
# mintpy.geocode.interpMethod = auto  #[nearest], auto for nearest, interpolation method
# mintpy.geocode.fillValue    = auto  #[np.nan, 0, ...], auto for np.nan, fill value for outliers.
# mintpy.networkInversion.minTempCoh  = 0.3  #[0.0-1.0], auto for 0.7, min temporal coherence for mask
# mintpy.troposphericDelay.weatherDir   = $DATA_DIR/$PROJECT_NAME/ #[path2directory], auto for WEATHER_DIR or "./"
# EOF
