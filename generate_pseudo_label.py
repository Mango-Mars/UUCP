#Running environment Python 3.7 Windows
from osgeo import gdal
from utils.XImage import CXImage
import numpy as np
from skimage.filters import threshold_otsu
import cv2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from osgeo import gdal
from tools.RSUtilityTool import CLabelPython, GetRoughness
from scipy.signal import find_peaks
import random
from scipy.io import loadmat
from skimage.filters import threshold_otsu
from scipy.signal import argrelextrema
import utils.eval_indicators as ei
from utils.tools import ignore_interference
import os
SEED = 971103
random.seed(SEED)
np.random.seed(SEED)
def out_fun(data, data_img, np_type,band_count, output_path):
    out_img = CXImage()
    out_img.Create(band_count, data_img.m_nLines, data_img.m_nSamples, np_type, output_path, options=['COMPRESS=LZW'])
    out_img.setHeaderInformation(data_img)
    out_img.WriteImgData(data)

def getTwoThreshold(water_index,single_threshold):
    part1 = water_index[(water_index > single_threshold)]
    high_threshold=np.percentile(part1,30,interpolation='midpoint')
    part2 = water_index[(water_index < single_threshold)]
    low_threshold=np.percentile(part2,70,interpolation='midpoint')

    if high_threshold < single_threshold:
        high_threshold = single_threshold
    if low_threshold > single_threshold:
        low_threshold = single_threshold
    return high_threshold,low_threshold

def two_threshold_water(water_index,certain_water,potential_water,single_threshold,min_region=0):
    seg_water = ((certain_water + potential_water) > 0)
    height, width = certain_water.shape
    seg_water = seg_water.astype(np.ubyte)
    label_img = np.ones(certain_water.shape, dtype=np.int32)
    region_main_pos = CLabelPython(seg_water, label_img, width, height, background=0, is8Neibor=True)
    region_main_num = int(region_main_pos.size / 4)

    for region in range(1, region_main_num+1):
        r_min_all = region_main_pos[4 * (region - 1)]
        r_max_all = region_main_pos[4 * (region - 1) + 1] + 1
        c_min_all = region_main_pos[4 * (region - 1) + 2]
        c_max_all = region_main_pos[4 * (region - 1) + 3] + 1
        label_img_temp = label_img[r_min_all:r_max_all, c_min_all: c_max_all]
        certain_water_temp = certain_water[r_min_all:r_max_all, c_min_all: c_max_all]
        potential_water_temp = potential_water[r_min_all:r_max_all, c_min_all: c_max_all]
        water_index_temp = water_index[r_min_all:r_max_all, c_min_all: c_max_all]

        r, c = np.where(label_img_temp == region)
        potenwial_cols = np.where(potential_water_temp[(r, c)] == 1)
        certain_cols = np.where(certain_water_temp[(r, c)] == 1)

        certain_water_count = certain_water_temp[(r, c)][certain_cols].size
        potential_water_count = potential_water_temp[(r, c)][potenwial_cols].size

        if(potential_water_count != 0 ):
            certain_percent = certain_water_count / potential_water_count
            if  (np.mean(water_index_temp[(r,c)]) < -0.2):
                certain_water_temp[(r, c)] = 0
                continue
        if r.size <= min_region:
            certain_water_temp[(r, c)] = 0
            continue

        certain_water_temp[(r, c)] = 1
    return certain_water

def compute_certrain_potential(water_index,data,ignore_mask,WI_mask):
    water_index[ignore_mask==1]="nan"
    # Remove outliers
    rows, cols = np.where((water_index < -1) | (water_index > 1))
    WI_mask_copy=WI_mask.copy()
    water_index[rows, cols] = 'nan'
    WI_mask_copy[rows, cols] = 'nan'
    mask_copy=ignore_mask.copy().astype(np.bool)
    mask_copy[rows,cols]=True
    invalid_mask = np.any(data < 0, axis=0)
    mask_copy[invalid_mask] = True
    WI_mask_copy[invalid_mask] = np.nan
    # Standardization
    water_index[~mask_copy] = RobustScaler(copy=False).fit_transform(water_index[~mask_copy].reshape(-1, 1)).reshape(-1)
    water_index[~mask_copy] = MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(water_index[~mask_copy].reshape(-1, 1)) \
        .reshape(-1)
    # Draw a histogram and calculate the peak
    n, bin, patches = plt.hist(water_index[~mask_copy], bins=50, density=True, facecolor='green', alpha=0.75)
    n, _= np.histogram(water_index[~mask_copy].ravel(), bins=50)
    peaks, _ = find_peaks(np.histogram(water_index[~mask_copy], bins=50)[0], height=int(0.005*len(water_index[~mask_copy])),distance=20)

    if len(peaks)==2:
        WI_threshold = threshold_otsu(water_index[~mask_copy])
    elif len(peaks)>2:
        WI_threshold = (bin[peaks[-1]]+bin[peaks[-2]])/2
    else:
        WI_threshold = threshold_otsu(water_index[WI_mask_copy == 1])

    high_WI,low_WI=getTwoThreshold(water_index[~mask_copy],WI_threshold)
    if high_WI<0:
        MNDWI_threshold=0
        high_WI, low_WI = getTwoThreshold(water_index[~mask_copy], MNDWI_threshold)
    # Calculate roughness
    roughness_threshold=0.7
    shape = water_index.shape
    nWidth, nHeight = shape[1], shape[0]
    roughness = np.ones(water_index.shape, dtype=np.float32)
    GetRoughness(roughness, water_index, 2, nWidth, nHeight, min_data=-1, max_data=1,
                 min_proc=np.float64(low_WI), max_proc=np.float64(high_WI), noise_filter=True)
    # Generate water body classification mask
    certain_water1 = (roughness > roughness_threshold)
    certain_water2 = (water_index >= high_WI).astype(int)
    certain_WI = (certain_water1 + certain_water2) > 0
    certain_no_WI = (water_index <= low_WI).astype(int)
    potential_WI = ((water_index > low_WI) & (water_index < high_WI)).astype(int)
    return certain_WI,certain_no_WI,potential_WI,WI_threshold

def read_img(filename):
    dataset = gdal.Open(filename)  # Opening a file
    im_width = dataset.RasterXSize  # Number of columns in the grid matrix
    im_height = dataset.RasterYSize  # The number of rows in the grid matrix
    im_geotrans = dataset.GetGeoTransform()  # Affine matrix, geodetic coordinates of the upper left pixel and pixel resolution
    im_proj = dataset.GetProjection()  # Map projection information, string representation
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset  # Close the object dataset and release memory
    return im_proj, im_geotrans, im_data, im_width, im_height
def read_tif(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return im_data
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

img_name=["T30TXQ_20180911_Bordeaux_summer.tif","T30TXQ_20190223_Bordeaux_winter.tif","T30TXR_20190223_Gironde_winter.tif","T30TYQ_20190222_Marmande_winter.tif",
          "T30UXU_20180708_Bretagne_summer.tif","T30UXU_20190223_Bretagne_winter.tif","T31TCH_20181023_Ariege_summer.tif","T31TCH_20190322_Ariege_winter.tif",
          "T31TCM_20180819_Chateauroux_summer.tif","T31TCM_20190225_Chateauroux_winter.tif","T31TFJ_20180927_Camargue_summer.tif","T31TFJ_20190331_Camargue_winter.tif",
          "T31TGL_20180828_Alpes_summer.tif","T32ULU_20180912_Alsace_summer.tif","T32ULU_20190321_Alsace_winter.tif","Guangzhou.tif","Wuhan.tif"]
for name in img_name:
    print(name)
    data1 = CXImage()
    data1.Open("..\Sentinel2\image\\" + name)
    create_path("Sentinel2/"+name[:-4])
    data = data1.GetData(np.float32)
    #DN value to projection rate
    data = data / 10000.
    data=data.transpose(2,0,1)
    #Read masks, including clouds, shadows, snow, oceans, inland areas, etc.
    ignore_mask = np.zeros((data.shape[1], data.shape[2]))
    ignore_mask = ignore_interference(ignore_mask,name)
    #The projection rate ranges from 0 to 1
    NIR=data[3,:,:].copy()
    ignore_mask[NIR<0]=1
    ignore_mask[NIR>1]=1
    mask=ignore_mask.copy()
    mask=mask.astype(np.bool)

    #Select samples for water index calculation based on NIR bands
    # NIR distinguishes between water and non-water bodies. It can be adjusted according to the region and image characteristics, or it can be set to 0.1 or 0.01 based on experience.
    #NIR_threshild=0.1
    NIR_threshold=0.01
    NIR_threshold_no_water=0.4
    # Initialize water_NIR and assign a default value of 2
    water_NIR = np.full((data.shape[1], data.shape[2]), 2, dtype=np.int64)
    water_NIR[NIR < NIR_threshold] = 1
    water_NIR[NIR >= NIR_threshold_no_water] = 0
    water_NIR[ignore_mask == 1] = 2
    # Constructing sample masks
    WI_mask = np.zeros((data.shape[1], data.shape[2]))
    WI_mask[(water_NIR == 0) | (water_NIR == 1)] = 1
    WI_mask[mask] = 0

    #Multiple index constraints
    #MNDWI
    MNDWI=(data[1, :, :] - data[4, :, :]) / (data[1, :, :] + data[4, :, :] + 1e-12)
    certain_MNDWI,certain_no_MNDWI,potential_MNDWI,MNDWI_threshold=compute_certrain_potential(MNDWI,data,ignore_mask,WI_mask)
    #NDWI
    NDWI=(data[1, :, :] - data[3, :, :]) / (data[1, :, :] + data[3, :, :] + 1e-12)
    certain_NDWI,certain_no_NDWI,potential_NDWI,NDWI_threshold=compute_certrain_potential(NDWI,data,ignore_mask,WI_mask)

    ###Dual water index filtering noise
    Difference=np.abs(MNDWI-NDWI)
    no_water_mask=((MNDWI<-0.2 )& (NDWI<-0.2)) | (((MNDWI>0) & (MNDWI<0.3))& (NDWI<=0)&(Difference>0.3)) | ((MNDWI<=0 )& ((NDWI>0) & (NDWI<0.3))& (Difference>0.3))
    water_mask=(MNDWI>0.3) | (NDWI>0.3)

    # Extracting certain water bodies and potential water bodies, i.e. simple samples and difficult samples
    #NIR_threshold can also be set to 0.05 or 0.1 based on experience
    NIR_threshold_no_water = 0.2
    certain_water = certain_MNDWI & certain_NDWI & (NIR<0.1)
    NIR_threshold = np.mean(NIR[certain_water == 1])
    if NIR_threshold<0.01 and np.sum(certain_water)/len(np.where(ignore_mask==0)[0])>0.005:
        NIR_threshold=0.05
    elif NIR_threshold>0.01 and np.sum(certain_water)/len(np.where(ignore_mask==0)[0])<0.005:
        NIR_threshold=0.05
    elif np.sum(certain_water)/len(np.where(ignore_mask==0)[0])>0.05:
        NIR_threshold=0.05
    #NIR_threshold=0.05 or NIR_threshold=0.1,adjust settings according to scene characteristics

    certain_no_water = certain_no_MNDWI & certain_no_NDWI
    certain_water_copy = certain_water.copy()
    potential_water = (potential_MNDWI | potential_NDWI) & (NIR < NIR_threshold)
    potential_no_water = (potential_MNDWI | potential_NDWI) & (NIR > NIR_threshold_no_water)
    # Remove the noise area
    potential_water[no_water_mask==1]=0
    potential_no_water[water_mask==1]=0
    #Spatial connectivity refinement of water bodies
    water_extraction = two_threshold_water(MNDWI, certain_water, potential_water, single_threshold=MNDWI_threshold,min_region=0)
    # Generate the final potential water layer
    potential_no_water=1-potential_no_water
    certain_no_water=1-certain_no_water
    certain_water_copy=certain_water_copy+0
    potential_water=water_extraction-certain_water_copy
    potential_water[potential_water==-1]=0
    result=water_extraction.astype(np.int32)
    result[ignore_mask==1]=0
    label = read_tif("..\inland_masks\\" + name)
    for arr in [certain_no_water, potential_no_water, potential_water, certain_water_copy]:
        arr[label == 255] = 2
        arr[NIR <0] = 2
    data1 = CXImage()
    data1.Open("..\image\\" + name)
    out_fun(potential_water+0,data1,np.float32,1,"Sentinel2/"+name[:-4]+"/potential.tif")
    out_fun(certain_water_copy+0,data1,np.float32,1,"Sentinel2/"+name[:-4]+"/certain.tif")
    out_fun(certain_no_water+0,data1,np.float32,1,"Sentinel2/"+name[:-4]+"/certain_no_water.tif")
    out_fun(potential_no_water+0,data1,np.float32,1,"Sentinel2/"+name[:-4]+"/potential_no_water.tif")
    out_fun(result,data1,np.float32,1,"Sentinel2/"+name[:-4] + "/"+name)



