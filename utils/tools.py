from osgeo import gdal
from utils.XImage import CXImage
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.signal import find_peaks
import random
from skimage.filters import threshold_otsu
import os
import argparse
from model.CAMFNet import CAMFNet

def read_tif(filename):
    dataset = gdal.Open(filename)  # Opening a file
    im_width = dataset.RasterXSize  # The number of columns in the grid matrix
    im_height = dataset.RasterYSize  # The number of rows in the grid matrix
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return im_data

def ignore_interference(ignore_mask,name):
    #remove snow,
    if name=="T31TGL_20180828_Alpes_summer.tif":
        snow_mask = read_tif(
            "..\Sentinel2\snow_mask\SENTINEL2B_20180828-103013-461_L2B-SNOW_T31TGL_D_V2-0\SENTINEL2B_20180828-103013-461_L2B-SNOW_T31TGL_C_V2-0\SENTINEL2B_20180828-103013-461_L2B-SNOW_T31TGL_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
    if name=="T31TCH_20181023_Ariege_summer.tif":
        snow_mask = read_tif(
            "..\Sentinel2\snow_mask\SENTINEL2B_20181023-105107-455_L2B-SNOW_T31TCH_D_V2-0\SENTINEL2B_20181023-105107-455_L2B-SNOW_T31TCH_C_V2-0\SENTINEL2B_20181023-105107-455_L2B-SNOW_T31TCH_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T31TCH_20190322_Ariege_winter.tif":
        snow_mask = read_tif(
            "..\Sentinel2\snow_mask\SENTINEL2B_20190322-105912-800_L2B-SNOW_T31TCH_D_V2-0\SENTINEL2B_20190322-105912-800_L2B-SNOW_T31TCH_C_V2-0\SENTINEL2B_20190322-105912-800_L2B-SNOW_T31TCH_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T32ULU_20180912_Alsace_summer.tif":
        snow_mask = read_tif(
            "..\Sentinel2\snow_mask\SENTINEL2A_20180912-103308-022_L2B-SNOW_T32ULU_D_V2-0\SENTINEL2A_20180912-103308-022_L2B-SNOW_T32ULU_C_V2-0\SENTINEL2A_20180912-103308-022_L2B-SNOW_T32ULU_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T32ULU_20190321_Alsace_winter.tif":
        snow_mask = read_tif(
            "..\Sentinel2\snow_mask\SENTINEL2A_20190321-103733-710_L2B-SNOW_T32ULU_D_V2-0\SENTINEL2A_20190321-103733-710_L2B-SNOW_T32ULU_C_V2-0\SENTINEL2A_20190321-103733-710_L2B-SNOW_T32ULU_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T31TFJ_20180927_Camargue_summer.tif":
        snow_mask = read_tif(
            "..\Sentinel2\snow_mask\SENTINEL2B_20180927-103016-458_L2B-SNOW_T31TFJ_D_V2-0\SENTINEL2B_20180927-103016-458_L2B-SNOW_T31TFJ_C_V2-0\SENTINEL2B_20180927-103016-458_L2B-SNOW_T31TFJ_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
    if name=="T31TFJ_20190331_Camargue_winter.tif":
        snow_mask = read_tif(
            "..\Sentinel2\snow_mask\SENTINEL2A_20190331-103854-278_L2B-SNOW_T31TFJ_D_V2-0\SENTINEL2A_20190331-103854-278_L2B-SNOW_T31TFJ_C_V1-7\SENTINEL2A_20190331-103854-278_L2B-SNOW_T31TFJ_C_V1-7_SNW_R2.tif")
        snow_mask= cv2.resize(snow_mask,dsize=None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1

    #Remove clouds, shadows
    if name!="Guangzhou.tif" and name != "Wuhan.tif":
        shanow_mask=read_tif("..\Sentinel2\MG2\\"+name)
        for value in np.unique(shanow_mask):
            binary = format(value, '08b')
            if binary[0] == "1" or binary[1] == "1" or binary[2] == "1" or binary[3] == "1" or binary[4] == "1"  or binary[6] == "1":
                ignore_mask[shanow_mask == value] = 1

    #Remove the Ocean
    label = read_tif("..\inland_masks\\" + name)
    ignore_mask[label==255]=1

    return ignore_mask

def create_model(cfg):
    # Network definition
    if cfg["model"]=="CAMFNet":
        # Setting up the model
        parser = argparse.ArgumentParser(description='train CAMFNet')
        parser.add_argument('--in_channels', type=int, default=6,
                            help='Number of input channels')
        parser.add_argument('--num_classes', type=int, default=2,
                            help='Number of output categories')
        parser.add_argument('--block_channels', type=tuple, default=(96, 128, 192, 256),
                            help='Number of module channels')
        parser.add_argument('--reduction_ratio', type=float, default=1.0,
                            help='reduction_ratio')
        parser.add_argument('--inner_dim', type=int, default=128,
                            help='inner_dim')
        parser.add_argument('--num_blocks', type=tuple, default=(1, 1, 1, 1),
                            help='num_blocks')
        args = parser.parse_args()
        model = CAMFNet(args)
    return model

def getTwoThreshold(water_index,single_threshold):
    part1 = water_index[(water_index >= single_threshold)]
    high_threshold=np.percentile(part1,30,interpolation='midpoint')
    part2 = water_index[(water_index < single_threshold)]
    low_threshold=np.percentile(part2,70,interpolation='midpoint')

    if high_threshold < single_threshold:
        high_threshold = single_threshold
    if low_threshold > single_threshold:
        low_threshold = single_threshold

    return high_threshold,low_threshold

#Calculate loss weight
def compute_weight(data,ignore_mask):
    data = data / 10000.
    NIR=data[3,:,:].copy()
    mask=ignore_mask.copy()
    mask=mask.astype(np.bool_)

    #Select samples for water index calculation based on NIR bands
    # NIR distinguishes between water and non-water bodies. It can be adjusted according to the region and image characteristics, or it can be set to 0.1 or 0.01 based on experience.
    NIR_threshold=0.1
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

    #
    water_index=(data[1, :, :] - data[4, :, :]) / (data[1, :, :] + data[4, :, :] + 1e-12)
    water_index[ignore_mask==1]="nan"
    # Remove outliers
    rows, cols = np.where((water_index < -1) | (water_index > 1))
    WI_mask_copy=WI_mask.copy()
    water_index[rows, cols] = 'nan'
    WI_mask_copy[rows, cols] = 'nan'
    mask_copy=ignore_mask.copy().astype(np.bool_)
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
        high_WI=0
        high_WI, low_WI = getTwoThreshold(water_index[~mask_copy], high_WI)

    # Calculate weight
    weight=np.zeros((data.shape[1],data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if water_index[i,j]>high_WI or water_index[i,j]<low_WI:
                weight[i,j]=1
            elif water_index[i,j]<=high_WI and water_index[i,j]>WI_threshold :
                weight[i,j]=(water_index[i,j]-WI_threshold)/(high_WI-WI_threshold)
            elif water_index[i,j]>=low_WI and water_index[i,j]<=WI_threshold :
                weight[i,j]=(WI_threshold -water_index[i,j])/(WI_threshold -low_WI)
    ##Give a bias to weight
    weight=weight+0.5
    return weight

def get_inviad_mask(pseudo_label,MG2,label_island,data,name):
    '''
    :param pseudo_label:
    :param MG2:Cloud, shadow mask
    :param label_island: For removing ocean,Not used during training
    :param data:Original input image
    :param name:
    :return:
    '''
    #Read masks, including clouds, shadows, snow, oceans, inland areas, etc.
    ignore_mask = np.zeros((data.shape[1], data.shape[2]))
    if name=="T31TGL_20180828_Alpes_summer":
        snow_mask = read_tif(
            "../Sentinel2/snow/SENTINEL2B_20180828-103013-461_L2B-SNOW_T31TGL_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
    if name=="T31TCH_20181023_Ariege_summer":
        snow_mask = read_tif(
            "../Sentinel2/snow/SENTINEL2B_20181023-105107-455_L2B-SNOW_T31TCH_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T31TCH_20190322_Ariege_winter":
        snow_mask = read_tif(
            "../Sentinel2/snow/SENTINEL2B_20190322-105912-800_L2B-SNOW_T31TCH_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T32ULU_20180912_Alsace_summer":
        snow_mask = read_tif(
            "../Sentinel2/snow/SENTINEL2A_20180912-103308-022_L2B-SNOW_T32ULU_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T32ULU_20190321_Alsace_winter":
        snow_mask = read_tif(
            "../Sentinel2/snow/SENTINEL2A_20190321-103733-710_L2B-SNOW_T32ULU_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
        kernel=np.ones((3,3),np.uint8)
        ignore_mask = cv2.dilate(ignore_mask, kernel=kernel, iterations=40)
        pass
    if name=="T31TFJ_20180927_Camargue_summer":
        snow_mask = read_tif(
            "../Sentinel2/snow/SENTINEL2B_20180927-103016-458_L2B-SNOW_T31TFJ_C_V2-0_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1
    if name=="T31TFJ_20190331_Camargue_winter":
        snow_mask = read_tif(
            "../Sentinel2/snow/SENTINEL2A_20190331-103854-278_L2B-SNOW_T31TFJ_C_V1-7_SNW_R2.tif")
        snow_mask = cv2.resize(snow_mask, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ignore_mask[snow_mask==100]=1
        ignore_mask[snow_mask==205]=1

    for value in np.unique(MG2):
        binary = format(value, '08b')
        if binary[0] == "1" or binary[1] == "1" or binary[2] == "1" or binary[3] == "1" or binary[4] == "1"  or binary[6] == "1":
            ignore_mask[MG2 == value] = 1
    pseudo_label[ignore_mask==1]=3
    NIR=data[3,:,:].copy()/10000.

    ignore_mask[label_island==255]=1
    ignore_mask[NIR<0]=1
    ignore_mask[NIR>1]=1

    pseudo_label[label_island == 255] = 2
    pseudo_label[NIR < 0] = 2
    pseudo_label[NIR > 1] = 2

    return pseudo_label,ignore_mask
