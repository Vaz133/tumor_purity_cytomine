from skimage import io, measure, color, segmentation, draw
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image, ImageDraw
import csv
from scipy import ndimage
import pandas as pd 


im_mask = io.imread('segmented_Tiled_1038277_1/1038277_1_1_3.png - Masks.png')
im_orig = io.imread('Tiled_1038277_1/1038277_1_1_3.png')
im_orig_gray = color.rgb2gray(im_orig)
im_orig_r = im_orig[:,:,0]
im_orig_g = im_orig[:,:,1]
im_orig_b = im_orig[:,:,2]


all_features_dict = {}


shape_features = ['area', 'bbox', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity',
 'equivalent_diameter', 'euler_number', 'extent', 'filled_area', 'filled_image', 'image', 'inertia_tensor',
 'inertia_tensor_eigvals', 'label', 'local_centroid', 'major_axis_length', 
 'minor_axis_length', 'moments', 'moments_central', 'moments_hu', 'moments_normalized', 'orientation',
 'perimeter', 'solidity']


intensity_features = ['intensity_image', 'max_intensity', 'mean_intensity', 'min_intensity',
					  'weighted_centroid', 'weighted_local_centroid', 'weighted_moments',
					  'weighted_moments_central', 'weighted_moments_hu', 'weighted_moments_normalized']




regions = measure.regionprops(im_mask, im_orig_gray)
regions_r = measure.regionprops(im_mask, im_orig_r)
regions_g = measure.regionprops(im_mask, im_orig_g)
regions_b = measure.regionprops(im_mask, im_orig_b)


for cell in range(len(regions)):
	for feature in shape_features:
		all_features_dict.setdefault(cell, []).append(regions[cell][feature])
	for feature in intensity_features:
		all_features_dict[cell].append(regions[cell][feature])
		all_features_dict[cell].append(regions_r[cell][feature])
		all_features_dict[cell].append(regions_g[cell][feature])
		all_features_dict[cell].append(regions_b[cell][feature])

df = pd.DataFrame.from_dict(all_features_dict, orient='index')
df.columns = ['area', 'bbox', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity',
 'equivalent_diameter', 'euler_number', 'extent', 'filled_area', 'filled_image', 'image', 'inertia_tensor',
 'inertia_tensor_eigvals', 'label', 'local_centroid', 'major_axis_length', 
 'minor_axis_length', 'moments', 'moments_central', 'moments_hu', 'moments_normalized', 'orientation',
 'perimeter', 'solidity', 'intensity_image_gray', 'intensity_image_r', 'intensity_image_g', 'intensity_image_b',
 'max_intensity_gray', 'max_intensity_r', 'max_intensity_g', 'max_intensity_b', 'mean_intensity_gray',
 'mean_intensity_r', 'mean_intensity_g', 'mean_intensity_b', 'min_intensity_gray', 'min_intensity_r',
 'min_intensity_g', 'min_intensity_b', 'weighted_centroid_gray', 'weighted_centroid_r', 'weighted_centroid_g',
 'weighted_centroid_b', 'weighted_local_centroid_gray', 'weighted_local_centroid_r', 'weighted_local_centroid_g',
 'weighted_local_centroid_b', 'weighted_moments_gray', 'weighted_moments_r', 'weighted_moments_g', 
 'weighted_moments_b', 'weighted_moments_central_gray', 'weighted_moments_central_r', 
 'weighted_moments_central_g', 'weighted_moments_central_b', 'weighted_moments_hu_gray', 'weighted_moments_hu_r',
 'weighted_moments_hu_g', 'weighted_moments_hu_b', 'weighted_moments_normalized_gray', 
 'weighted_moments_normalized_r', 'weighted_moments_normalized_g', 'weighted_moments_normalized_b']

# print df['max_intensity_r'][0]

df.to_csv('ex.tsv', sep='\t')


	



