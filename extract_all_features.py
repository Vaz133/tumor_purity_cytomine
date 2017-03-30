from skimage import io, measure, color, segmentation, draw, feature
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image, ImageDraw
import csv
from scipy import ndimage
import pandas as pd
from optparse import OptionParser
import os
import numpy as np 

parser = OptionParser()
parser.add_option("-m", "--mask_image", dest="mask_image", type="string", action="store")
parser.add_option("-i", "--orig_image", dest="orig_image", type="string", action="store")
parser.add_option("-o", "--output", dest="output_file", type="string", action="store")
(options, args) = parser.parse_args()

mask_image = options.mask_image+' - Masks.png'
im_mask = io.imread(mask_image)
im_orig = io.imread(options.orig_image)
im_orig_gray = color.rgb2gray(im_orig)
im_orig_r = im_orig[:,:,0]
im_orig_g = im_orig[:,:,1]
im_orig_b = im_orig[:,:,2]
row_max = im_orig_r.shape[0]
col_max = im_orig_r.shape[1]

name = options.orig_image.split('.png')[0].split('/')[-1]

all_features_dict = {}

shape_features = ['area', 'bbox', 'centroid', 'convex_area', 'eccentricity',
 'equivalent_diameter', 'extent', 'filled_area', 'inertia_tensor',
 'inertia_tensor_eigvals', 'label', 'major_axis_length', 
 'minor_axis_length',
 'perimeter', 'solidity']


intensity_features = ['max_intensity', 'mean_intensity', 'min_intensity']

regions = measure.regionprops(im_mask, im_orig_gray)
regions_r = measure.regionprops(im_mask, im_orig_r)
regions_g = measure.regionprops(im_mask, im_orig_g)
regions_b = measure.regionprops(im_mask, im_orig_b)

for cell in range(len(regions)):
	for features in shape_features:
		all_features_dict.setdefault(cell, []).append(regions[cell][features])
	for features in intensity_features:
		all_features_dict[cell].append(regions[cell][features])
		all_features_dict[cell].append(regions_r[cell][features])
		all_features_dict[cell].append(regions_g[cell][features])
		all_features_dict[cell].append(regions_b[cell][features])

df = pd.DataFrame.from_dict(all_features_dict, orient='index')
df.columns = ['area', 'bbox', 'centroid', 'convex_area', 'eccentricity',
 'equivalent_diameter', 'extent', 'filled_area', 'inertia_tensor',
 'inertia_tensor_eigvals', 'label', 'major_axis_length', 
 'minor_axis_length',
 'perimeter', 'solidity', 
 'max_intensity_gray', 'max_intensity_r', 'max_intensity_g', 'max_intensity_b', 'mean_intensity_gray',
 'mean_intensity_r', 'mean_intensity_g', 'mean_intensity_b', 'min_intensity_gray', 'min_intensity_r',
 'min_intensity_g', 'min_intensity_b']

df['tile'] = name

centroid = df['centroid'].tolist()

## create lists to hold features
contrast_r_list = [[] for c in range(20)]
contrast_g_list = [[] for c in range(20)]
contrast_b_list = [[] for c in range(20)]

dissimilarity_r_list = [[] for c in range(20)]
dissimilarity_g_list = [[] for c in range(20)]
dissimilarity_b_list = [[] for c in range(20)]

homogeneity_r_list = [[] for c in range(20)]
homogeneity_g_list = [[] for c in range(20)]
homogeneity_b_list = [[] for c in range(20)]

energy_r_list = [[] for c in range(20)]
energy_g_list = [[] for c in range(20)]
energy_b_list = [[] for c in range(20)]

correlation_r_list = [[] for c in range(20)]
correlation_g_list = [[] for c in range(20)]
correlation_b_list = [[] for c in range(20)]

ASM_r_list = [[] for c in range(20)]
ASM_g_list = [[] for c in range(20)]
ASM_b_list = [[] for c in range(20)]

for i in centroid:

	#centroid_list  = re.findall(r"[^L(),']+", i)
	centroid_row = int(round(i[0]))
	centroid_col = int(round(i[1]))
	min_row = centroid_row - 32
	max_row = centroid_row + 32
	min_col = centroid_col - 32
	max_col = centroid_col + 32

	if centroid_row < 32:
		min_row = 0
		max_row = 64
	if centroid_col < 32:
		min_col = 0
		max_col = 64
	if centroid_row > (row_max - 32):
		max_row = row_max
		min_row = row_max - 64
	if centroid_col > (col_max - 32):
		max_col = col_max
		min_col = col_max - 64

	im_patch_r = im_orig_r[min_row:max_row, min_col:max_col]
	im_patch_g = im_orig_g[min_row:max_row, min_col:max_col]
	im_patch_b = im_orig_b[min_row:max_row, min_col:max_col]

	glcm_r = feature.greycomatrix(im_patch_r, range(1,6), [0, np.pi/4, np.pi/2, 3*np.pi/4],
		normed=False)
	glcm_g = feature.greycomatrix(im_patch_g, range(1,6), [0, np.pi/4, np.pi/2, 3*np.pi/4],
		normed=False)
	glcm_b = feature.greycomatrix(im_patch_b, range(1,6), [0, np.pi/4, np.pi/2, 3*np.pi/4],
		normed=False)

	contrast_r = feature.greycoprops(glcm_r, 'contrast').flatten().tolist()
	for i, j in enumerate(contrast_r):
		contrast_r_list[i].append(j)
	contrast_g = feature.greycoprops(glcm_g, 'contrast').flatten().tolist()
	for i, j in enumerate(contrast_g):
		contrast_g_list[i].append(j)
	contrast_b = feature.greycoprops(glcm_b, 'contrast').flatten().tolist()
	for i, j in enumerate(contrast_b):
		contrast_b_list[i].append(j)

	dissimilarity_r = feature.greycoprops(glcm_r, 'dissimilarity').flatten().tolist()
	for i, j in enumerate(dissimilarity_r):
		dissimilarity_r_list[i].append(j)
	dissimilarity_g = feature.greycoprops(glcm_g, 'dissimilarity').flatten().tolist()
	for i, j in enumerate(dissimilarity_g):
		dissimilarity_g_list[i].append(j)
	dissimilarity_b = feature.greycoprops(glcm_b, 'dissimilarity').flatten().tolist()
	for i, j in enumerate(dissimilarity_b):
		dissimilarity_b_list[i].append(j)

	homogeneity_r = feature.greycoprops(glcm_r, 'homogeneity').flatten().tolist()
	for i, j in enumerate(homogeneity_r):
		homogeneity_r_list[i].append(j)
	homogeneity_g = feature.greycoprops(glcm_g, 'homogeneity').flatten().tolist()
	for i, j in enumerate(homogeneity_g):
		homogeneity_g_list[i].append(j)
	homogeneity_b = feature.greycoprops(glcm_b, 'homogeneity').flatten().tolist()
	for i, j in enumerate(homogeneity_b):
		homogeneity_b_list[i].append(j)

	energy_r = feature.greycoprops(glcm_r, 'energy').flatten().tolist()
	for i, j in enumerate(energy_r):
		energy_r_list[i].append(j)
	energy_g = feature.greycoprops(glcm_g, 'energy').flatten().tolist()
	for i, j in enumerate(energy_g):
		energy_g_list[i].append(j)
	energy_b = feature.greycoprops(glcm_b, 'energy').flatten().tolist()
	for i, j in enumerate(energy_b):
		energy_b_list[i].append(j)

	correlation_r = feature.greycoprops(glcm_r, 'correlation').flatten().tolist()
	for i, j in enumerate(correlation_r):
		correlation_r_list[i].append(j)
	correlation_g = feature.greycoprops(glcm_g, 'correlation').flatten().tolist()
	for i, j in enumerate(correlation_g):
		correlation_g_list[i].append(j)
	correlation_b = feature.greycoprops(glcm_b, 'correlation').flatten().tolist()
	for i, j in enumerate(correlation_b):
		correlation_b_list[i].append(j)

	ASM_r = feature.greycoprops(glcm_r, 'ASM').flatten().tolist()
	for i, j in enumerate(ASM_r):
		ASM_r_list[i].append(int(j))
	ASM_g = feature.greycoprops(glcm_g, 'ASM').flatten().tolist()
	for i, j in enumerate(ASM_g):
		ASM_g_list[i].append(int(j))
	ASM_b = feature.greycoprops(glcm_b, 'ASM').flatten().tolist()
	for i, j in enumerate(ASM_b):
		ASM_b_list[i].append(int(j))

df_contrast_r = pd.DataFrame(contrast_r_list)
df_contrast_r = df_contrast_r.T
df_contrast_r.columns = ['contrast_r1', 'contrast_r2', 'contrast_r3',
	'contrast_r4', 'constrast_r5', 'contrast_r6', 'contrast_r7', 'contrast_r8', 'contrast_r9',
	'contrast_r10', 'contrast_r11', 'contrast_r12', 'contrast_r13', 'contrast_r14', 'contrast_r15',
	'contrast_r16', 'contrast_r17', 'contrast_r18', 'contrast_r19', 'contrast_r20']

df_contrast_g = pd.DataFrame(contrast_g_list)
df_contrast_g = df_contrast_g.T
df_contrast_g.columns = ['contrast_g1', 'contrast_g2', 'contrast_g3',
	'contrast_g4', 'constrast_g5', 'contrast_g6', 'contrast_g7', 'contrast_g8', 'contrast_g9',
	'contrast_g10', 'contrast_g11', 'contrast_g12', 'contrast_g13', 'contrast_g14', 'contrast_g15',
	'contrast_g16', 'contrast_g17', 'contrast_g18', 'contrast_g19', 'contrast_g20']

df_contrast_b = pd.DataFrame(contrast_b_list)
df_contrast_b = df_contrast_b.T
df_contrast_b.columns = ['contrast_b1', 'contrast_b2', 'contrast_b3',
	'contrast_b4', 'constrast_b5', 'contrast_b6', 'contrast_b7', 'contrast_b8', 'contrast_b9',
	'contrast_b10', 'contrast_b11', 'contrast_b12', 'contrast_b13', 'contrast_b14', 'contrast_b15',
	'contrast_b16', 'contrast_b17', 'contrast_b18', 'contrast_b19', 'contrast_b20']

df_dissimilarity_r = pd.DataFrame(dissimilarity_r_list)
df_dissimilarity_r = df_dissimilarity_r.T
df_dissimilarity_r.columns = ['dissimilarity_r1', 'dissimilarity_r2',
	'dissimilarity_r3', 'dissimilarity_r4', 'dissimilarity_r5', 'dissimilarity_r6', 'dissimilarity_r7', 
	'dissimilarity_r8', 'dissimilarity_r9', 'dissimilarity_r10', 'dissimilarity_r11', 'dissimilarity_r12',
	'dissimilarity_r13', 'dissimilarity_r14', 'dissimilarity_r15', 'dissimilarity_r16', 'dissimilarity_r17',
	'dissimilarity_r18', 'dissimilarity_r19', 'dissimilarity_r20']

df_dissimilarity_g = pd.DataFrame(dissimilarity_g_list)
df_dissimilarity_g = df_dissimilarity_g.T
df_dissimilarity_g.columns = ['dissimilarity_g1', 'dissimilarity_g2',
	'dissimilarity_g3', 'dissimilarity_g4', 'dissimilarity_g5', 'dissimilarity_g6', 'dissimilarity_g7', 
	'dissimilarity_g8', 'dissimilarity_g9', 'dissimilarity_g10', 'dissimilarity_g11', 'dissimilarity_g12',
	'dissimilarity_g13', 'dissimilarity_g14', 'dissimilarity_g15', 'dissimilarity_g16', 'dissimilarity_g17',
	'dissimilarity_g18', 'dissimilarity_g19', 'dissimilarity_g20']

df_dissimilarity_b = pd.DataFrame(dissimilarity_b_list)
df_dissimilarity_b = df_dissimilarity_b.T
df_dissimilarity_b.columns = ['dissimilarity_b1', 'dissimilarity_b2',
	'dissimilarity_b3', 'dissimilarity_b4', 'dissimilarity_b5', 'dissimilarity_b6', 'dissimilarity_b7', 
	'dissimilarity_b8', 'dissimilarity_b9', 'dissimilarity_b10', 'dissimilarity_b11', 'dissimilarity_b12',
	'dissimilarity_b13', 'dissimilarity_b14', 'dissimilarity_b15', 'dissimilarity_b16', 'dissimilarity_b17',
	'dissimilarity_b18', 'dissimilarity_b19', 'dissimilarity_b20']

df_homogeneity_r = pd.DataFrame(homogeneity_r_list)
df_homogeneity_r = df_homogeneity_r.T
df_homogeneity_r.columns = ['homogeneity_r1', 'homogeneity_r2', 'homogeneity_r3',
	'homogeneity_r4', 'homogeneity_r5', 'homogeneity_r6', 'homogeneity_r7', 'homogeneity_r8', 'homogeneity_r9',
	'homogeneity_r10', 'homogeneity_r11', 'homogeneity_r12', 'homogeneity_r13', 'homogeneity_r14',
	'homogeneity_r15', 'homogeneity_r16', 'homogeneity_r17', 'homogeneity_r18', 'homogeneity_r19', 'homogeneity_r20']

df_homogeneity_g = pd.DataFrame(homogeneity_g_list)
df_homogeneity_g = df_homogeneity_g.T
df_homogeneity_g.columns = ['homogeneity_g1', 'homogeneity_g2', 'homogeneity_g3',
	'homogeneity_g4', 'homogeneity_g5', 'homogeneity_g6', 'homogeneity_g7', 'homogeneity_g8', 'homogeneity_g9',
	'homogeneity_g10', 'homogeneity_g11', 'homogeneity_g12', 'homogeneity_g13', 'homogeneity_g14',
	'homogeneity_g15', 'homogeneity_g16', 'homogeneity_g17', 'homogeneity_g18', 'homogeneity_g19', 'homogeneity_g20']

df_homogeneity_b = pd.DataFrame(homogeneity_b_list)
df_homogeneity_b = df_homogeneity_b.T
df_homogeneity_b.columns = ['homogeneity_b1', 'homogeneity_b2', 'homogeneity_b3',
	'homogeneity_b4', 'homogeneity_b5', 'homogeneity_b6', 'homogeneity_b7', 'homogeneity_b8', 'homogeneity_b9',
	'homogeneity_b10', 'homogeneity_b11', 'homogeneity_b12', 'homogeneity_b13', 'homogeneity_b14',
	'homogeneity_b15', 'homogeneity_b16', 'homogeneity_b17', 'homogeneity_b18', 'homogeneity_b19', 'homogeneity_b20']

df_energy_r = pd.DataFrame(energy_r_list)
df_energy_r = df_energy_r.T
df_energy_r.columns = ['energy_r1', 'energy_r2', 'energy_r3', 'energy_r4', 'energy_r5',
	'energy_r6', 'energy_r7', 'energy_r8', 'energy_r9', 'energy_r10', 'energy_r11', 'energy_r12', 'energy_r13',
	'energy_r14', 'energy_r15', 'energy_r16', 'energy_r17', 'energy_r18', 'energy_r19', 'energy_r20']

df_energy_g = pd.DataFrame(energy_g_list)
df_energy_g = df_energy_g.T
df_energy_g.columns = ['energy_g1', 'energy_g2', 'energy_g3', 'energy_g4', 'energy_g5',
	'energy_g6', 'energy_g7', 'energy_g8', 'energy_g9', 'energy_g10', 'energy_g11', 'energy_g12', 'energy_g13',
	'energy_g14', 'energy_g15', 'energy_g16', 'energy_g17', 'energy_g18', 'energy_g19', 'energy_g20']

df_energy_b = pd.DataFrame(energy_b_list)
df_energy_b = df_energy_b.T
df_energy_b.columns = ['energy_b1', 'energy_b2', 'energy_b3', 'energy_b4', 'energy_b5',
	'energy_b6', 'energy_b7', 'energy_b8', 'energy_b9', 'energy_b10', 'energy_b11', 'energy_b12', 'energy_b13',
	'energy_b14', 'energy_b15', 'energy_b16', 'energy_b17', 'energy_b18', 'energy_b19', 'energy_b20']

df_correlation_r = pd.DataFrame(correlation_r_list)
df_correlation_r = df_correlation_r.T
df_correlation_r.columns = ['correlation_r1', 'correlation_r2', 'correlation_r3',
	'correlation_r4', 'correlation_r5', 'correlation_r6', 'correlation_r7', 'correlation_r8', 'correlation_r9',
	'correlation_r10', 'correlation_r11', 'correlation_r12', 'correlation_r13', 'correlation_r14', 'correlation_r15',
	'correlation_r16', 'correlation_r17', 'correlation_r18', 'correlation_r19', 'correlation_r20']

df_correlation_g = pd.DataFrame(correlation_g_list)
df_correlation_g = df_correlation_g.T
df_correlation_g.columns = ['correlation_g1', 'correlation_g2', 'correlation_g3',
	'correlation_g4', 'correlation_g5', 'correlation_g6', 'correlation_g7', 'correlation_g8', 'correlation_g9',
	'correlation_g10', 'correlation_g11', 'correlation_g12', 'correlation_g13', 'correlation_g14', 'correlation_g15',
	'correlation_g16', 'correlation_g17', 'correlation_g18', 'correlation_g19', 'correlation_g20']

df_correlation_b = pd.DataFrame(correlation_b_list)
df_correlation_b = df_correlation_b.T
df_correlation_b.columns = ['correlation_b1', 'correlation_b2', 'correlation_b3',
	'correlation_b4', 'correlation_b5', 'correlation_b6', 'correlation_b7', 'correlation_b8', 'correlation_b9',
	'correlation_b10', 'correlation_b11', 'correlation_b12', 'correlation_b13', 'correlation_b14', 'correlation_b15',
	'correlation_b16', 'correlation_b17', 'correlation_b18', 'correlation_b19', 'correlation_b20']

df_ASM_r = pd.DataFrame(ASM_r_list)
df_ASM_r = df_ASM_r.T
df_ASM_r.columns = ['ASM_r1', 'ASM_r2', 'ASM_r3', 'ASM_r4', 'ASM_r5', 'ASM_r6', 'ASM_r7', 'ASM_r8',
	'ASM_r9', 'ASM_r10', 'ASM_r11', 'ASM_r12', 'ASM_r13', 'ASM_r14', 'ASM_r15', 'ASM_r16', 'ASM_r17', 'ASM_r18',
	'ASM_r19', 'ASM_r20']

df_ASM_g = pd.DataFrame(ASM_g_list)
df_ASM_g = df_ASM_g.T
df_ASM_g.columns = ['ASM_g1', 'ASM_g2', 'ASM_g3', 'ASM_g4', 'ASM_g5', 'ASM_g6', 'ASM_g7', 'ASM_g8',
	'ASM_g9', 'ASM_g10', 'ASM_g11', 'ASM_g12', 'ASM_g13', 'ASM_g14', 'ASM_g15', 'ASM_g16', 'ASM_g17', 'ASM_g18',
	'ASM_g19', 'ASM_g20']

df_ASM_b = pd.DataFrame(ASM_b_list)
df_ASM_b = df_ASM_b.T
df_ASM_b.columns = ['ASM_b1', 'ASM_b2', 'ASM_b3', 'ASM_b4', 'ASM_b5', 'ASM_b6', 'ASM_b7', 'ASM_b8',
	'ASM_b9', 'ASM_b10', 'ASM_b11', 'ASM_b12', 'ASM_b13', 'ASM_b14', 'ASM_b15', 'ASM_b16', 'ASM_b17', 'ASM_b18',
	'ASM_b19', 'ASM_b20']

df = pd.concat([df, df_contrast_r, df_contrast_g, df_contrast_b,
df_dissimilarity_r, df_dissimilarity_g, df_dissimilarity_b,
df_homogeneity_r, df_homogeneity_g, df_homogeneity_b,
df_energy_r, df_energy_g, df_energy_b,
df_correlation_r, df_correlation_g, df_correlation_b,
df_ASM_r, df_ASM_g, df_ASM_b], axis=1)


df.to_csv(options.output_file+ '_all_objs_with_texture_NEW.csv', sep=',', index=False)
