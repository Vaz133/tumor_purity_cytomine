## Append annotation labels (cancer, normal, lymph, stromal) to feature files

from skimage import io
import numpy as np 
import pandas as pd 
import os
from optparse import OptionParser
from ast import literal_eval as make_tuple

parser = OptionParser()
parser.add_option("-f", "--feature_file_path", dest="feature_file_path", type="string", action="store")
parser.add_option("-r", "--region_image_path", dest="region_image_path", type="string", action="store")
(options, args) = parser.parse_args()

i = 1
for item in os.listdir(options.feature_file_path):
	if '.tsv' in item:
		label_list = []
		tile_list = []
		image_id = item.split('.tsv')[0].split('_')[-2] + '_' + item.split('.tsv')[0].split('_')[-1]
		df = pd.read_csv(options.feature_file_path+item, sep='\t')
		region_path = options.region_image_path+image_id+'.png'
		region_im = io.imread(region_path)
		for centroid in df['centroid']:
			centroid_tup = make_tuple(centroid)
			row_centroid = int(round(centroid_tup[0]))
			col_centroid = int(round(centroid_tup[1]))
			label = region_im[row_centroid, col_centroid]
			label_list.append(label)
			tile_list.append(region_path.split('/')[-2] + '_' + region_path.split('/')[-1])
		label_list = np.asarray(label_list)
		tile_list = np.asarray(tile_list)
		df['label'] = label_list
		df['tile'] = tile_list
		df = df[df.label != 0]
		print region_path.split('/')[-2] + '_' + region_path.split('/')[-1]
		if i == 1:
			df_new = df
			i += 1
		else: 
			# df_big = pd.read_csv(options.feature_file_path+'final_list.txt', sep='\t')
			df_new = pd.concat([df_new, df], ignore_index=True)
			df_new = df_new.drop('Unnamed: 0', 1)

df_new.to_csv(options.feature_file_path+'final_list.txt', sep='\t', index_label=False, index=False)

		
