## use this specifically for images where multiple cropped regions are saved as separate images

from optparse import OptionParser
import os
from PIL import Image

parser = OptionParser()
parser.add_option("-d", "--directory", dest="dir", type="string", action="store", help="path to directory containing .png images to be tiled")
(options, args) = parser.parse_args()

for file in os.listdir(options.dir):
	if file.split('.')[-1] == 'png':
		id = file.split('.')[0]+'_'+file.split('_')[-1].split('.')[0]
		x = 0
		y = 0
		count_x = 0
		count_y = 0
		length = 2500
		im = Image.open(options.dir+file)
		width, height = im.size
		tiledir = options.dir+'Tiled_'+id+'/'
		os.mkdir(tiledir)
		rem_width = width
		rem_height = height
		while y <= height:
			while x <= width:
				if (rem_width < 2500 and rem_height > 2500) and (rem_width > 0 and rem_height > 0): 
                                        box = (x, y, x+rem_width, y+length)
                                elif (rem_width > 2500 and rem_height < 2500) and (rem_width > 0 and rem_height > 0):
                                        box = (x, y, x+length, y+rem_height)
                                elif (rem_width < 2500 and rem_height < 2500) and (rem_width > 0 and rem_height > 0):
                                        box = (x, y, x+rem_width, y+rem_height)
                                else:
                                        box = (x, y, x+length, y+length)
				tile = im.crop(box)
				tile.save(tiledir+id+'_'+str(count_y)+'_'+str(count_x)+'.png')
				x += length
				count_x += 1
				rem_width -= 2500
			rem_height -= 2500
			y += length
			x = 0
			count_x = 0
			count_y += 1
			rem_width = width
	
