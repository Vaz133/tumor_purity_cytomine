## Original author: "Stévens Benjamin <b.stevens@ulg.ac.be>" 
## https://github.com/cytomine/




from cytomine import Cytomine
from cytomine.models import *
import numpy as np
from skimage import io
import string
from operator import sub
import os


#Cytomine connection parameters
cytomine_host="http://localhost-core"
cytomine_public_key="beca034c-9db1-44f2-b96a-1b5b51e4b066"
cytomine_private_key="cf9d9af2-d9bf-4cd9-9efb-6e0f3327f096"


#Connection to Cytomine Core
conn = Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)
# token = conn.build_token_key(username, True)
# print token
#Replace XXX by your values
id_user="azimiv"
id_project=5956833
# id_term = 5895229
#If you want to filter by image or term, uncomment the following line and in the get_annotations call
#If you want not to filter by user, comment the previous line
#id_image=XXX
#id_term=XXX


#This retrieve the JSON description of existing annotations with full details (wkt, GIS information)
#If you don't need full details (e.g. only to count the number of annotations), comment showWKT,showMeta,showGIS
#to speed up the query

image_instances = ImageInstanceCollection()
image_instances.project = id_project
image_instances = conn.fetch(image_instances)
images = image_instances.data()


identity = string.maketrans("", "")
for image in images:
  ## Cytomine image ID for a given annotated image
  if image.id == 6016373:
    annotations = conn.get_annotations(
                                     id_project = id_project,
                                     # id_user = id_user, 
                                     id_image = 6016373, 
                                     # id_term = id_term, 
                                     showWKT=True, 
                                     showMeta=True,
                                     showGIS=True,
                                     reviewed_only = False)

    ## get image dimensions
    image_w = image.width
    image_h = image.height

    ## create blank image
    im = np.zeros((image_h, image_w), dtype=np.uint8)

    ## iterate through annotations and append results to large image
    for a in annotations.data():
      if len(a.term) == 1:
        if a.image == image.id:
          ## Filter annotations for only pathologist's annotations using Cytomine user ID's
          if a.user == 6048201 or a.user == 6175413 or a.user == 6090392 or a.user == 6089221:

            ## get annotation coordinates 
            location = [s.translate(identity, "()POINTLYGMU").strip(" ") for s in a.location.encode("utf-8").split(', ')]
            location = [s.translate(identity, "()POINTLYGMU").strip(" ") for s in a.location.encode("utf-8").split(', ')]
            location = [tuple(int(round(float(y))) for y in reversed(x.split())) for x in location]
            location = [tuple(map(sub, (image_h, 2*s[1]), s)) for s in location]
            term = a.term[0]
            user = a.user

            ## get bounding box coordinates
            max_x = max(zip(*location)[1])
            max_y = max(zip(*location)[0])
            min_x = min(zip(*location)[1])
            min_y = min(zip(*location)[0])

            # if annotation corresponds to a region annotation
            if term == 5983163 or term == 5983213 or term == 5983193 or term == 5983239:

              ## build the URL and download the cropped region
              url = 'http://localhost-core/api/imageinstance/'+str(a.image)+'/window-'+str(min_x)+'-'+str(min_y)+'-'+str(max_x-min_x)+'-'+str(max_y-min_y)+'.png?&mask=true'
              if not os.path.exists('/Users/vahidazimi/Desktop/Research/young_segmentation/Cytomine/annotation_ROIs/'+str(a.image)+'/'):
                  os.makedirs('/Users/vahidazimi/Desktop/Research/young_segmentation/Cytomine/annotation_ROIs/'+str(a.image)+'/')
              img = conn.fetch_url_into_file(url,
                                              '/Users/vahidazimi/Desktop/Research/young_segmentation/Cytomine/annotation_ROIs/'+str(a.image)+'/'
                                              + str(a.id)+'.png', is_image=True)

              # read the image of the cropped region
              try:
                im_crop = io.imread(
                               '/Volumes/TOSHIBA/Images/annotation_ROIs/'
                                + str(a.id)+'.png', dtype=np.uint8)[:,:,0]
              except:
                continue

              ## label pixels in the region based on the Cytomine ontology term ID
              ## and add region to large image
              if np.count_nonzero(im_crop) != 0: ## to skip 'blank' images caused by Cytomine bug
                if term == 5983163:
                  im_crop[im_crop != 0] = 1
                  im[min_y:min_y+im_crop.shape[0], min_x:min_x+im_crop.shape[1]] = im_crop
                elif term == 5983213:
                  im_crop[im_crop != 0] = 2
                  im[min_y:max_y, min_x:max_x] = im_crop
                elif term == 5983193:
                  im_crop[im_crop != 0] = 3
                  im[min_y:max_y, min_x:max_x] = im_crop
                elif term == 5983239:
                  im_crop[im_crop != 0] = 4
                  im[min_y:max_y, min_x:max_x] = im_crop

            ## do the same for point annotations
            if term == 5983151: # cancer cell
              im[tuple(zip(*location))] = 1
            elif term == 5983205: # normal cell
              im[tuple(zip(*location))] = 2
            elif term == 5983179: # lymphocyte
              im[tuple(zip(*location))] = 3
            elif term == 5983227: # stromal cell
              im[tuple(zip(*location))] = 4


 #   split large image into tiles and save
    y = 0
    x = 0
    length = 2500
    y_count = 0
    x_count = 0
    x_dim = 2500
    y_dim = 2500
    rem_width = image.width
    rem_height = image.height
    while x <= image.width:
      while y <= image.height:
        if (rem_width < 2500 and rem_height > 2500) and (rem_width > 0 and rem_height > 0):
          im_crop = im[y:y_dim, x:image.width]
        elif (rem_width > 2500 and rem_height < 2500) and (rem_width > 0 and rem_height > 0):
          im_crop = im[y:image.height, x:x_dim]
        elif (rem_width < 2500 and rem_height < 2500) and (rem_width > 0 and rem_height > 0):
          im_crop = im[y:image.height, x:image.width]
        else:
          im_crop = im[y:y_dim, x:x_dim]
        if not os.path.exists('/Users/vahidazimi/Desktop/Research/young_segmentation/Cytomine/' + str(image.id)):
          os.makedirs('/Users/vahidazimi/Desktop/Research/young_segmentation/Cytomine/' + str(image.id))
        if 'crop' not in image.originalFilename:
          io.imsave('/Volumes/TOSHIBA/Images/annotated_tiles/' + str(image.id) + '/'+str(x_count)+'_'+str(y_count)+'.png', im_crop)
        else:
          io.imsave('/Volumes/TOSHIBA/Images/annotated_tiles/' + str(image.id) + '/'+str(y_count)+'_'+str(x_count)+'.png', im_crop)
        y += length
        y_dim += length
        y_count += 1
        rem_height -= 2500
        print rem_width, rem_height
      y_dim = 2500
      x += length
      x_dim += length
      rem_width -= 2500
      y = 0
      y_count = 0
      x_count += 1
      rem_height = image.height






                                       
                                       





