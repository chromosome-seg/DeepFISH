#
#
#
####----------------------------Import libs ----------------------------------------------------------------------
import sys
import itertools
import numpy as np # linear algebra
from scipy import ndimage as nd
import skimage as sk
from skimage import segmentation as skg
from skimage.measure import label
from skimage import io
import cv2
from matplotlib import pyplot as plt

import h5py
import string
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../dataset/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#
#
#
def extractParticles_2(greyIm, LabIm, smooth_grey_edge = 0.75):
    #print 'start', greyIm.dtype, LabIm.dtype
    LabelImg= LabIm
    GreyImg = greyIm
    locations = nd.find_objects(LabelImg)
    #print locations
    i=1
    extracted_images=[]
    for loc in locations:
        
        lab_image = np.copy(LabelImg[loc])
        grey_image = np.copy(GreyImg[loc])
        
        lab_image[lab_image != i] = 0
        grey_image[lab_image != i] = 0
        nicer_grey = np.maximum.reduce((nd.filters.gaussian_filter(grey_image, sigma=0.75), grey_image))
        extracted_images.append(nicer_grey)
        i=i+1
    return extracted_images

def ResizeImages(ImList):
        '''Find the largest width and height of images belonging to a list.
        Returns a list of images of same width/height
        '''
        maxwidth = 0
        maxheight = 0
        if len(np.shape(ImList[0])) == 3:
            components = np.shape(ImList[0])[2]
        imtype = ImList[0].dtype
        for i in range(len(ImList)):
            width = np.shape(ImList[i])[1]#width=column
            height = np.shape(ImList[i])[0]#height=line
            #print "width:height",width,":",height
            if width > maxwidth:maxwidth = width
            if height > maxheight:maxheight = height
        #print "maxwidth:maxheight",maxwidth,":",maxheight
        NewList=[]
        for i in range(0,len(ImList)):
            width = np.shape(ImList[i])[1]
            height = np.shape(ImList[i])[0]

            diffw = maxwidth-width
            startw = round(diffw/2)
            diffh = maxheight-height
            starth = int(round(diffh/2))
            startw = int(round(diffw/2))
            if len(np.shape(ImList[0])) == 3:
                newIm = np.zeros((maxheight,maxwidth,components), dtype=imtype)
                newIm[starth:starth+height,startw:startw+width,:] = ImList[i][:,:,:]
                NewList.append(newIm)
            if len(np.shape(ImList[0])) == 2:
                newIm = np.zeros((maxheight,maxwidth), dtype = imtype)
                newIm[starth:starth+height,startw:startw+width]=ImList[i][:,:]
                NewList.append(newIm)
        return NewList
    
### kept
def translate(image, vector):
    #print 'vector',vector, vector[0]
    image = np.roll(image, int(vector[0]), axis = 0)
    image = np.roll(image, int(vector[1]), axis = 1)
    return image

###kept
def add_mask_to_image(image, threshold, mask_value = 1):
    mask = nd.binary_closing(nd.binary_opening(image > threshold))
    image = np.dstack((image, mask_value*mask))
    return image

#Let's generate chromosome overlapps:
#Each chromosome is rotated by an angle value, for eg 36 degree yielding 10x10 pairs of images then for each pair one chromosome is
#translated horizontally and vertically yielding more examples of possible chromosome overlapps. Finally, non overlapping chromosomes are discarded.
#The more rotation angle is small and the more le translation number is large, 
#the more the number of overlapping chromosomes is large et potentially overcome the RAM capacity.
#


def make_rotated_pairs(im1,im2, step):
    
    angles = [theta for theta in range(0,360, step)]
    rot1 = [nd.rotate(im1, angle) for angle in angles]# set of rotated chromosomes from one chrom
    rot2 = [nd.rotate(im2, angle) for angle in angles]# set of rotated chromosomes from the other chrom
    rotated_couples = itertools.product(rot1, rot2)
    return [p for p in rotated_couples]# not good for the RAM...

### try to add inplace a component 
def add_mask_to_list_of_tuples_of_images(list_of_pairs, threshold, mask1 = 1, mask2 = 2):
    masked_pairs = []
    for a_pair in list_of_pairs:
        first = add_mask_to_image(a_pair[0], threshold, mask_value = mask1)
        second = add_mask_to_image(a_pair[1], threshold, mask_value = mask2)
        masked_pairs.append((first,second))
    return masked_pairs


def filter_only_touching_chromosomes(candidates, touch_label = 3):
    overlapps = []
    for ovlp in candidates:
        if np.any(ovlp[:,:,1][:,:]==touch_label):
            overlapps.append(ovlp)
    return overlapps


def clip_img_to_bounding_box(img):
    ''' clipthe image fromthe mask (last image of the third axis)
    '''
    bb = nd.find_objects(img[:,:,-1]>0)
    slice0 = bb[0][0]
    slice1= bb[0][1]
    clip = img[slice0,slice1]
    return clip


def rolling_merge(moving_img, static_img, number_of_translations = 10, pix_border= 6,reclip = False):
    '''input :two images with a mask
    output: list of images where thr moving img is translated
    Horizontally and vertically relatively to the still img.
        '''
    
    row1, col1, _ = moving_img.shape
    row2, col2, _ = static_img.shape
    border = pix_border #border of pixels to remove, to get some chr intersections
    row = 2*row1 + row2 - border
    col = 2*col1 + col2 - border

    target1 = np.zeros((row, col,2), dtype = int)#+1
    target2 = np.zeros((row, col,2), dtype = int)#+1

    target1[0:row1, 0:col1, :] = moving_img
    target2[row1 - border/2:row1 + row2 - border/2, col1 - border/2:col1 + col2 - border/2, :] = static_img
    
    #generate the translation vectors components: first horizontally, then vertically.
    max_row_trans = row1 + row2 - border
    max_col_trans = col1 + col2 - border
    #generate intermediate translations vectors
    row_T = [int((1.0*t/number_of_translations)*max_row_trans) for t in range(0, number_of_translations + 1)]
    col_T = [int((1.0*t/number_of_translations)*max_col_trans) for t in range(0, number_of_translations + 1)]
    U = [t for t in itertools.product(row_T,col_T)]
    #generate images translations
    #translated_images = [translate(moving_img, t) for t in U]
    if reclip == False:
        return [target2 + translate(target1, t) for t in U]    
    elif reclip == True:
        return [clip_img_to_bounding_box(target2 + translate(target1, t)) for t in U]


def rolling_merge_h5py(moving_img, static_img, hdf5container, suffixname = 'array', 
                       number_of_translations = 10, 
                       pix_border= 6,
                       reclip = False,
                      touch_label = 3):
    '''input :two images with a mask
    output: list of images stored in an hdf5 container,where thr moving img is translated
    Horizontally and vertically relatively to the still img.
        '''
    
    row1, col1, _ = moving_img.shape
    row2, col2, _ = static_img.shape
    border = pix_border #border of pixels to remove, to get some chr intersections
    row = 2*row1 + row2 - border
    col = 2*col1 + col2 - border

    target1 = np.zeros((row, col,2), dtype = int)#+1
    target2 = np.zeros((row, col,2), dtype = int)#+1

    target1[0:row1, 0:col1, :] = moving_img
    target2[row1 - border/2:row1 + row2 - border/2, col1 - border/2:col1 + col2 - border/2, :] = static_img
    
    #generate the translation vectors components: first horizontally, then vertically.
    max_row_trans = row1 + row2 - border
    max_col_trans = col1 + col2 - border
    #generate intermediate translations vectors
    row_T = [int((1.0*t/number_of_translations)*max_row_trans) for t in range(0, number_of_translations + 1)]
    col_T = [int((1.0*t/number_of_translations)*max_col_trans) for t in range(0, number_of_translations + 1)]
    U = [t for t in itertools.product(row_T,col_T)]
    #generate images translations
    #translated_images = [translate(moving_img, t) for t in U]
    
    if reclip == False:
        #return [target2 + translate(target1, t) for t in U]
        for i,t in enumerate(U):
            image = target2 + translate(target1, t)
            
            ##just put image with overlapp
            if np.any(image[:,:,1][:,:] == touch_label):
                ## try to add the image in an hdf5 container,Why : hope to fix the RAM issue
                print '----------------- add:', suffixname+'_'+str(i).zfill(5)
                hdf5container.create_dataset(suffixname+'_'+str(i).zfill(5), 
                                             data = image, 
                                             compression="gzip", compression_opts=9 )
                
    if reclip == True:
        #return [target2 + translate(target1, t) for t in U]
        for i,t in enumerate(U):
            image = clip_img_to_bounding_box(target2 + translate(target1, t))
            
            ##just put image with overlapp
            if np.any(image[:,:,1][:,:] == touch_label):
                ## try to add the image in an hdf5 container,Why : hope to fix the RAM issue
                #print '----------------- add:', suffixname+'_'+str(i).zfill(5)
                hdf5container.create_dataset(suffixname+'_'+str(i).zfill(5), 
                                             data = image, 
                                             compression="gzip", compression_opts=9 )

                
                
                
def generate_overlapping_from_chrom_pairs(list_of_pairs, threshold = 2, rotationStep = 45, Ntranslation = 3):
    all_pairs_rotated_translated_merged = []
    #print len(list_of_pairs),' pairs of chrom'
    
    for p, onepair in enumerate(list_of_pairs):# take a/two pair(s) of chrom
        #print p,' processing pair id:',id(onepair)
        img_chrom1 = onepair[0]
        img_chrom2 = onepair[1]
        rotated_pairs_from_one_pair = make_rotated_pairs(img_chrom1, img_chrom2, rotationStep)
        threshold = 2
        masked_rotated_pairs_from_pair = add_mask_to_list_of_tuples_of_images(rotated_pairs_from_one_pair,threshold)
        
        rotated_translated_merged = []
        for alpha, one_orientation in enumerate(masked_rotated_pairs_from_pair):
            #print ' -', alpha, '       rotated pair ', id(one_orientation)
            still_img = one_orientation[0]
            translated_img = one_orientation[1]
            merged = filter_only_touching_chromosomes(rolling_merge(translated_img, 
                                                                    still_img,
                                                                    number_of_translations = Ntranslation,reclip = True))
            rotated_translated_merged = rotated_translated_merged + merged
            #print '               ', len(rotated_translated_merged),' images'
        all_pairs_rotated_translated_merged = all_pairs_rotated_translated_merged + rotated_translated_merged
        print p,' number of merged:', len(all_pairs_rotated_translated_merged)
    return(all_pairs_rotated_translated_merged)



def generate_overlapping_from_chrom_pairs_store_in_h5(list_of_pairs, 
                                             Container_name = "./Container.h5",
                                             threshold = 2, 
                                             rotationStep = 45, 
                                             Ntranslation = 3):
    
    all_pairs_rotated_translated_merged = []
    #print len(list_of_pairs),' pairs of chrom'
    ##
    ## Create an empty hdf5 container
    ##
    print "create a hdf5 container ..", Container_name
    container = h5py.File(Container_name, "w")
    
    for p, onepair in enumerate(list_of_pairs):# take a/two pair(s) of chrom
        #print p,' processing pair id:',id(onepair)
        img_chrom1 = onepair[0]
        img_chrom2 = onepair[1]
        rotated_pairs_from_one_pair = make_rotated_pairs(img_chrom1, img_chrom2, rotationStep)
        threshold = 2
        masked_rotated_pairs_from_pair = add_mask_to_list_of_tuples_of_images(rotated_pairs_from_one_pair,threshold)
        
        rotated_translated_merged = []
        for alpha, one_orientation in enumerate(masked_rotated_pairs_from_pair):
            #print ' -', alpha, '       rotated pair ', id(one_orientation)
            still_img = one_orientation[0]
            translated_img = one_orientation[1]
            rolling_merge_h5py(translated_img, 
                               still_img, container,
                               number_of_translations = Ntranslation,
                               reclip = True)
    container.close()
    

