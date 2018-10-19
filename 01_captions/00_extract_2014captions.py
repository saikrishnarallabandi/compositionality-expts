import json
import os
from utils import *
import re

# This script uses Python2 since COCO API depends on Python2 
# This script follows the ideology behind https://github.com/fregu856/CS224n_project/blob/master/preprocess_captions.py

train_annotations_file = '../repos/imgcap/data/mscoco/captions_train2014.json'
val_annotations_file = '../repos/imgcap/data/mscoco/captions_val2014.json'

import sys
sys.path.append("/home2/srallaba/courses/11777/project/repos/coco-caption/")
from pycocotools.coco import COCO

train_write_file = 'train2014.captions.txt'
twf = open(train_write_file,'w')
val_write_file = 'val2014.captions.txt'
vwf = open(val_write_file,'w')

caption_id_2_img_id = {}
val_caption_id_2_caption = {}
train_caption_id_2_caption = {}

def get_captions(type_of_data):
 
    if type_of_data == 'val':
       captions_file = val_annotations_file
    elif type_of_data == 'train':
       captions_file = train_annotations_file
    else:
       print "yay! You have succesfully broke me! You deserve a cookie dear!"   

    # initialize COCO api for captions:
    coco = COCO(captions_file)

    # get indices for all "type_of_data" images (all train or val imgs
    # (original data split on mscoco.org)):
    img_ids = coco.getImgIds()

    for step, img_id in enumerate(img_ids):
        if step % 1000 == 0:
            print step
            log(str(step))

        # get the ids of all captions for the image:
        caption_ids = coco.getAnnIds(imgIds=img_id)
        # get all caption objects for the image:
        caption_objs = coco.loadAnns(caption_ids)

        for caption_obj in caption_objs:
            # save the caption id and the corresponding img id:
            caption_id = caption_obj["id"]
            caption_id_2_img_id[caption_id] = img_id

            # get the caption:
            caption = caption_obj["caption"]
            # remove empty spaces in the start or end of the caption:
            caption = caption.strip()
            # make the caption lower case:
            caption = caption.lower()
            # remove all non-alphanum chars (keep spaces between word):
            caption = re.sub("[^a-z0-9 ]+", "", caption)
            # remove all double spaces in the caption:
            caption = re.sub("  ", " ", caption)
            # convert the caption into a vector of words:
            #caption = caption.split(" ")
            # remove any empty chars still left in the caption:
            #while "" in caption:
            #    index = caption.index("")
            #    del caption[index]

            # store the caption in the corresponding captions dict:
            if type_of_data == 'val':
                 val_caption_id_2_caption[caption_id] = caption
                 vwf.write(str(caption_id) + ' ' + caption + '\n')
            elif type_of_data == 'train':
                 train_caption_id_2_caption[caption_id] = caption
                 twf.write(str(caption_id) + ' ' + caption + '\n')
            else:
                 print "You are not satisfied with a cookie are you?" 
 


get_captions('val')
vwf.close()

get_captions('train')
vwf.close()

