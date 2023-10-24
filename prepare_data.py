#!/usr/bin/env python3
"""
script for metaseg input preparation
"""

import os
import numpy as np 
from PIL import Image
from collections import namedtuple

from global_defs import CONFIG


Label = namedtuple('Label',['name','Id','trainId','color'])

class Cityscapes():  

    def __init__(self, **kwargs):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.filename = []  # image filename
        self.preds = []     # where to load depth maps - absolute paths
        self.preds_adv = [] # where to load semantic segmentation predictions - absolute paths

        for city in sorted(os.listdir(os.path.join(CONFIG.DATA_DIR, 'cityscapes', 'leftImg8bit', 'val'))):
            for img in sorted(os.listdir(os.path.join(CONFIG.DATA_DIR, 'cityscapes', 'leftImg8bit', 'val', city))):

                self.images.append(os.path.join(CONFIG.DATA_DIR, 'cityscapes', 'leftImg8bit', 'val', city, img)) 
                self.targets.append(os.path.join(CONFIG.DATA_DIR, 'cityscapes', 'gtFine', 'val', city, img.replace('leftImg8bit','gtFine_labelTrainIds'))) 
                self.filename.append(img.split('_left')[0])
                if 'patch_eot' in CONFIG.ATTACK:
                    self.preds.append(os.path.join(CONFIG.PROBS_DIR.replace('Adversarials_attack/cityscapes','Adversarials/cityscape'), img.replace('.png','.npy'))) 
                else:
                    self.preds.append(os.path.join(CONFIG.PROBS_DIR, img.replace('_leftImg8bit.png','.npy'))) 
                if 'patch_eot' in CONFIG.ATTACK:
                    self.preds_adv.append(os.path.join(CONFIG.PROBSA_DIR.replace('Adversarials_attack/cityscapes','Adversarials/cityscape'), img.replace('png','npy'))) 
                else:
                    self.preds_adv.append(os.path.join(CONFIG.PROBSA_DIR, img.replace('_leftImg8bit.png','.npy'))) 


    def __getitem__(self, index):
        """Generate one sample of data"""
        image = Image.open(self.images[index]).convert('RGB')
        image = image.resize((1024,512), Image.BILINEAR)
        image = np.asarray(image)
        target = Image.open(self.targets[index])
        target = target.resize((1024,512), Image.NEAREST)
        target = np.asarray(target)
        return image, target, self.filename[index], self.preds[index], self.preds_adv[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


cs_labels = [
    #       name                       Id   trainId    color
    Label(  'unlabeled'            ,    0 ,     255 ,  (255,255,255) ),
    Label(  'ego vehicle'          ,    1 ,     255 ,  (  0,  0,  0) ),
    Label(  'rectification border' ,    2 ,     255 ,  (  0,  0,  0) ),
    Label(  'out of roi'           ,    3 ,     255 ,  (  0,  0,  0) ),
    Label(  'static'               ,    4 ,     255 ,  (  0,  0,  0) ),
    Label(  'dynamic'              ,    5 ,     255 ,  (111, 74,  0) ),
    Label(  'ground'               ,    6 ,     255 ,  ( 81,  0, 81) ),
    Label(  'road'                 ,    7 ,       0 ,  (128, 64,128) ),
    Label(  'sidewalk'             ,    8 ,       1 ,  (244, 35,232) ),
    Label(  'parking'              ,    9 ,     255 ,  (250,170,160) ),
    Label(  'rail track'           ,   10 ,     255 ,  (230,150,140) ),
    Label(  'building'             ,   11 ,       2 ,  ( 70, 70, 70) ),
    Label(  'wall'                 ,   12 ,       3 ,  (102,102,156) ),
    Label(  'fence'                ,   13 ,       4 ,  (190,153,153) ),
    Label(  'guard rail'           ,   14 ,     255 ,  (180,165,180) ),
    Label(  'bridge'               ,   15 ,     255 ,  (150,100,100) ),
    Label(  'tunnel'               ,   16 ,     255 ,  (150,120, 90) ),
    Label(  'pole'                 ,   17 ,       5 ,  (153,153,153) ),
    Label(  'polegroup'            ,   18 ,     255 ,  (153,153,153) ),
    Label(  'traffic light'        ,   19 ,       6 ,  (250,170, 30) ),
    Label(  'traffic sign'         ,   20 ,       7 ,  (220,220,  0) ),
    Label(  'vegetation'           ,   21 ,       8 ,  (107,142, 35) ),
    Label(  'terrain'              ,   22 ,       9 ,  (152,251,152) ),
    Label(  'sky'                  ,   23 ,      10 ,  ( 70,130,180) ),
    Label(  'person'               ,   24 ,      11 ,  (220, 20, 60) ),
    Label(  'rider'                ,   25 ,      12 ,  (255,  0,  0) ),
    Label(  'car'                  ,   26 ,      13 ,  (  0,  0,142) ),
    Label(  'truck'                ,   27 ,      14 ,  (  0,  0, 70) ),
    Label(  'bus'                  ,   28 ,      15 ,  (  0, 60,100) ),
    Label(  'caravan'              ,   29 ,     255 ,  (  0,  0, 90) ),
    Label(  'trailer'              ,   30 ,     255 ,  (  0,  0,110) ),
    Label(  'train'                ,   31 ,      16 ,  (  0, 80,100) ),
    Label(  'motorcycle'           ,   32 ,      17 ,  (  0,  0,230) ),
    Label(  'bicycle'              ,   33 ,      18 ,  (119, 11, 32) ),
    Label(  'license plate'        ,   -1 ,      -1 ,  (  0,  0,142) ),
]


class Pascal_voc():  

    def __init__(self, **kwargs):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.filename = []  # image filename
        self.preds = []     # where to load depth maps - absolute paths
        self.preds_adv = [] # where to load semantic segmentation predictions - absolute paths

        with open(os.path.join(os.path.join(CONFIG.DATA_DIR, 'VOCdevkit/VOC2012', 'ImageSets/Segmentation','val.txt')), "r") as lines:
            for line in lines:
                self.images.append( os.path.join(CONFIG.DATA_DIR, 'VOCdevkit/VOC2012', 'JPEGImages', line.rstrip('\n') + ".jpg") )
                self.targets.append( os.path.join(CONFIG.DATA_DIR, 'VOCdevkit/VOC2012', 'SegmentationClass', line.rstrip('\n') + ".png") )
                self.filename.append(line.rstrip('\n'))
                self.preds.append(os.path.join(CONFIG.PROBS_DIR, line.rstrip('\n') + ".npy")) 
                self.preds_adv.append(os.path.join(CONFIG.PROBSA_DIR, line.rstrip('\n') + ".npy")) 

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = Image.open(self.images[index]).convert('RGB')
        image = image.resize((473, 473), Image.BILINEAR)
        image = np.asarray(image)
        target = Image.open(self.targets[index])
        target = target.resize((473, 473), Image.NEAREST)
        target = np.asarray(target)
        return image, target, self.filename[index], self.preds[index], self.preds_adv[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


voc_labels = [
    #       name                  Id   trainId    color
    Label(  'background'      ,    0 ,       0 ,  (  0,  0,  0) ),
    Label(  'aeroplane'       ,    1 ,       1 ,  (128,  0,  0) ),
    Label(  'bicycle'         ,    2  ,      2 ,  (  0,128,  0) ),
    Label(  'bird'            ,    3 ,       3 ,  (128,128,  0) ),
    Label(  'boat'            ,    4 ,       4 ,  (  0,  0,128) ),
    Label(  'bottle'          ,    5 ,       5 ,  (128,  0,128) ),
    Label(  'bus'             ,    6 ,       6 ,  (  0,128,128) ),
    Label(  'car'             ,    7 ,       7 ,  (128,128,128) ),
    Label(  'cat'             ,    8 ,       8 ,  ( 64,  0,  0) ),
    Label(  'chair'           ,    9 ,       9 ,  (192,  0,  0) ),
    Label(  'cow'             ,   10 ,      10 ,  ( 64,128,  0) ),
    Label(  'diningtable'     ,   11 ,      11 ,  (192,128,  0) ),
    Label(  'dog'             ,   12 ,      12 ,  ( 64,  0,128) ),
    Label(  'horse'           ,   13 ,      13 ,  (192,  0,128) ),
    Label(  'motorbike'       ,   14 ,      14 ,  ( 64,128,128) ),
    Label(  'person'          ,   15 ,      15 ,  (192,128,128) ),
    Label(  'pottedplant'     ,   16 ,      16 ,  (  0, 64,  0) ),
    Label(  'sheep'           ,   17 ,      17 ,  (128, 64,  0) ),
    Label(  'sofa'            ,   18 ,      18 ,  (  0,192,  0) ),
    Label(  'train'           ,   19 ,      19 ,  (128,192,  0) ),
    Label(  'tv/monitor'      ,   20 ,      20 ,  (  0, 64,128) ),
    Label(  'void/unlabelled' ,   21 ,     255 ,  (224,224,192) ),
]

    
