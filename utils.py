#!/usr/bin/env python3
'''
script including utility functions
'''

import os
import numpy as np 
from PIL import Image
from scipy.stats import entropy
import matplotlib
import matplotlib.pyplot as plt

from global_defs import CONFIG
from prepare_data import cs_labels, voc_labels

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def vis_pred_i(item):

    if not os.path.isfile(item[4]):
        return 0 

    if CONFIG.DATASET == 'cityscapes':
        labels = cs_labels
        classes = np.concatenate( (np.arange(0,19),[255]), axis=0)
    elif CONFIG.DATASET == 'pascal_voc_2012':
        labels = voc_labels
        classes = np.concatenate( (np.arange(0,21),[255]), axis=0)
    trainId2label = { label.trainId : label for label in reversed(labels) }

    image = item[0]
    label = item[1]
    probs = np.load(item[3])
    seg = np.argmax(probs, axis=0)
    probs_adv = np.load(item[4])
    seg_adv = np.argmax(probs_adv, axis=0)

    I1 = image.copy()
    I2 = image.copy()
    I3 = image.copy()
    I4 = image.copy()

    for c in classes:
        I2[label==c,:] = np.asarray(trainId2label[c].color)
        I3[seg==c,:] = np.asarray(trainId2label[c].color)
        I4[seg_adv==c,:] = np.asarray(trainId2label[c].color)
    
    plt.imsave(CONFIG.VIS_PRED_DIR + item[2] + '_tmp1.png', entropy(probs,axis=0), cmap='inferno')
    I5 = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[2] + '_tmp1.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[2] + '_tmp1.png')

    plt.imsave(CONFIG.VIS_PRED_DIR + item[2] + '_tmp2.png', entropy(probs_adv,axis=0), cmap='inferno')
    I6 = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[2] + '_tmp2.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[2] + '_tmp2.png')

    img12   = np.concatenate( (I2,I1), axis=0 )
    img34  = np.concatenate( (I3,I4), axis=1 )
    img56  = np.concatenate( (I5,I6), axis=1 )
    img3456   = np.concatenate( (img34,img56), axis=0 )
    img   = np.concatenate( (img12,img3456), axis=1 )

    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img.save(CONFIG.VIS_PRED_DIR + item[2] + '.png')
    plt.close()
    print('stored:', item[2]+'.png')


def compute_miou(loader, adv=False):

    if adv:
        save_path = os.path.join(CONFIG.COMP_MIOU_DIR,'miou_acc_apsr.npy')
    else:
        save_path = os.path.join(CONFIG.COMP_MIOU_DIR.replace(CONFIG.ATTACK+'/',''),'miou_acc_apsr.npy')
    
    if not os.path.exists( os.path.dirname(save_path) ):
        os.makedirs( os.path.dirname(save_path) )

    if not os.path.isfile(save_path):

        if CONFIG.DATASET == 'cityscapes':
            num_classes = 19
            class_id = 11
            labels = cs_labels
        elif CONFIG.DATASET == 'pascal_voc_2012':
            num_classes = 21
            class_id = 15
            labels = voc_labels
        trainId2label = { label.trainId : label for label in reversed(labels) }

        seg_all = []
        gt_all = []

        counter = 0
        for item in loader:
            # if CONFIG.ATTACK != 'smm_dynamic' or class_id in np.unique(item[1]):
            if os.path.isfile(item[4]):
                print(item[2])

                gt = item[1]
                if adv:
                    seg = np.argmax(np.load(item[4]), axis=0)
                else:
                    seg = np.argmax(np.load(item[3]), axis=0)
                seg[gt==255] = 255

                seg_all.append(seg)
                gt_all.append(gt)

                counter += 1
        
        print('num images miou:', counter)
        
        seg_all = np.stack(seg_all, 0)
        gt_all = np.stack(gt_all, 0)

        # intersection, union
        seg_iu = np.zeros((num_classes,2))
        num_pix = 0

        for c in range(num_classes):
            seg_iu[c,0] = np.sum(np.logical_and(seg_all==c,gt_all==c))
            seg_iu[c,1] = np.sum(np.logical_or(seg_all==c,gt_all==c))
        num_pix = np.sum(gt_all != 255)

        result_path = save_path.replace('npy','txt')
        with open(result_path, 'a') as fi:
            print('(adversarial) prediction ',  adv, ':', file=fi)
            counter_c = 0
            iou_all = 0
            for c in range(num_classes):
                if seg_iu[c,1] > 0:
                    counter_c += 1
                    iou_c = seg_iu[c,0] / seg_iu[c,1]
                    iou_all += iou_c
                    print('IoU of class', trainId2label[c].name, ':', iou_c, file=fi)
            print('mIoU:', iou_all / counter_c, file=fi)
            print('accuracy:', np.sum(seg_iu[:,0]) / num_pix, file=fi)
            print('APSR:', (num_pix-np.sum(seg_iu[:,0])) / num_pix, file=fi)
            print(' ', file=fi)
        
        metrics = np.zeros((3))
        metrics[0] = iou_all / counter_c
        metrics[1] = np.sum(seg_iu[:,0]) / num_pix
        metrics[2] = (num_pix-np.sum(seg_iu[:,0])) / num_pix
        np.save(save_path, metrics)

