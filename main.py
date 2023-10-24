#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

import os
import numpy as np

from global_defs import CONFIG
from prepare_data import Cityscapes, Pascal_voc
from utils import vis_pred_i, compute_miou 

np.random.seed(0)


def main():

    """
    Load dataset
    """
    print('load dataset')

    if CONFIG.DATASET == 'cityscapes':
        loader = Cityscapes( )
    elif CONFIG.DATASET == 'pascal_voc':
        loader = Pascal_voc( )
        
    print('dataset:', CONFIG.DATASET)
    print('number of images: ', len(loader))
    print('semantic segmentation network:', CONFIG.MODEL_NAME)
    print('attack:', CONFIG.ATTACK)
    print(' ')


    """
    For visualizing the (attacked) input data and predictions.
    """
    if CONFIG.PLOT_ATTACK:
        print("visualize (attacked) input data and predictions")

        if not os.path.exists( CONFIG.VIS_PRED_DIR ):
            os.makedirs( CONFIG.VIS_PRED_DIR )
        
        for i in range(len(loader)):
            vis_pred_i(loader[i])
    

    """
    Computation of mean IoU of ordinary and adversarial prediction.
    """
    if CONFIG.COMP_MIOU:
        print('compute mIoU')
        compute_miou(loader)
        compute_miou(loader, adv=True)
    

if __name__ == '__main__':
  
    print( "===== START =====" )
    main()
    print( "===== DONE! =====" )