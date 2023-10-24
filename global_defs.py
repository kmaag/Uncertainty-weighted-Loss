#!/usr/bin/env python3
'''
script including
class object with global settings
'''

class CONFIG:
  
    #---------------------#
    # set necessary paths #
    #---------------------#
  
    io_path   = '/home/user/outputs/'   # directory with inputs and outputs, i.e. saving and loading data

    #------------------#
    # select or define #
    #------------------#
  
    datasets = ['cityscapes', 'pascal_voc_2012']
    DATASET = datasets[0]

    model_names = ['bisenetv1','deeplabv3plus','pspnet','bisenetX39','ddrnet23Slim']
    MODEL_NAME = model_names[1]
    
    FGSM_eps = 8 
    FLAG_unc = True
    attacks = ['FGSM_untargeted'+str(FGSM_eps),
               'FGSM_untargeted_iterative'+str(FGSM_eps),
               'minimal_pgd',
               'alma_prox',
               'cr_pgd_l2',
               'cr_pgd_linf',
               'patch_eot']
    ATTACK = attacks[0]
    if FLAG_unc:
        ATTACK = 'unce_'+ATTACK
        if 'patch_eot' in ATTACK:
            ATTACK = ATTACK+str(FGSM_eps)
  
    #----------------------------#
    # paths for data preparation #
    #----------------------------#
    
    DATA_DIR   = io_path + '/data/'
    PROBS_DIR  = io_path + '/' + DATASET + '/' + MODEL_NAME + '/probs/'
    PROBSA_DIR = io_path + '/' + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/probs/'  

    #--------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    #--------------------------------------------------------------------#

    PLOT_ATTACK    = False
    COMP_MIOU      = False
    
    #-----------#
    # optionals #
    #-----------#
    
    SAVE_OUT_DIR       = io_path + DATASET + '/' + MODEL_NAME + '/' 
    VIS_PRED_DIR       = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/vis_pred/'
    COMP_MIOU_DIR      = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/miou/'
    
    