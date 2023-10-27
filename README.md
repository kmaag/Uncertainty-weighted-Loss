## Uncertainty-weighted Loss Functions for Improved Adversarial Attacks on Semantic Segmentation

For a detailed description, please refer to https://arxiv.org/abs/2310.17436 .

### Packages and their versions:
Code is tested with ```Python 3.9.12``` and ```pip 22.3.1```.
Install Python packages via
```sh
pip install -r requirements.txt
```

### Adversarial example generation:
As modelzoo we consider the mmsegmentation framework https://github.com/open-mmlab/mmsegmentation. 

For the FGSM attacks, e.g. the I-FGSM with uncertainty-weighted loss $L^E$ for the DeepLabv3+ network trained on the VOC dataset, run
```python
python inference_attack.py \
    --img-path /home/user/datasets/ \
    --config configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py \
    --checkpoint checkpoints/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth \
    --out-path /home/user/output/ \
    --device cuda:0 \
    --dataset pascal_voc_2012 \
    --attack unce_FGSM_untargeted_iterative
```
with implementations based on the framework https://github.com/kmaag/Adversarial-Attack-Detection-Uncertainty. 

For the PGD and the ALMA prox attack, we use the framework https://github.com/jeromerony/alma_prox_segmentation which also employs the mmsegmentation modelzoo as basis. To run the attacks with our uncertainty-based loss function, insert the loss function `unc_loss_U()` from `uncertainty_weighting.py` into the file `attacks/pgd.py` of the original framework for PGD and `adv_lib/attacks/segmentation/alma_prox.py` for ALMA prox.

For the CR attack, we use the available framework https://github.com/randomizedheap/CR_Attack and replace their CR-based loss by our `unc_loss_U()` function in file `whitebox_lib.py`.

For the patch attack, we consider the framework https://github.com/retis-ai/SemSegAdvPatch/ and replace the function `optimize_patch()` in `untargeted_patch_attack.py` by the function `unc_attack_pix()` provided in `uncertainty_weighting.py`.

Note, for all modified files the license from the original framework is valid.

### Evaluation:
Edit all necessary paths stored in `global_defs.py`. The outputs will be saved in the chosen `io_path`. 
By setting the corresponding boolean variable (True/False) you can select tasks to be executed. Run
```python
python main.py
```

## Author:
Kira Maag, maag@tu-berlin.de


