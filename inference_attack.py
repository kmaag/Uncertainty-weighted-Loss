from argparse import ArgumentParser

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.apis.inference import LoadImage
from attacks import Attacks


trans = transforms.Compose([transforms.ToTensor()])

class Cityscapes():  

    def __init__(self, data_path, data_pipeline, split='val'):
        """
        Dataset loader for Cityscapes
        """
        test_pipeline = [LoadImage()] + data_pipeline
        self.test_pipeline = Compose(test_pipeline)

        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.filename = []  # image filename

        for city in sorted(os.listdir(os.path.join(data_path, 'leftImg8bit', split))):
            for img in sorted(os.listdir(os.path.join(data_path, 'leftImg8bit', split, city))):
                self.images.append(os.path.join(data_path, 'leftImg8bit', split, city, img)) 
                self.targets.append(os.path.join(data_path, 'gtFine', split, city, img.replace('leftImg8bit', 'gtFine_labelTrainIds'))) 
                self.filename.append(img.split('_leftImg8bit')[0])

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = self.test_pipeline(dict(img=self.images[index]))
        image = image['img'][0].unsqueeze(0)
        image = F.interpolate(image, size=(512,1024), mode='bilinear', align_corners=True)
        target = Image.open(self.targets[index])
        target = trans(target)*255
        target[target==255] = -1
        target = F.interpolate(target.unsqueeze(0).float(), size=(512,1024), mode='nearest').squeeze(0).long()
        return image, target, self.filename[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


class Pascal_voc():  

    def __init__(self, data_path, data_pipeline, split='val'):
        """
        Dataset loader for VOC
        """

        test_pipeline = [LoadImage()] + data_pipeline
        self.test_pipeline = Compose(test_pipeline)
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.filename = []  # image filename

        with open(os.path.join(os.path.join(data_path, 'VOC2012', 'ImageSets/Segmentation', split+'.txt')), "r") as lines:
            for line in lines:
                self.images.append( os.path.join(data_path, 'VOC2012', 'JPEGImages', line.rstrip('\n') + ".jpg") )
                self.targets.append( os.path.join(data_path, 'VOC2012', 'SegmentationClass', line.rstrip('\n') + ".png") )
                self.filename.append(line.rstrip('\n'))

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = self.test_pipeline(dict(img=self.images[index]))
        image = image['img'][0].unsqueeze(0)
        image = F.interpolate(image, size=(473,473), mode='bilinear', align_corners=True)
        target = Image.open(self.targets[index])
        target = trans(target)*255
        target[target==255] = -1
        target = F.interpolate(target.unsqueeze(0).float(), size=(473,473), mode='nearest').squeeze(0).long()
        return image, target, self.filename[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


def main():
    parser = ArgumentParser()
    parser.add_argument('--img-path', help='Image path')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-path', default=None, help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--dataset', default='cityscapes', help='Dataset') # pascal_voc_2012
    parser.add_argument('--attack', default='FGSM_untargeted', help='Type of attack')
    args = parser.parse_args()

    #### choose and define ####
    flag_save_raw_probs = False
    eps_value = 4

    save_path_probs = os.path.join(args.out_path, args.dataset, args.config.split('/')[1], 'probs')
    if not os.path.exists( save_path_probs ):
        os.makedirs( save_path_probs )
    if 'FGSM' in args.attack:
        save_path = os.path.join(args.out_path, args.dataset, args.config.split('/')[1], args.attack+str(eps_value), 'probs')
    else:
        save_path = os.path.join(args.out_path, args.dataset, args.config.split('/')[1], args.attack, 'probs')
    if not os.path.exists( save_path ):
        os.makedirs( save_path )
    
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # load dataset
    if args.dataset == 'cityscapes':
        loader = Cityscapes(args.img_path, model.cfg.data.test.pipeline[1:])
        train_loader = Cityscapes(args.img_path, model.cfg.data.test.pipeline[1:], 'train')
    elif args.dataset == 'pascal_voc_2012':
        loader = Pascal_voc(args.img_path, model.cfg.data.test.pipeline[1:])
        train_loader = Pascal_voc(args.img_path, model.cfg.data.test.pipeline[1:], 'train')
    print('number of images: ', len(loader))

    # initialize attack
    if args.dataset == 'cityscapes':
        params = {'ori_shape': (512, 1024, 3), 'img_shape': (1024, 2048, 3), 'flip': False}
    elif args.dataset == 'pascal_voc_2012':
        params = {'ori_shape': (473, 473, 3), 'img_shape': (480, 520, 3), 'flip': False}
    attack = Attacks(model, args.device, params)

    if args.attack == 'smm_static' or args.attack == 'smm_dynamic':
        save_path_noise = os.path.join(args.out_path, args.dataset, args.config.split('/')[1], args.attack)
        if not os.path.exists(save_path_noise):
            os.makedirs(save_path_noise)
        if not os.path.isfile(save_path_noise + '/uni_adv_noise.pt'):
            if args.attack == 'smm_static':
                noise = attack.universal_adv_pert_static(train_loader)
            elif args.attack == 'smm_dynamic':
                noise = attack.universal_adv_pert_dynamic(train_loader)
            torch.save(noise, save_path_noise + '/uni_adv_noise.pt')
        else:
            noise = torch.load(save_path_noise + '/uni_adv_noise.pt')
    
    for item,i in zip(loader,range(len(loader))):
        print(item[2])

        img_tens = item[0].to(args.device)
        img_tens.requires_grad = True
        mask_tens = item[1].to(args.device)

        if flag_save_raw_probs:
            with torch.no_grad():
                conv_output = model.whole_inference(img_tens, img_meta=[params], rescale=True)
            conv_output = torch.softmax(conv_output[0],0).cpu().detach().numpy()
            np.save(os.path.join(save_path_probs, item[2]+'.npy'), conv_output.astype('float16'))

        # run attack
        if args.attack == 'FGSM_untargeted':
            adv_img_tens, noise = attack.FGSM_untargeted(img_tens, mask_tens, eps=eps_value)
        elif args.attack == 'FGSM_untargeted_iterative':
            adv_img_tens, noise = attack.FGSM_untargeted_iterative(img_tens, mask_tens, eps=eps_value) 
        elif args.attack == 'unce_FGSM_untargeted':
            adv_img_tens, noise = attack.unc_FGSM_untar(img_tens, mask_tens, eps=eps_value, metric='e')
        elif args.attack == 'uncm_FGSM_untargeted':
            adv_img_tens, noise = attack.unc_FGSM_untar(img_tens, mask_tens, eps=eps_value, metric='m')
        elif args.attack == 'uncd_FGSM_untargeted':
            adv_img_tens, noise = attack.unc_FGSM_untar(img_tens, mask_tens, eps=eps_value, metric='d')
        elif args.attack == 'uncmbar_FGSM_untargeted':
            adv_img_tens, noise = attack.unc_FGSM_untar(img_tens, mask_tens, eps=eps_value, metric='mbar')
        elif args.attack == 'uncz_FGSM_untargeted':
            adv_img_tens, noise = attack.unc_FGSM_untar(img_tens, mask_tens, eps=eps_value, metric='z')
        elif args.attack == 'unce_FGSM_untargeted_iterative':
            adv_img_tens, noise = attack.unc_FGSM_untar_it(img_tens, mask_tens, eps=eps_value, metric='e')
        elif args.attack == 'uncm_FGSM_untargeted_iterative':
            adv_img_tens, noise = attack.unc_FGSM_untar_it(img_tens, mask_tens, eps=eps_value, metric='m')
        elif args.attack == 'uncd_FGSM_untargeted_iterative':
            adv_img_tens, noise = attack.unc_FGSM_untar_it(img_tens, mask_tens, eps=eps_value, metric='d')
        elif args.attack == 'uncmbar_FGSM_untargeted_iterative':
            adv_img_tens, noise = attack.unc_FGSM_untar_it(img_tens, mask_tens, eps=eps_value, metric='mbar')
        elif args.attack == 'uncz_FGSM_untargeted_iterative':
            adv_img_tens, noise = attack.unc_FGSM_untar_it(img_tens, mask_tens, eps=eps_value, metric='z')

        with torch.no_grad():
            adv_output = model.whole_inference(adv_img_tens, img_meta=[params], rescale=True)
        adv_output = torch.softmax(adv_output[0],0).cpu().detach().numpy()
        np.save(os.path.join(save_path, item[2]+'.npy'), adv_output.astype('float16'))



if __name__ == '__main__':
    main()