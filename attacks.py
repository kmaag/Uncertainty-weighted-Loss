import torch
import torch.nn as nn
from tqdm import trange
from torch.autograd import Variable


class Attacks():
    """
    Attacks class contains different attacks 
    """
    def __init__(self, model, device='cpu', params=[]): 
        """ Initialize the FGSM class
        Args:
            model (torch.nn model): model to be attacked
            device  (device):       device
        """
        self.model = model
        self.device = device
        self.params = params

    def model_pred(self, img):
        """ individual model prediction
        Args:
            img (torch.tensor): input image
        Returns:
           pred (torch.tensor): predicted semantic segmentation
        """
        pred = self.model.whole_inference(img, img_meta=[self.params], rescale=True)
        return pred


    def FGSM_untargeted(self, img, label, eps=2):
        """ FGSM untargeted attack (FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label of the input image
            eps  (float):          size of adversarial perturbation
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        eps = eps / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.model.zero_grad()
        pred = self.model_pred(img)

        lo = loss(pred, label.detach())
        lo.backward()
        im_grad = img.grad

        noise = eps * torch.sign(im_grad)
        adv_img = img + noise
        return adv_img, noise 


    def FGSM_untargeted_iterative(self, img, label, alpha=1, eps=2, num_it=None):
        """ FGSM iterative untargeted (I-FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label 
            alpha  (float):        step size of the attack
            eps    (float):        size of adversarial perturbation
            num_it (int):          number of attack iterations 
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        if num_it == None:
            num_it = min(int(eps+4), int(1.25*eps))
        eps = eps / 255
        alpha = alpha / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        adv_img = img 
        adv_img.requires_grad = True
        
        tbar=trange(num_it)
        for i in tbar:

            self.model.zero_grad()
            pred = self.model_pred(adv_img)

            lo = loss(pred, label.detach())
            lo.backward()
            im_grad = adv_img.grad

            noise = (alpha * torch.sign(im_grad)).clamp(-eps,eps)
            adv_img = (adv_img + noise).clamp(img-eps,img+eps)
            adv_img = Variable(adv_img, requires_grad=True)

            tbar.set_description('Iteration: {}/{} of I-FGSM attack'.format(i, num_it))
        return adv_img, noise 


    def unc_loss_E( probs, dim=0 ):
        output = torch.sum(-probs * torch.log(probs+1e-10), dim=dim)
        output = torch.div(output.clone(), torch.log(torch.tensor(probs.shape[dim])))
        return output
        
    def unc_loss_M( probs, dim=0 ):
        largest = torch.topk(probs, 2, dim=0).values
        output = 1 - largest[0,:,:] + largest[1,:,:]
        return output
    
    def unc_loss_D( probs, dim=0 ):
        sorted, _ = torch.sort(probs, dim=0, descending=True)
        output = 1 - sorted[0,:,:] + sorted[-1,:,:]
        return output

    def unc_loss_Mbar( probs, dim=0 ):
        sorted, _ = torch.sort(probs, dim=0, descending=True)
        output = 1 - torch.sum(sorted[0]-sorted,0)/(sorted.shape[0]-1)
        return output
    

    def unc_FGSM_untar(self, img, label, eps=2, metric='e'):
        """ FGSM untargeted attack (FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label of the input image
            eps  (float):          size of adversarial perturbation
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        eps = eps / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
        _, _, H, W = img.size()

        self.model.zero_grad()
        pred = self.model_pred(img)

        J_cls = loss(pred, label.detach()) 

        if metric == 'e':
            unc_metric = torch.exp( Attacks.unc_loss_E(torch.softmax(pred[0],0)) )
        elif metric == 'm':
            unc_metric = torch.exp( Attacks.unc_loss_M(torch.softmax(pred[0],0)) )
        elif metric == 'd':
            unc_metric = torch.exp( Attacks.unc_loss_D(torch.softmax(pred[0],0)) )
        elif metric == 'mbar':
            unc_metric = torch.exp( Attacks.unc_loss_Mbar(torch.softmax(pred[0],0)) )
        elif metric == 'z':
            J_cls[0,torch.logical_and(torch.argmax(pred[0],0)!=label[0], torch.max(pred[0],0)[0]>0.75)] = 0

        J_ss = torch.sum( unc_metric * J_cls[0] ) / (H * W)
        J_ss.backward()

        im_grad = img.grad
        noise = eps * torch.sign(im_grad)
        adv_img = img + noise
        return adv_img, noise 


    def unc_FGSM_untar_it(self, img, label, alpha=1, eps=2, num_it=None, metric='e'):
        """ FGSM iterative untargeted (I-FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label 
            alpha  (float):        step size of the attack
            eps    (float):        size of adversarial perturbation
            num_it (int):          number of attack iterations 
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        if num_it == None:
            num_it = min(int(eps+4), int(1.25*eps))
        eps = eps / 255
        alpha = alpha / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
        _, _, H, W = img.size()

        adv_img = img 
        adv_img.requires_grad = True
        
        tbar=trange(num_it)
        for i in tbar:

            self.model.zero_grad()
            pred = self.model_pred(adv_img)

            J_cls = loss(pred, label.detach()) 

            if metric == 'e':
                unc_metric = torch.exp( Attacks.unc_loss_E(torch.softmax(pred[0],0)) )
            elif metric == 'm':
                unc_metric = torch.exp( Attacks.unc_loss_M(torch.softmax(pred[0],0)) )
            elif metric == 'd':
                unc_metric = torch.exp( Attacks.unc_loss_D(torch.softmax(pred[0],0)) )
            elif metric == 'mbar':
                unc_metric = torch.exp( Attacks.unc_loss_Mbar(torch.softmax(pred[0],0)) )
            elif metric == 'z':
                J_cls[0,torch.logical_and(torch.argmax(pred[0],0)!=label[0], torch.max(pred[0],0)[0]>0.75)] = 0

            J_ss = torch.sum( unc_metric * J_cls[0] ) / (H * W)
            J_ss.backward()
            
            im_grad = adv_img.grad
            noise = (alpha * torch.sign(im_grad)).clamp(-eps,eps)
            adv_img = (adv_img + noise).clamp(img-eps,img+eps)
            adv_img = Variable(adv_img, requires_grad=True)

            tbar.set_description('Iteration: {}/{} of I-FGSM attack'.format(i, num_it))
        return adv_img, noise 
    

