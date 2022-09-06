import torch
import torch.nn as nn
import pdb


def image_loss(model_out, gt, mask=None):
    gt_rgb = gt['rgb']
    # pdb.set_trace()
    ## transpose first
    gt = torch.transpose(torch.transpose(gt['rgb'],2,3),3,4)
    b,x,w,h,c = gt.shape
    gt = gt.reshape(b,x,w*h,c)
    return nn.MSELoss()(gt, model_out['rgb']) * 200 ## why multiply by 200?


class LFLoss():
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_dict['img_loss'] = image_loss(model_out, gt)
        loss_dict['reg'] = (model_out['z']**2).mean() * self.reg_weight ## this loss most likely does not work for my autoencoder..
        return loss_dict, {}


