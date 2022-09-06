import matplotlib
matplotlib.use('Agg')

import torch
import util
import torchvision
import wandb

import pdb



def img_summaries(model, model_input, ground_truth, loss_summaries, model_output, iter, prefix="", img_shape=None):
    predictions = model_output['rgb']
    trgt_imgs = ground_truth['rgb']
    indices = model_input['query']['instance_idx']

    predictions = util.flatten_first_two(predictions)
    trgt_imgs = util.flatten_first_two(trgt_imgs)


    with torch.no_grad():
        if 'context' in model_input and model_input['context']:
            context_images = model_input['context']['rgb'] * model_input['context']['mask'][..., None]
            context_images = util.lin2img(util.flatten_first_two(context_images), image_resolution=img_shape)
            context_images_wandb = wandb.Image(context_images, 'The context images')
            wandb.log({f'{prefix}context_images': context_images, 'total_steps':iter})
            
        b,c,w,h, = trgt_imgs.shape
        trgt_imgs = torch.transpose(torch.transpose(trgt_imgs, 1,2),2,3)
        output_vs_gt = torch.cat((predictions, trgt_imgs.reshape(b,w*h,c)), dim=0)
        output_vs_gt = util.lin2img(output_vs_gt, image_resolution=img_shape)
        output_vs_gt_wandb = wandb.Image(output_vs_gt, 'The output vs GT')
        wandb.log({f'{prefix}output_vs_gt': output_vs_gt_wandb, 'total_steps':iter})
        
        wandb.log({f'{prefix}out_min':predictions.min(), 'total_steps':iter})
        wandb.log({f'{prefix}out_max':predictions.max(), 'total_steps':iter})
        
        wandb.log({f'{prefix}trgt_min':trgt_imgs.min(), 'total_steps':iter})
        wandb.log({f'{prefix}trgt_max':trgt_imgs.max(), 'total_steps':iter})

        wandb.log({f'{prefix}idx_min':indices.max(), 'total_steps':iter})
        wandb.log({f'{prefix}idx_max':indices.max(), 'total_steps':iter})
