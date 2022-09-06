# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Manager
import torch
import models
import training
import summaries
import hdf5_dataio
import configargparse
from torch.utils.data import DataLoader
import loss_functions
import config

import wandb

## Please see the shift-equivariant-autoencoders repository for the modules containing the models below...
from auto_encoder import Encoder,BasicEncoderBlock
from auto_encoder.resnet_decoder import Decoder
from auto_encoder.resnet_aencoder import AutoEncoder, Bottleneck
from auto_encoder_aps.resnet_aencoder import AutoEncoderAPS

import pdb
#auth wandb
apikey= os.environ.get("WANDB_API_KEY")
os.system(f"python3 -m wandb.cli login {apikey}")


parser = configargparse.ArgumentParser()
parser.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

parser.add_argument('--logging_root', type=str, default=config.logging_root, required=False, help='root for logging')
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--csv_loc', type=str, default=None, help='location of the csv file')
parser.add_argument('--network', type=str, default='relu')
parser.add_argument('--architecture', type=str, required=True, help="name of the architecture")
parser.add_argument('--run_name', type=str, required=True, help="name of the run")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--conditioning', type=str, default='hyper')
parser.add_argument('--custom_aencoder', type=bool, required=False) # keep it a boolean for now
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--num_trgt', type=int, default=1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=40001)
parser.add_argument('--epochs_til_ckpt', type=int, default=10)
parser.add_argument('--steps_til_summary', type=int, default=1000)
parser.add_argument('--max_num_instances', type=int, default=None)
parser.add_argument('--max_observations_per_instance', type=int, default=None)
parser.add_argument('--iters_til_ckpt', type=int, default=10000)
parser.add_argument('--checkpoint_path', default=None)
parser.add_argument('--optim_checkpoint_path', default=None)

## WandB Params
parser.add_argument('--wandb_notes', type=str, dest="wandb_notes", required=True, help="The notes describing the run")
parser.add_argument('--wandb_tags', nargs="+", required=True, help="Tags for the run in WandB")

## AE Params
parser.add_argument('--aencoder_state_dict', type=str, required=False)
# specific APS model params...
# TODO: these params are currently hardcoded, fix it..
parser.add_argument('--in_channels', type=int, dest="in_channels", default=3, help="The number of input channels")
parser.add_argument('--out_channels', type=int, dest="out_channels", default=3, help="The number of output channels")
parser.add_argument('--inner_channels', type=str, dest="inner_channels", default='[64, 128, 256, 512, 1024]', help="The configuration of the inner channels, pass the input as list, e.g. \"[1,2,3]\" ")
parser.add_argument('--bilinear', type=bool, dest="bilinear", default=False, help="Specify whether the model should be bilinear..")
parser.add_argument('--padding_mode', type=str, dest="padding_mode", default='circular', help="The padding mode used in the sampling layers")
parser.add_argument('--filter_size', type=int, dest="filter_size", default=1, help="The filter size of the APS method")

opt = parser.parse_args()

batch_sizes = 64,64
sidelens =  128,128

wandb_config = {
    'learning_rate':opt.lr,
    'epochs': opt.num_epochs,
    'architecture':opt.architecture,
    'dataset': opt.dataset,
    'run_name': opt.run_name,
    'project':'YOUR WANDB PROJECT NAME'
}

wandb.init(
    project = "YOUR WANDB PROJECT NAME",
    entity = "YOUR WANBD ENTITY NAME", ## WANDB = Weights and Biases
    notes = opt.wandb_notes,
    tags = opt.wandb_tags,
    config = wandb_config
)

wandb.run.name = opt.run_name

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)


def multigpu_train(gpu, opt, cache):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    def create_dataloader_callback(sidelength, batch_size, query_sparsity):
        train_dataset = hdf5_dataio.SceneClassDataset(num_context=0, num_trgt=opt.num_trgt,
                                                      data_root=opt.data_root, csv_loc=opt.csv_loc, query_sparsity=query_sparsity,
                                                      img_sidelength=sidelength, vary_context_number=True, cache=cache,
                                                      max_num_instances=opt.max_num_instances,
                                                      max_observations_per_instance=opt.max_observations_per_instance)
        print(f"the dataset length {len(train_dataset)}")        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0)
        return train_loader

    num_instances = hdf5_dataio.get_num_instances(opt.data_root)
    if opt.csv_loc is not None:
        num_instances = hdf5_dataio.get_num_instances_csv(opt.csv_loc)
    model = models.LFAutoDecoder(latent_dim=256, num_instances=num_instances, parameterization='plucker',
                                 network=opt.network, conditioning=opt.conditioning).cuda()
    # encoder = Encoder(3,32, BasicEncoderBlock, [2,2,2,2]).cuda()
    # decoder = Decoder(32, 256).cuda()
    # bottleneck = Bottleneck(256,256).cuda()
    # ae = AutoEncoder(32, 3, 256, encoder, decoder,bottleneck).cuda()
    ae = AutoEncoderAPS(3, 32).cuda()

    if opt.custom_aencoder:
        ## the LFEncoder contains our autoencoder, which is pretrained
        ## we need to load the weights for this model
        ## remove decoder layers, and attach this to the actual LFN
        ## loading the checkpoint is best done in the model class itself..
        print('-----custom aencoder ------')
        model = models.LFEncoderCustom(latent_dim=256, num_instances=num_instances, parameterization='plucker',
                                        encoder=ae,state_dict=opt.aencoder_state_dict,
                                        network=opt.network, conditioning=opt.conditioning).cuda()

    
    if opt.checkpoint_path is not None:
        state_dict = torch.load(opt.checkpoint_path)
        model.load_state_dict(state_dict)

    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    summary_fn = summaries.img_summaries
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    loss_fn = val_loss_fn = loss_functions.LFLoss(reg_weight=1)

    if opt.optim_checkpoint_path is not None:
        optimizers = [torch.optim.Adam(lr=opt.lr, params=model.parameters())]
        state_dict = torch.load(opt.optim_checkpoint_path)
        optimizers[0].load_state_dict(state_dict)
    else:
        optimizers=None
    
    wandb.watch(model)
    training.multiscale_training(model=model, dataloader_callback=create_dataloader_callback,
                                 dataloader_iters=(10000, 500000),
                                 dataloader_params=((sidelens[0], batch_sizes[0], None), (sidelens[1], batch_sizes[1], None)),
                                 epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                 epochs_til_checkpoint=opt.epochs_til_ckpt,
                                 model_dir=root_path, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
                                 iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                                 overwrite=True, optimizers=optimizers,
                                 rank=gpu, train_function=training.train, gpus=opt.gpus)

if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()

    opt = parser.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, shared_dict))
    else:
        multigpu_train(0, opt, shared_dict)
