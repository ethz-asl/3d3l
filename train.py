# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import torch
import torch.optim as optim

from tools import common, trainer
from tools.dataloader import *
from nets.patchnet import *
from nets.losses import *

default_net = "Quad_L2Net_ConfCFS()"

db_nc_synth_os1 = """SyntheticPairDataset(
    LidarSynthetic('/media/dominic/Extreme SSD/datasets/new_college/long_experiment/data', skip=(0, -1, 10), crop=False), 
            'RandomScale(64,80,can_upscale=True)', 
            'RandomTilting(0.1), PixelNoise(10), RandomTranslation(50)')"""

db_nc_true_os1 = """LidarPairDataset('/media/dominic/Extreme SSD/datasets/new_college/long_experiment', crop=False, type="OS1")"""

db_eth_synth_os0 = """SyntheticPairDataset(
    LidarSynthetic('/media/dominic/Extreme SSD/datasets/lidarmace_data/ethz_outside/data/', skip=(0, -1, 2), crop=False, type='OS0'), 
            'RandomScale(80,128,can_upscale=True)', 
            'RandomTilting(0.1), PixelNoise(10), RandomTranslation(50)')"""

db_lee_ter_synth_os0 = """SyntheticPairDataset(
    LidarSynthetic('/media/dominic/Extreme SSD/datasets/lidarmace_data/lee_terrace/data/', skip=(0, -1, 2), crop=False, type='OS0'), 
            'RandomScale(80,128,can_upscale=True)', 
            'RandomTilting(0.1), PixelNoise(10), RandomTranslation(50)')"""
data_sources = dict(
    N=db_nc_synth_os1,
    P=db_nc_true_os1,
    E=db_eth_synth_os0,
    T=db_lee_ter_synth_os0
)
lidar_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(64,64,can_upscale=True)',
    distort = 'ColorJitter(0.1,0.1,0.2,0.1)',
    crop    = 'RandomCrop((64, 180))')"""

default_sampler = """NghSampler2(ngh=9, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)"""


default_loss = """MultiLoss(
        1, ReliabilityLoss(`sampler`, base=0.5, nq=12),
        1, CosimLoss(N=`N`),
        1, PeakyLoss(N=`N`))"""


class MyTrainer(trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """

    def forward_backward(self, inputs):
        output = self.net(imgs=[inputs.pop('img1'), inputs.pop('img2')])
        allvars = dict(inputs, **output)
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Train R2D2")

    parser.add_argument("--data-loader", type=str, default=lidar_dataloader)
    parser.add_argument("--train-data", type=str, default=list('P'), nargs='+',
                        choices=set(data_sources.keys()))
    parser.add_argument("--net", type=str, default=default_net, help='network architecture')

    parser.add_argument("--pretrained", type=str, default="", help='pretrained model path')
    parser.add_argument("--save-path", type=str, required=True, help='model save_path path')

    parser.add_argument("--loss", type=str, default=default_loss, help="loss function")
    parser.add_argument("--sampler", type=str, default=default_sampler, help="AP sampler")
    parser.add_argument("--N", type=int, default=8, help="patch size for repeatability")

    parser.add_argument("--epochs", type=int, default=20, help='number of training epochs')
    parser.add_argument("--batch-size", "--bs", type=int, default=4, help="batch size")
    parser.add_argument("--learning-rate", "--lr", type=str, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", type=float, default=5e-4)

    parser.add_argument("--threads", type=int, default=8, help='number of worker threads')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')

    args = parser.parse_args()

    iscuda = common.torch_set_gpu(args.gpu)
    common.mkdir_for(args.save_path)

    # Create data loader
    from datasets import *

    db = [data_sources[key] for key in args.train_data]
    db = eval(args.data_loader.replace('`data`', ','.join(db)).replace('\n', ''))
    print("Training image database =", db)
    loader = threaded_loader(db, iscuda, args.threads, args.batch_size, shuffle=True)

    # create network
    print("\n>> Creating net = " + args.net)
    net = eval(args.net)
    print(f" ( Model size: {common.model_size(net) / 1000:.0f}K parameters )")

    # initialization
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, lambda a, b: a)
        net.load_pretrained(checkpoint['state_dict'])

    # create losses
    loss = args.loss.replace('`sampler`', args.sampler).replace('`N`', str(args.N))
    print("\n>> Creating loss = " + loss)
    loss = eval(loss.replace('\n', ''))

    # create optimizer
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                           lr=args.learning_rate, weight_decay=args.weight_decay)

    train = MyTrainer(net, loader, loss, optimizer)
    if iscuda: train = train.cuda()

    # Training loop #
    for epoch in range(args.epochs):
        print(f"\n>> Starting epoch {epoch}...")
        train()

    print(f"\n>> Saving model to {args.save_path}")
    torch.save({'net': args.net, 'state_dict': net.state_dict()}, args.save_path)
