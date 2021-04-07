from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--inf', help="path to input file")
parser.add_argument('--outf', help="path to output file")
parser.add_argument('--imageSize', type=int, default=64, help='imageSize')
opt = parser.parse_args()

dataset = dset.ImageFolder(root=opt.inf,
                        transform=transforms.Compose([
                        transforms.Resize(opt.imageSize),
                        transforms.CenterCrop(opt.imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=32)
for i, data in enumerate(dataloader, 0):
	pic, _ = data
	vutils.save_image(pic,
		            '%s/scaled_%03d.png' % (opt.outf, i),
		            normalize=True,padding=0)
