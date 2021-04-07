#!/home/super/anaconda2/bin/python
import argparse
import torch
import os
import torch.nn as nn
#from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.utils as vutils
#from GAN_losses_iter import DCGAN_G
#from sagan_models import Generator, Discriminator
#from utils import *
#del param
cifar_path = "/home/super/ZZX/cifar-10-batches-py/test"
models_path = './models/sagan_1'

#p="/home/super/ZZX/RelativisticGAN-master/code/extra/2_"
z_dim = 128
batch_size = 64
g_conv_dim = 64
imsize = 64
sample_path = './__out'


ngf = 64
nz = 128
nc=3
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


def genPic(model_path):
	net = _netG(1).cuda()
	net.load_state_dict(torch.load(model_path))
	#net = torch.load(model_path)
	fake = torch.FloatTensor(100, z_dim, 1, 1).cuda()
	for step in range(100):

		z = Variable( fake.normal_(0,1) )

		fake_images=net(z)
		
		#save_image(fake_images.data,os.path.join(sample_path, '{}_fake.png'.format(step + 1)),normalize=True,padding=0)
		for i in range(100):

			vutils.save_image((fake_images.data[i]*.50)+.50, os.path.join(sample_path,'{}_fake.png'.format( (step )*100+(i) )) , normalize=False, padding=0)

	



if __name__ =='__main__':
	parser = argparse.ArgumentParser()

	# Model hyper-parameters
 	parser.add_argument('path', type=str)
	parser.add_argument('--z_dim', type=int, default=128)
	parser.add_argument('--targ', type=str, default='./__out')
	args =parser.parse_args()
	sample_path = args.targ
	print(args.path+":")
	genPic(args.path)
	print("done")

