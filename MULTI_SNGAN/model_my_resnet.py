# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm

channels = 3
leak = 0.1
w_g = 4

class ResBlockD(nn.Module):

    def __init__(self, in_channels, out_channels,downsample=None):
        super(ResBlockD, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1))
        #self.conv_out = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        self.model = nn.Sequential(
                self.conv1,
                nn.LeakyReLU(leak),
                self.conv2
                )
        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            return self.model(x) + self.downsample(x)
        else:
            return self.model(x) + x

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        #
        self.ln = nn.Linear(z_dim,512*4*4)
        #
        self.model = nn.Sequential(
            #nn.ConvTranspose2d(512, 512, 1, stride=1),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(self.ln(z).view(-1, 512, 4, 4))


class GeneratorG(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        #
        self.ln = nn.Linear(z_dim,512*4*4)
        #
        self.model = nn.Sequential(
            #nn.ConvTranspose2d(512, 512, 1, stride=1),
            #nn.BatchNorm2d(512),
        #nn.ReLU(),

            nn.Sequential(

                nn.Conv2d(512, 256, 3, stride=1, padding=(1,1)),
                nn.LeakyReLU(leak),
                nn.Conv2d(256, 256, 3, stride=1, padding=(1,1))
            ),



            nn.Sequential(

                nn.BatchNorm2d(256),
                nn.LeakyReLU(leak),
                nn.Conv2d(256, 128, 3, stride=1, padding=(1,1)),
                nn.LeakyReLU(leak),
                nn.Conv2d(128, 128, 3, stride=1, padding=(1,1))
            ),



            nn.Sequential(

                nn.BatchNorm2d(128),
                nn.LeakyReLU(leak),
                nn.Conv2d(128, 64, 3, stride=1, padding=(1,1)),
                nn.LeakyReLU(leak),
                nn.Conv2d(64, 64, 3, stride=1, padding=(1,1))
            ),

            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64, channels, 3, stride=1, padding=(1,1)),
                nn.Tanh()
            ),


        )

    def forward(self, z):
        return self.model(self.ln(z).view(-1, 512, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        '''
        self.seq = nn.Sequential(
            SpectralNorm(nn.Conv2d(channels,64,3, 1, padding=1)),
            nn.LeakyReLU(leak),
            ResBlockD(64,64),

            SpectralNorm(nn.Conv2d(64,128, 3, 1, padding=1)),
            nn.LeakyReLU(leak),
            ResBlockD(128,128),

            SpectralNorm(nn.Conv2d(128,256, 3, 1, padding=1)),
            nn.LeakyReLU(leak),
            ResBlockD(256,256),

            SpectralNorm(nn.Conv2d(256,512, 3, 1, padding=1)),
            nn.LeakyReLU(leak),
            ResBlockD(512,512),
        )
        '''
        self.seq = nn.Sequential(
            self.make_layer(channels,64),
            self.make_layer(64,64),
            self.make_layer(64,128),
            self.make_layer(128,128),
            #self.make_layer(128,256),
            #self.make_layer(256,512)
            )
        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 128, 1))

    def make_layer(self,in_channels,out_channels,num=2):
        downsample = None
        if in_channels != out_channels:
            downsample = SpectralNorm(nn.Conv2d(in_channels,out_channels,kernel_size=3 ,stride=1,padding=1,bias=False))
        layers = []
        layers.append(ResBlockD(in_channels,out_channels,downsample))
        for _ in range(1,num):
            layers.append(ResBlockD(out_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        m = x

        m = self.seq(m)
        return self.fc(m.view(-1,w_g * w_g * 128))

