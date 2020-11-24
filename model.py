import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    Generator to convert black and white images to coloured images

    Arguments:
    in_channels (int) : Number of input channels
    out_channels (int) : Number of output channels
    hidden_dim (int) : Hidden dimension of the neural network
    kernel_size (int) : size of the kernel during convolution layers

    '''

    def __init__(
      self, in_channels = 1, out_channels = 3, hidden_dim = 16, kernel_size = 3
      ):
    
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    '''
    Discriminator to distinguish black and white images with coloured images

    Arguments:
    in_channels (int) : Number of input channels
    out_channels (int) : Number of output channels
    hidden_dim (int) : Hidden dimension of the neural network
    kernel_size (int) : size of the kernel during convolution layers
    stride (int) : Stride for the kernels 

    Note : Default parameters and the number of layers are adjusted such that
    the output of the final layer is (batch_size, 1) for the image dimension (3, 256, 256)
    for higher dimensions parameters and number of layers need to be changed

    '''
    
    def __init__(
      self, in_channels = 3, out_channels = 1, hidden_dim = 16, kernel_size = 6, stride = 2
      ):
    
        super(Discriminator, self).__init__()
    
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size, stride),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size , stride * 2),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size, stride * 2),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim * 4, out_channels, kernel_size, stride * 2)
        )

    def forward(self, x):
        pred = self.disc(x)
        return pred.view(len(pred), -1)