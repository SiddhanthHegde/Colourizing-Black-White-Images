import time
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor, num_images=4, size=(3, 256, 256)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.

    Arguments:
    image_tensor (torch.tensor) : images to visualize
    num_images (int) : number of images to show at once
    size (tuple of size 3) : size of the image in the form of (in_channels, image_height, image_width) 
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def weights_init(m):
    '''
    Initializes the weights of the corresponding nn layer 
    to the normal distribution

    Arguments:
    m (torch.nn.layers) : type of neural network layer
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def epoch_time(start_time, end_time):
    '''
    Calculates the number of minutes 
    and numbers of seconds elapsed from start_time to end_time

    Arguments:
    start_time (time object) : starting time
    end_time (time object) : ending time
    '''

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




