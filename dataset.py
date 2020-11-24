import os
import torch
from PIL import Image

class ColorBW(torch.utils.data.DataLoader):
    """
    Creates a Pytorch dataset to load the both
    coloured and grayscale images according to the
    provided transform.

    Arguments:
    img_dir (path) : directory of the image
    img_list (list) : list containing names of all the images
    transform (torch.transform) : transform required for the image

    """
    def __init__(self, img_dir, img_list, transform = None):
        self.img_dir = img_dir
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
    
        return img