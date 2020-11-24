import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import show_tensor_images, weights_init, epoch_time
from dataset import ColorBW
from model import Generator, Discriminator

#hyperparameters
img_height, img_width = 256, 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
learning_rate = 0.002
num_workers = 4
display_step = 25
criterion = nn.BCEWithLogitsLoss()
img_dir = 'images'
n_epochs = 300
mean_generator_loss = 0
mean_discriminator_loss = 0
cur_step = 0

#transforms
transform_1 = transforms.Compose([
    transforms.Resize((img_height, img_width), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), # to get values between -1 and 1
])

transform_2 = transforms.Compose([
    transforms.Resize((img_height, img_width), interpolation=Image.NEAREST),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), # to get the values between -1 and 1
])

#get image names
img_list = [img_name for img_name in os.listdir(img_dir)]

#create dataset
color_dataset = ColorBW(
    transform=transform_1, img_dir = img_dir, img_list = img_list
    )
bw_dataset = ColorBW(
    transform=transform_2, img_dir = img_dir, img_list = img_list
    )

#create data loader
color_loader = DataLoader(
        dataset=color_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

bw_loader = DataLoader(
        dataset=bw_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

#initialize models
gen = Generator().to(device)
gen_opt = optim.Adam(gen.parameters(), lr = learning_rate)
disc = Discriminator().to(device)
disc_opt = optim.Adam(disc.parameters(), lr = learning_rate)

#intitalize weights
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

#start training!!!
for epoch in range(n_epochs):
    start_time = time.time()
    for color, bw in zip(tqdm(color_loader), tqdm(bw_loader)):
        cur_batch_size = len(color)
        color = color.to(device)
        bw = bw.to(device)

        #train discriminator freezing generator
        disc_opt.zero_grad()
        gen_pred = gen(bw)
        disc_pred_bw = disc(gen_pred.detach())
        disc_loss_bw = criterion(disc_pred_bw, torch.zeros_like(disc_pred_bw))
        disc_pred_co = disc(color)
        disc_loss_co = criterion(disc_pred_co, torch.ones_like(disc_pred_co))

        disc_loss = (disc_loss_bw + disc_loss_co) / 2

        mean_discriminator_loss += disc_loss.item() / display_step
        disc_loss.backward(retain_graph = True)
        disc_opt.step()

        #train generator freezing discriminator
        gen_opt.zero_grad()
        gen_pred_2 = gen(bw)
        disc_pred_2 = disc(gen_pred_2)
        gen_loss = criterion(disc_pred_2, torch.ones_like(disc_pred_2))
        gen_loss.backward()
        gen_opt.step()

        mean_generator_loss += gen_loss.item() / display_step

        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            show_tensor_images(gen_pred)
            show_tensor_images(color)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print()
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

