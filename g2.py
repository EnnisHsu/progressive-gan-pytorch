from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from progan_modules import Generator, Discriminator


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def imagefolder_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)
        return data_loader

    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size + int(image_size * 0.2) + 1),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)

    return loader


def train(generator, discriminator, init_step, loader, total_iter=600000, test = 0):
    step = init_step  # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128
    data_loader = sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)

    # total_iter = 600000
    total_iter_remain = total_iter - (total_iter // 6) * (step - 1)
    # print(total_iter_remain)

    pbar = tqdm(range(total_iter_remain))

    alpha = 0
    # one = torch.FloatTensor([1]).to(device)
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    iteration = 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2 / (total_iter // 6)) * iteration)

        if iteration > total_iter // 6:
            alpha = 0
            iteration = 0
            step += 1

            if step > 6:
                alpha = 1
                step = 6
            data_loader = sample_data(loader, 4 * 2 ** step)
            dataset = iter(data_loader)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label = next(dataset)

        iteration += 1

        while(test > 0):
            with torch.no_grad():
                images = g_running(torch.randn(5 * 10, input_code_size).to(device), step=step, alpha=alpha).data.cpu()

                utils.save_image(
                    images,
                    f'result/{str(test)}_{str(i + 1).zfill(6)}.png',
                    nrow=10,
                    normalize=True,
                    range=(-1, 1))
                test = test - 1
        break;




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    parser.add_argument('--path', type=str,
                        help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--trial_name', type=str, default="test1", help='a brief description of the training trial')
    parser.add_argument('--load_G', type=str, default="test1", help='error load G')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=128,
                        help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=4, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train Dhow many times while train G 1 time')
    parser.add_argument('--init_step', type=int, default=1,
                        help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--total_iter', type=int, default=300000,
                        help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=False, action="store_true",
                        help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--tanh', default=False, action="store_true",
                        help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')

    args = parser.parse_args()

    print(str(args))

    trial_name = args.trial_name
    device = torch.device("cuda:%d" % (args.gpu_id))
    input_code_size = args.z_dim
    batch_size = args.batch_size
    n_critic = args.n_critic

    generator = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm,
                          tanh=args.tanh).to(device)
    discriminator = Discriminator(feat_dim=args.channel).to(device)
    g_running = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm,
                          tanh=args.tanh).to(device)

    ## you can directly load a pretrained model here
    # generator.load_state_dict(torch.load('./trial_experiment-1_2022-01-03_11_36/checkpoint/200000_g.model'))
    # g_running.load_state_dict(torch.load('./trial_experiment-1_2022-01-03_11_36/checkpoint/200000_g.model'))
    generator.load_state_dict(torch.load(args.load_G))
    g_running.load_state_dict(torch.load(args.load_G))    # generator.load_state_dict(torch.load('./tr checkpoint/150000_g.model'))
    # g_running.load_state_dict(torch.load('checkpoint/150000_g.model'))
    # discriminator.load_state_dict(torch.load('checkpoint/150000_d.model'))

    g_running.train(False)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    loader = imagefolder_loader(args.path)

    train(generator, discriminator, args.init_step, loader, args.total_iter, 6)
