"""
training code for our super resolution framework
This code has been borrowed from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py
"""
import argparse
import os
import numpy as np
import scipy.io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from model import *
from datasets import *
import torch.nn as nn
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=96, help="size of high res. image height")
    parser.add_argument("--hr_width", type=int, default=96, help="size of high res. image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False
    # models
    generator = GeneratorUNet()
    discriminator = Discriminator()
    feature_extractor = FeatureExtractor()

    # Losses
    criterion_MSE = torch.nn.MSELoss()
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_content = torch.nn.MSELoss()
    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    input_lr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)
    input_hr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)
    input_seg = Tensor(opt.batch_size, 3, opt.hr_height, opt.hr_width)

    dataloader = DataLoader(ImageDataset("data/"), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # Training
    count = 0
    flagD = -1
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            # Configure model input
            imgs_lr = input_lr.copy_(imgs["lr"])
            imgs_hr = input_hr.copy_(imgs["hr"])
            imgs_seg = input_seg.copy_(imgs["seg"])
            name = imgs["name"]

            # optimizers
            LR = opt.lr * (0.5 ** (count // 10000))
            count = count + 1
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(opt.b1, opt.b2))
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR * 0.1, betas=(opt.b1, opt.b2))
            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator((imgs_lr, imgs_seg))

            if count < 1000:
                # Making pretrain model with just mse loss
                loss_mse = criterion_MSE(gen_hr, imgs_hr)
                loss_G = loss_mse

            if count >= 1000:
                # Using content loss and adversarial loss
                flagD = 0
                # Adversarial loss
                pred_fake = discriminator(gen_hr)
                label_true = torch.empty_like(pred_fake).fill_(True)
                loss_gan = criterion_GAN(pred_fake, label_true)
                # Content loss
                gen_features = feature_extractor(gen_hr.repeat(1, 3, 1, 1))
                real_features_high = Variable(feature_extractor(imgs_hr.repeat(1, 3, 1, 1)), requires_grad=False)
                loss_content_high = criterion_MSE(gen_features, real_features_high) * 1

                # Total loss
                loss_G = loss_gan * 5e-3 + loss_content_high * 0.01

            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator

            loss_D = Tensor(1)
            batches_done = epoch * len(dataloader) + i
            if batches_done % 6 == tempD:
                optimizer_D.zero_grad()

                # real image
                pred_real = discriminator(imgs_hr)
                label_true = torch.empty_like(pred_fake).fill_(True)
                loss_d_real = criterion_GAN(pred_real, label_true)

                # fake image
                pred_fake = discriminator(gen_hr.detach())
                label_false = torch.empty_like(pred_fake).fill_(False)
                loss_d_fake = criterion_GAN(pred_fake, label_false)
                loss_D = loss_d_real + loss_d_fake
                loss_D.backward()
                optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [MSE loss: %f] [Learning rate: %f] "
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), loss_mse.item(), LR)
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                adict = {}

                adict["h"] = imgs_hr.data.cpu().numpy()
                adict["gen"] = gen_hr.data.cpu().numpy()
                adict["l"] = imgs_lr.data.cpu().numpy()
                adict["seg"] = imgs_seg.data.cpu().numpy()
                # adict['X'] = imgs.cpu().numpy()
                aa = "images/" + str(batches_done) + ".mat"
                scipy.io.savemat(aa, adict)
                torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % batches_done)
                torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % batches_done)


if __name__ == "__main__":

    main()
