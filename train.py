import argparse
import os
from math import log10

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator, Normalize, DeNormalize

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--smallest_crop_size', default=185, type=int, help='training images crop size')
parser.add_argument('--largest_crop_size', default=192, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--start_epoch', default=1, type=int, help='epoch number to resume at')
parser.add_argument('--num_epochs', default=5000, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--lr_G', default=0.0001, type=float, help='learning rate')
parser.add_argument('--lr_D', default=0.0001, type=float, help='learning rate')
parser.add_argument('--lr_adjust_epoch', default=1875, type=int, help='half learning rate at epochs divisible by this number')
parser.add_argument('--generator_pretrain_path', default=None, help='path to pretrained generator network')
parser.add_argument('--discriminator_pretrain_path', default=None, help='path to pretrained discriminator network')
parser.add_argument('--generator_optim_pretrain_path', default=None, help='path to pretrained generator network')
parser.add_argument('--discriminator_optim_pretrain_path', default=None, help='path to pretrained discriminator network')

opt = parser.parse_args()

SMALLEST_CROP_SIZE = opt.smallest_crop_size
LARGEST_CROP_SIZE = opt.largest_crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
START_EPOCH = opt.start_epoch

print(NUM_EPOCHS)
print(START_EPOCH)

train_set = TrainDatasetFromFolder('data/VOC2012/train', smallest_crop_size=SMALLEST_CROP_SIZE, largest_crop_size=LARGEST_CROP_SIZE, upscale_factor=UPSCALE_FACTOR, batch_size=opt.batch_size)
val_set = ValDatasetFromFolder('data/VOC2012/val', crop_size=LARGEST_CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=16, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=16, batch_size=1, shuffle=False)

#div2kmean = ([0.4488, 0.4371, 0.4040])
#div2kstd = ([0.2845, 0.2701, 0.2920])
netG = Generator(UPSCALE_FACTOR)
if opt.generator_pretrain_path is not None:
    print('loading pretrained model at ' + opt.generator_pretrain_path)
    netG.load_state_dict(torch.load(opt.generator_pretrain_path))
else:
    print('initializing generator weights')
    netG.initialize_weights()
#netG = torch.nn.Sequential(
#            Normalize(div2kmean, div2kstd),
#            netG,
#            DeNormalize(div2kmean, div2kstd),
#            )
print('# generator parameters:     ', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
if opt.discriminator_pretrain_path is not None:
    print('loading pretrained model at ' + opt.discriminator_pretrain_path)
    netD.load_state_dict(torch.load(opt.discriminator_pretrain_path))
else:
    print('initializing discriminator weights')
    netD.initialize_weights()
print('# discriminator parameters: ', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()
discriminator_criterion = nn.BCEWithLogitsLoss()

interpolation_layer = nn.functional.interpolate

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()
    discriminator_criterion.cuda()
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_G)
if opt.generator_optim_pretrain_path is not None:
    print('loading pretrained model at ' + opt.generator_optim_pretrain_path)
    optimizerG.load_state_dict(torch.load(opt.generator_optim_pretrain_path))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_D)
if opt.discriminator_optim_pretrain_path is not None:
    print('loading pretrained model at ' + opt.discriminator_optim_pretrain_path)
    optimizerD.load_state_dict(torch.load(opt.discriminator_optim_pretrain_path))

results = {'d_loss': [], 'g_loss': [], 'psnr': [], 'ssim': []}


def adjust_learning_rate(optimizer, epoch, n):
    """Sets the learning rate to the initial LR decayed by half every n epochs"""
    if epoch % n == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2


for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
    adjust_learning_rate(optimizerG, epoch, opt.lr_adjust_epoch)
    adjust_learning_rate(optimizerD, epoch, opt.lr_adjust_epoch)
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'g_c_a_loss': 0, 'g_c_p_loss': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize Relativistic Adversarial Loss
        ###########################
        real_img = target
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = data
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)
        #print(real_img.size()[2:])
        fake_img = interpolation_layer(fake_img, size=(real_img.size()[2:]), mode='bicubic', align_corners=True)

        netD.zero_grad()
        #real_out = netD(real_img).mean()
        #fake_out = netD(fake_img).mean()
        real_out = netD(real_img)
        fake_out = netD(fake_img)
        #d_loss = 1 - real_out + fake_out
        d_loss_real = discriminator_criterion(real_out - fake_out.mean(), torch.ones_like(real_out))
        d_loss_fake = discriminator_criterion(fake_out - real_out.mean(), torch.zeros_like(fake_out))
        d_loss = (d_loss_fake + d_loss_real) / 2
        
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize Relativistic Adversarial Loss + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, real_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        ##fake_img = netG(z)
        ##fake_out = netD(fake_img).mean()
        #fake_img = netG(z)
        #fake_out = netD(fake_img)

        #g_loss = generator_criterion(fake_out, real_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.data * batch_size
        ##d_loss = 1 - real_out + fake_out
        #d_loss_real = discriminator_criterion(real_out - fake_out.mean(), torch.ones_like(real_out))
        #d_loss_fake = discriminator_criterion(fake_out - real_out.mean(), torch.zeros_like(fake_out))
        #d_loss = (d_loss_fake + d_loss_real) / 2
        running_results['d_loss'] += d_loss.data * batch_size
        #running_results['d_score'] += real_out.data * batch_size
        #running_results['g_score'] += fake_out.data * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.10f Loss_G: %.10f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes']))

    netG.eval()
    out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    #val_images = []
    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = val_lr
        lr.requires_grad = False
        hr = val_hr
        hr.requires_grad = False
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).data
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

        #val_images.extend(
        #    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
        #     display_transform()(sr.data.cpu().squeeze(0))])
    #val_images = torch.stack(val_images)
    #val_images = torch.chunk(val_images, val_images.size(0) // 15)
    #val_save_bar = tqdm(val_images, desc='[saving training results]')
    #index = 1
    #for image in val_save_bar:
    #    image = utils.make_grid(image, nrow=3, padding=5)
    #    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
    #    index += 1

    # save model parameters
    if (epoch % 500 == 0) or (epoch == (NUM_EPOCHS + 1)):
        torch.save(netG.state_dict(), 'epochs/experimental_netG_epoch_{}_{}.pth'.format(UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/experimental_netD_epoch_{}_{}.pth'.format(UPSCALE_FACTOR, epoch))
        torch.save(optimizerG.state_dict(), 'epochs/experimental_optimizerG_epoch_{}_{}.pth'.format(UPSCALE_FACTOR, epoch))
        torch.save(optimizerD.state_dict(), 'epochs/experimental_optimizerD_epoch_{}_{}.pth'.format(UPSCALE_FACTOR, epoch))
        #torch.save(generator_criterion.state_dict(), 'epochs/experimental_netG_criterion_epoch_{}_{}.pth'.format(UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 10 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(START_EPOCH, epoch + 1))
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')


