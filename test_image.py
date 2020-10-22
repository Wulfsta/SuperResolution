import argparse
import time

import torch
from PIL import Image
from torchvision import transforms

from model import Generator, Normalize, DeNormalize

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

#div2kmean = torch.tensor([0.4488, 0.4371, 0.4040])
#div2kstd = torch.tenosr([0.2845, 0.2701, 0.2920])
model = Generator(UPSCALE_FACTOR).eval()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
#model = Sequential(
#            Normalize(div2kmean, div2kstd),
#            model,
#            DeNormalize(div2kmean, div2kstd),
#        )
if TEST_MODE:
    model.cuda()

image = Image.open(IMAGE_NAME)
loader = transforms.Compose([
            transforms.ToTensor(), 
        ])

image = loader(image).requires_grad_(False).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()


out = model(image)

out_img = transforms.ToPILImage()(out[0].data.clamp_(0, 1).cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
