import torch
from torch import nn
import vgg_models
from collections import OrderedDict


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        loss_network = vgg_models.__dict__['artistic_vgg_experimental'](pretrained='vgg_models/artistic_vgg_experimental_pretrain.pth').features

        dlayers_captures = []
        cap_dlayers = []
        for ayer in loss_network[0].dlayers:
            cap_dlayers.append(ayer)
            if isinstance(ayer, nn.Conv2d):
                dlayers_captures.append(CaptureOutput())
                cap_dlayers.append(dlayers_captures[-1])
        loss_network[0].dlayers = nn.Sequential(*cap_dlayers)
        
        flayers_captures = []
        cap_flayers = []
        for ayer in loss_network[0].flayers:
            cap_flayers.append(ayer)
            if isinstance(ayer, nn.Conv2d):
                flayers_captures.append(CaptureOutput())
                cap_flayers.append(flayers_captures[-1])
        loss_network[0].flayers = nn.Sequential(*cap_flayers)

        convpool_captures = []
        cap_convpool = []
        for ayer in loss_network[0].convpool:
            cap_convpool.append(ayer)
            if isinstance(ayer, nn.Conv2d):
                convpool_captures.append(CaptureOutput())
                cap_convpool.append(convpool_captures[-1])
        loss_network[0].convpool = nn.Sequential(*cap_convpool)

        body_captures = []
        for i, seq in enumerate([yer.convs for yer in loss_network[1:]]):
            cap_body = []
            for ayer in seq:
                cap_body.append(ayer)
                if isinstance(ayer, nn.Conv2d):
                    body_captures.append(CaptureOutput())
                    cap_body.append(body_captures[-1])
            loss_network[i + 1].convs = nn.Sequential(*cap_body)

        loss_network[-1].convs = nn.Sequential(*loss_network[-1].convs[0:-4])
        body_captures = body_captures[0:-1]

        for param in loss_network.parameters():
            param.requires_grad = False
        self.captures = dlayers_captures + flayers_captures + convpool_captures + body_captures
        self.loss_network = loss_network
        self.adversarial_crit = nn.BCEWithLogitsLoss()
        self.element_wise_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, d_fake_out, d_real_out, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = ( self.adversarial_crit(d_real_out - d_fake_out.mean(), torch.zeros_like(d_real_out)) + self.adversarial_crit(d_fake_out - d_real_out.mean(), torch.ones_like(d_fake_out)) ) / 2
        # Perception Loss
        self.loss_network(out_images)
        out_loss_outputs = [cap.output for cap in self.captures]
        self.loss_network(target_images)
        target_loss_outputs = [cap.output for cap in self.captures]
        perception_loss = sum([self.element_wise_loss(o, t) for (o, t) in zip(out_loss_outputs, target_loss_outputs)])
        # Image Loss
        image_loss = self.element_wise_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        real_loss = image_loss + (image_loss.detach() / adversarial_loss.detach()) * adversarial_loss + 1 / 8 * (image_loss.detach() / perception_loss.detach()) * perception_loss + 2e-9 * tv_loss
        #print('Image Loss:       {}'.format(image_loss.detach()))
        #print('Adversarial Loss: {}'.format((image_loss.detach() / adversarial_loss.detach()) * adversarial_loss.detach()))
        #print('Perceptual Loss:  {}'.format((image_loss.detach() / perception_loss.detach()) * perception_loss.detach()))
        #print('TV Loss:          {}'.format(2e-8 * tv_loss.detach()))
        return real_loss


class CaptureOutput(nn.Module):
    def __init__(self):
        super(CaptureOutput, self).__init__()

    def forward(self, x):
        self.output = x
        return x


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
