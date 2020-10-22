import torch
from torch.utils.checkpoint import checkpoint_sequential as chkpt
import torch.nn as nn
from math import ceil

__all__ = [
    'VGG', 'artistic_vgg', 'artistic_vgg_experimental', 'artistic_vgg_experimental_rtrain',
]


class VGG(nn.Module):

    def __init__(self, features, fout_chan, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(fout_chan, 4096, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )
        #self.classifier = nn.Sequential(
        #    nn.Linear(fout_chan * 6 * 6, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096, num_classes),
        #)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.mean(dim=[2, 3])
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg):
    layers = []
    for tup in cfg:
        if tup is 'V':
            layers += [ReceptiveField()]
        else:
            (num, cks, entry, inter) = tup
            layers += [FeatBlock(num, cks, entry, inter)]
    return nn.Sequential(*layers)


class FeatBlock(nn.Module):
    def __init__(self, num_conv_layers, conv_kernel_size, entry_channels, internal_channels):
        super(FeatBlock, self).__init__()
        conv_kernel_padding = int(ceil((conv_kernel_size - 1) / 2))
        layers = [nn.Conv2d(entry_channels, internal_channels, kernel_size=conv_kernel_size, padding=conv_kernel_padding), nn.ReLU(True)]
        for i in range(num_conv_layers - 1):
            layers += [nn.Conv2d(internal_channels, internal_channels, kernel_size=conv_kernel_size, padding=conv_kernel_padding), nn.ReLU(True)]
        layers += [nn.Conv2d(internal_channels, internal_channels, kernel_size=2, stride=2), nn.ReLU(True)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)
        #x *= 0.1
        return x


class ReceptiveField(nn.Module):
    def __init__(self):
        super(ReceptiveField, self).__init__()
        self.dlayers = nn.Sequential(*[nn.Conv2d(3, 96, kernel_size=11, padding=5, stride=2), nn.ReLU(True)])
        self.flayers = nn.Sequential(*[nn.Conv2d(3, 96, kernel_size=3, padding=1), nn.ReLU(True),
                                    nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.ReLU(True),
                                    nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(96, 96, kernel_size=2, stride=2), nn.ReLU(True)])
        self.convpool = nn.Sequential(*[nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(96, 96, kernel_size=2, stride=2), nn.ReLU(True)])


    def forward(self, x):
        u = self.dlayers(x)
        #v = chkpt(self.flayers, 1, x)
        v = self.flayers(x)
        x = (u + v) / 2
        x = self.convpool(x)
        #x *= 0.1
        return x


cfg = {
        'D': [(2, 3, 3, 64), (2, 3, 64, 128), (3, 3, 128, 256), (3, 3, 256, 512), (3, 3, 512, 1024), (3, 3, 1024, 1024)],
        'E': ['V', (2, 3, 96, 192), (3, 3, 192, 384), (4, 3, 384, 768), (5, 3, 768, 1024)],
        'P': ['V', (1, 3, 96, 128), (1, 3, 128, 256), (2, 3, 256, 512), (2, 3, 512, 512)],
}


def artistic_vgg(pretrained=None, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained is not None:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), cfg['D'][-1][-1], **kwargs)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

def artistic_vgg_experimental(pretrained=None, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained is not None:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), cfg['E'][-1][-1], **kwargs)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

def artistic_vgg_experimental_rtrain(pretrained=None, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained is not None:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['P']), cfg['P'][-1][-1], **kwargs)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model



