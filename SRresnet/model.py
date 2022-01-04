import torch
import torch.nn as nn

class SRresnet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type='plain', norm='bnorm', nblk=16):
        super(SRresnet, self).__init__()

        self.learning_type = learning_type
        self.enc = CBR2d(in_channels, nker, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=0.0)

        res = [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0) for i in range(nblk)]

        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)

        ps1 = [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1),
               nn.PixelShuffle(2),
               nn.ReLU()]
        self.ps1 = nn.Sequential(*ps1)

        ps2 = [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1),
               nn.PixelUnshuffle(2),
               nn.ReLU()]
        self.ps2 = nn.Sequential(*ps2)

        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.enc(x)
        ck_x = x
        x = self.res(x)
        x = ck_x + self.dec(x)

        x = self.ps1(x)
        x = self.ps2(x)

        x = self.fc(x)

        return x


        
        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(ResBlock, self).__init__()
        layers = list()

        layers += [CBR2d(in_channels, out_channels, kernel_size, stride, padding, bias, norm, relu)]
        layers += [CBR2d(out_channels, out_channels, kernel_size, stride, padding, bias, norm, relu=None)]

        self.resblock = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblock(x)


class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(CBR2d, self).__init__()
        layer = [nn.ReflectionPad2d(padding=padding)]
        layer += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=0, bias=bias)]
        if not norm is None:
            layer += [nn.BatchNorm2d(out_channels) if norm == 'bnorm' else nn.InstanceNorm2d(out_channels)]
        if not relu is None:
            layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layer)

    def forward(self, x):
        x2 = torch.tensor(x.clone(), dtype=torch.float32)
        return self.cbr(x2)


class DECBDR2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(DECBDR2d, self).__init__()
        layer = []

        layer += [nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=bias, output_padding=padding)]

        if not norm is None:
            layer += [nn.BatchNorm2d(out_channel) if norm == 'bnorm' else nn.InstanceNorm2d(out_channel)]
        if not relu is None:
            layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.decbdr = nn.Sequential(*layer)

    def forward(self, x):
        x2 = torch.tensor(x.clone(), dtype=torch.float32)
        return self.decbdr(x2)






