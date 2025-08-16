from __future__ import division
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as  F

class latentEncodeScaleNet(nn.Module):
    def __init__(self, input_dim, gf_dim):
        super(latentEncodeScaleNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, gf_dim),nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(gf_dim, gf_dim * 2),nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(gf_dim * 2, gf_dim * 4),nn.Tanh())
        self.fc4 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc5 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc6 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc7 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc8 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc9 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc10 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc11 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc12 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc13 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())

    def forward(self, input_code):
        # print('input_code',input_code.shape) # ([1, 1, 1, 4])
        hc1 = self.fc1(input_code)
        # print('hc1',hc1.shape) # ([1, 1, 1, 64])
        hc2 = self.fc2(hc1)
        # print('hc2',hc2.shape) # ([1, 1, 1, 128])
        hc3 = self.fc3(hc2)
        # print('hc3',hc3.shape) # ([1, 1, 1, 256])
        hr11 = self.fc4(hc3)
        # print('hr11',hr11.shape) # ([1, 1, 1, 256])
        hr12 = self.fc5(hr11)
        hr21 = self.fc6(hr12)
        hr22 = self.fc7(hr21)
        hr31 = self.fc8(hr22)
        hr32 = self.fc9(hr31)
        hr41 = self.fc10(hr32)
        hr42 = self.fc11(hr41)
        hr51 = self.fc12(hr42)
        hr52 = self.fc13(hr51)
        latentscale = {'hc1': hc1, 'hc2': hc2, 'hc3': hc3,
                'hr11': hr11, 'hr12': hr12,
                'hr21': hr21, 'hr22': hr22,
                'hr31': hr31, 'hr32': hr32,
                'hr41': hr41, 'hr42': hr42,
                'hr51': hr51, 'hr52': hr52}
        for key in latentscale.keys():
            latentscale[key] = latentscale[key].permute(0, 3, 1, 2)
        # print(latentscale['hc1'].shape) # ([1, 64, 1, 1])
        return latentscale

class latentDecodeScaleNet(nn.Module):
    def __init__(self, input_dim, gf_dim):
        super(latentDecodeScaleNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, gf_dim),nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(gf_dim, gf_dim * 2),nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(gf_dim * 2, gf_dim * 4),nn.Tanh())
        self.fc4 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc5 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc6 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc7 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc8 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc9 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        self.fc10 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        # self.fc11 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        # self.fc12 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())
        # self.fc13 = nn.Sequential(nn.Linear(gf_dim * 4, gf_dim * 4),nn.Tanh())

    def forward(self, input_code):
        hd2 = self.fc1(input_code)
        hd1 = self.fc2(hd2)
        hr92 = self.fc3(hd1)
        hr91 = self.fc4(hr92)
        hr82 = self.fc5(hr91)
        hr81 = self.fc6(hr82)
        hr72 = self.fc7(hr81)
        hr71 = self.fc8(hr72)
        hr62 = self.fc9(hr71)
        hr61 = self.fc10(hr62)
        latentscale = {'hd2': hd2, 'hd1': hd1,
                'hr91': hr91, 'hr92': hr92,
                'hr81': hr81, 'hr82': hr82,
                'hr71': hr71, 'hr72': hr72,
                'hr61': hr61, 'hr62': hr62}
        for key in latentscale.keys():
            latentscale[key] = latentscale[key].permute(0, 3, 1, 2)
        return latentscale


class discriminator(nn.Module):
    def __init__(self, in_dim, df_dim):
        super(discriminator, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(in_dim, df_dim, kernel_size=4, stride=2, padding=1, bias=True),
                                   nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(nn.Conv2d(df_dim, df_dim*2, kernel_size=4, stride=2, padding=1, bias=True),
                                   nn.InstanceNorm2d(df_dim*2,affine=True),
                                   nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(df_dim*2, df_dim*4, kernel_size=4, stride=2, padding=1, bias=True),
                                   nn.InstanceNorm2d(df_dim*4,affine=True),
                                   nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(df_dim*4, df_dim*8, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(df_dim*8,affine=True),
                                   nn.LeakyReLU(0.2))

        self.conv4 = nn.Conv2d(df_dim*8, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        h0 = self.conv0(x)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        return h4


class define_D(nn.Module):
    def __init__(self, in_dim, df_dim, n_domains):
        super(define_D, self).__init__()
        self.plex_netD_temp = [discriminator(in_dim, df_dim) for _ in range(n_domains)]
        self.plex_netD = nn.Sequential(*self.plex_netD_temp)
    def forward(self, image, domain_code):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dA = torch.tensor(0.0).cuda().to(device)
        dB = torch.tensor(1.0).cuda().to(device)
        dC = torch.tensor(2.0).cuda().to(device)
        dD = torch.tensor(3.0).cuda().to(device)

        if domain_code[0, 0] == dA:
            netD = self.plex_netD[0](image)
        elif domain_code[0, 0] == dB:
            netD = self.plex_netD[1](image)
        elif domain_code[0, 0] == dC:
            netD = self.plex_netD[2](image)
        elif domain_code[0, 0] == dD:
            netD = self.plex_netD[3](image)
        else:
            return False

        return netD


class encoder_res(nn.Module):
    def __init__(self, in_dim, out_dim, ks=3, s=1):
        super(encoder_res, self).__init__()
        self.p = int((ks - 1) / 2)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=ks, stride=s)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=ks, stride=s)
        self.norm1 = nn.Sequential(nn.InstanceNorm2d(out_dim),
                                   nn.ReLU(inplace=True))
        self.norm2 = nn.InstanceNorm2d(out_dim)

    def forward(self, x, ls1, ls2):
        y = F.pad(x, (self.p,self.p,self.p,self.p),mode='reflect')
        y = torch.mul(self.conv1(y),ls1)
        y = self.norm1(y)
        y = F.pad(y, (self.p,self.p,self.p,self.p),mode='reflect')
        y = torch.mul(self.conv2(y),ls2)
        y = self.norm2(y)
        return y + x

class decoder_res(nn.Module):
    def __init__(self, in_dim, out_dim, ks=3, s=1):
        super(decoder_res, self).__init__()
        self.p = int((ks - 1) / 2)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=ks, stride=s)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=ks, stride=s)
        self.norm1 = nn.Sequential(nn.InstanceNorm2d(out_dim),
                                   nn.ReLU(inplace=True))
        self.norm2 = nn.InstanceNorm2d(out_dim)

    def forward(self, x, ls1, ls2):
        y = F.pad(x, (self.p,self.p,self.p,self.p),mode='reflect')
        y = torch.mul(self.conv1(y),ls1)
        y = self.norm1(y)
        y = F.pad(y, (self.p,self.p,self.p,self.p),mode='reflect')
        y = torch.mul(self.conv2(y),ls2)
        y = self.norm2(y)
        return y + x


class encoder_resnet(nn.Module):
    def __init__(self, in_dim, gf_dim):
        super(encoder_resnet, self).__init__()
        self.in_dim = in_dim
        self.gf_dim = gf_dim
        self.conv1 = nn.Conv2d(self.in_dim, self.gf_dim, kernel_size=7, stride=1)
        self.norm1 = nn.Sequential(nn.InstanceNorm2d(self.gf_dim),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Conv2d(self.gf_dim, self.gf_dim*2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.Sequential(nn.InstanceNorm2d(self.gf_dim*2),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Conv2d(self.gf_dim*2, self.gf_dim*4, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.Sequential(nn.InstanceNorm2d(self.gf_dim*4),
                                   nn.ReLU(inplace=True))

        # Define G network with resnet blocks
        self.res1 = encoder_res(self.gf_dim*4,self.gf_dim*4)
        self.res2 = encoder_res(self.gf_dim*4,self.gf_dim*4)
        self.res3 = encoder_res(self.gf_dim*4,self.gf_dim*4)
        self.res4 = encoder_res(self.gf_dim*4,self.gf_dim*4)
        self.res5 = encoder_res(self.gf_dim*4,self.gf_dim*4)

    def forward(self, x, latentscale):
        # print('x',x.shape) # ([1, 1, 240, 160])
        c0 = F.pad(x,(3,3,3,3),mode='reflect')
        c1 = torch.mul(self.conv1(c0),latentscale['hc1'])
        c1 = self.norm1(c1)
        # print('c1', c1.shape) # ([1, 64, 240, 160])
        c2 = torch.mul(self.conv2(c1),latentscale['hc2'])
        c2 = self.norm2(c2)
        # print('c2', c2.shape) # ([1, 128, 120, 80])
        c3 = torch.mul(self.conv3(c2),latentscale['hc3'])
        c3 = self.norm3(c3)
        # print('c3', c3.shape) # ([1, 256, 60, 40])
        r1 = self.res1(c3,latentscale['hr11'], latentscale['hr12'])
        # print('r1', r1.shape) # ([1, 256, 60, 40])
        r2 = self.res2(r1,latentscale['hr21'], latentscale['hr22'])
        # print('r2', r2.shape) # ([1, 256, 60, 40])
        r3 = self.res3(r2,latentscale['hr31'], latentscale['hr32'])
        # print('r3', r3.shape) # ([1, 256, 60, 40])
        r4 = self.res4(r3,latentscale['hr41'], latentscale['hr42'])
        # print('r4', r4.shape) # ([1, 256, 60, 40])
        r5 = self.res5(r4,latentscale['hr51'], latentscale['hr52'])
        # print('r5', r5.shape) # ([1, 256, 60, 40])
        return r5


class decoder_resnet(nn.Module):
    def __init__(self, in_dim, gf_dim, output_c_dim):
        super(decoder_resnet, self).__init__()
        self.in_dim = in_dim
        self.gf_dim = gf_dim
        self.output_c_dim = output_c_dim
        # Define G network with resnet blocks
        self.res6 = decoder_res(self.gf_dim*4,self.gf_dim*4)
        self.res7 = decoder_res(self.gf_dim*4,self.gf_dim*4)
        self.res8 = decoder_res(self.gf_dim*4,self.gf_dim*4)
        self.res9 = decoder_res(self.gf_dim*4,self.gf_dim*4)

        self.deconv1 = nn.ConvTranspose2d(self.gf_dim*4,self.gf_dim*2,kernel_size=4,stride=2,padding=1)
        self.norm1 = nn.Sequential(nn.InstanceNorm2d(self.gf_dim*2),
                                   nn.ReLU(inplace=True))
        self.deconv2 = nn.ConvTranspose2d(self.gf_dim*2,self.gf_dim,kernel_size=4,stride=2,padding=1)
        self.norm2 = nn.Sequential(nn.InstanceNorm2d(self.gf_dim),
                                   nn.ReLU(inplace=True))
        self.conv = nn.Sequential(nn.Conv2d(self.gf_dim, self.output_c_dim, kernel_size=7, stride=1),
                                  nn.Tanh())

    def forward(self, r5, latentscale):
        # print('r5',r5.shape) # ([1, 256, 60, 40])
        r6 = self.res6(r5,latentscale['hr61'], latentscale['hr62'])
        # print('r6', r6.shape) # ([1, 256, 60, 40])
        r7 = self.res7(r6,latentscale['hr71'], latentscale['hr72'])
        # print('r7', r7.shape) # ([1, 256, 60, 40])
        r8 = self.res8(r7,latentscale['hr81'], latentscale['hr82'])
        # print('r8', r8.shape) # ([1, 256, 60, 40])
        r9 = self.res9(r8,latentscale['hr91'], latentscale['hr92'])
        # print('r9', r9.shape) # ([1, 256, 60, 40])
        d1 = torch.mul(self.deconv1(r9),latentscale['hd1'])
        d1 = self.norm1(d1)
        # print('d1',d1.shape) # ([1, 128, 120, 80])
        d2 = torch.mul(self.deconv2(d1),latentscale['hd2'])
        d2 = self.norm2(d2)
        # print('d2',d2.shape) # ([1, 64, 240, 160])
        d2 = F.pad(d2,(3,3,3,3),mode='reflect')
        pred = self.conv(d2)
        # print('pred',pred.shape) # ([1, 1, 240, 160])
        return pred


def flip_gradient(input_, l=1.0):
    # positive_path = input_ * (l + 1).to(torch.float32)
    positive_path = input_ * (l + 1)
    p = positive_path.detach()
    negative_path = - input_ * l
    return p + negative_path



class classifer(nn.Module):
    def __init__(self, in_dim, df_dim, n_domains):
        super(classifer, self).__init__()
        self.df_dim = df_dim
        self.n_domains = n_domains
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, self.df_dim*2, kernel_size=1, stride=1),nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.df_dim*2, self.df_dim, kernel_size=1, stride=1),nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(self.df_dim, self.df_dim//4, kernel_size=1, stride=1),nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(self.df_dim//4, self.n_domains, kernel_size=1, stride=1),nn.LeakyReLU(0.2))
    def forward(self, r5):
        h0 = flip_gradient(r5, l=1.0)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        return h4


def abs_criterion(in_, target):
    return torch.mean(torch.abs(in_ - target))


def mae_criterion(in_, target):
    return torch.mean((in_-target)**2)


def sce_criterion(logits, labels):
    # print('logits', logits.shape) # torch.float32
    # print('labels',labels.shape) # torch.float32
    labels = labels.to(torch.long)
    result = F.cross_entropy(input=logits, target=labels)
    return torch.mean(result)


def prod_input_code(n_domains, DA):
    input_code = np.zeros(n_domains)
    input_code[DA] = 1.
    return input_code