from __future__ import division
import os
import time
import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
import nibabel as nib
from module import *
from utils import *
from PIL import Image
from skimage import transform
import imageio
import random

class ED_gan(nn.Module):
    def __init__(self, args):
        super(ED_gan, self).__init__()
        self.code_size2 = args.n_domains
        self.input_c_dim = args.input_nc
        self.gf_dim = args.ngf
        self.output_c_dim = args.output_nc
        # a fully connected layer for modifying the filters in generators based on input code.
        self.latentEncodeScaleNet = latentEncodeScaleNet(self.code_size2, self.gf_dim)
        self.latentDecodeScaleNet = latentDecodeScaleNet(self.code_size2, self.gf_dim)
        # cycle of "real_A to fake_B to recon_A" or cycle of "real_B to fake_A to recon_B"
        
        self.decoder_resnet = decoder_resnet(self.gf_dim*4, self.gf_dim, self.output_c_dim)
        self.encoder_resnet = encoder_resnet(self.input_c_dim, self.gf_dim)
        
        self.df_dim = args.ndf
        self.n_domains = args.n_domains
        self.classifer = classifer(self.gf_dim*4, self.df_dim, self.n_domains)
    def forward(self, real_codeA, real_codeB, real_A, real_B):
        # a fully connected layer for modifying the filters in generators based on input code.
        latEnScl_A = self.latentEncodeScaleNet(real_codeA)
        latEnScl_B = self.latentEncodeScaleNet(real_codeB)
        latDeScl_A = self.latentDecodeScaleNet(real_codeA)
        latDeScl_B = self.latentDecodeScaleNet(real_codeB)
        for key in latEnScl_A.keys():
            latEnScl_A[key] = latEnScl_A[key].unsqueeze(2)
        for key in latEnScl_B.keys():
            latEnScl_B[key] = latEnScl_B[key].unsqueeze(2)
        for key in latDeScl_A.keys():
            latDeScl_A[key] = latDeScl_A[key].unsqueeze(2)
        for key in latDeScl_B.keys():
            latDeScl_B[key] = latDeScl_B[key].unsqueeze(2)
        # cycle of "real_A to fake_B to recon_A"
        encode_A = self.encoder_resnet(real_A, latEnScl_A)
        fake_B = self.decoder_resnet(encode_A, latDeScl_B)
        encode_fakeB = self.encoder_resnet(fake_B, latEnScl_B)
        fake_A_ = self.decoder_resnet(encode_fakeB, latDeScl_A)

        # cycle of "real_B to fake_A to recon_B"
        encode_B = self.encoder_resnet(real_B, latEnScl_B)
        fake_A = self.decoder_resnet(encode_B, latDeScl_A)
        encode_fakeA = self.encoder_resnet(fake_A, latEnScl_A)
        fake_B_ = self.decoder_resnet(encode_fakeA, latDeScl_B)

        # self reconstruction.
        recon_A = self.decoder_resnet(encode_A, latDeScl_A)
        recon_B = self.decoder_resnet(encode_B, latDeScl_B)

        # classifier
        LabA = self.classifer(encode_A)
        LabB = self.classifer(encode_B)
        LabfakeB = self.classifer(encode_fakeB)
        LabfakeA = self.classifer(encode_fakeA)

        return encode_A, fake_B, encode_fakeB, fake_A_, encode_B, fake_A, encode_fakeA, fake_B_, recon_A, recon_B, LabA, LabB, LabfakeB, LabfakeA


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.df_dim = args.ndf
        self.n_domains = args.n_domains
        self.define_D = define_D(self.output_c_dim, self.df_dim, self.n_domains)
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.trunc_normal_(m.weight.data, std=0.02, a=-0.04, b=0.04)
    #             m.bias.data.fill_(0.00)
    #         if isinstance(m, nn.ConvTranspose2d):
    #             torch.nn.init.trunc_normal_(m.weight.data, std=0.02 ,a=-0.04, b=0.04)
    #             m.bias.data.fill_(0.00)
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.normal_(m.weight.data, std=0.02)
    #             m.bias.data.fill_(0.0)
    def forward(self, x, D):
        # print('x',x.shape) # ([1, 1, 237, 157])
        D_out = self.define_D(x, D)
        return D_out

# class Discriminator(nn.Module):
#     def __init__(self, args):
#         super(Discriminator, self).__init__()
#         self.input_c_dim = args.input_nc
#         self.output_c_dim = args.output_nc
#         self.df_dim = args.ndf
#         self.n_domains = args.n_domains
#         self.define_D = define_D(self.output_c_dim, self.df_dim, self.n_domains)
#     def forward(self, fake_A, fake_B, real_A, real_B, DB, DA):
#         DB_fake = self.define_D(fake_B, DB)
#         DA_fake = self.define_D(fake_A, DA)
#         DB_real = self.define_D(real_B, DB)
#         DA_real = self.define_D(real_A, DA)
#         return DB_fake, DA_fake, DB_real, DA_real

# class Classifier(nn.Module):
#     def __init__(self,args):
#         super(Classifier, self).__init__()
#         self.df_dim = args.ndf
#         self.gf_dim = args.ngf
#         self.n_domains = args.n_domains
#         self.classifer = classifer(self.gf_dim*4, self.df_dim, self.n_domains)
#     def forward(self, encode):
#         Lab = self.classifer(encode)
#         return Lab

# class Classifier(nn.Module):
#     def __init__(self,args):
#         super(Classifier, self).__init__()
#         self.df_dim = args.ndf
#         self.gf_dim = args.ngf
#         self.n_domains = args.n_domains
#         self.classifer = classifer(self.gf_dim*4, self.df_dim, self.n_domains)
#     def forward(self, encode_A, encode_B, encode_fakeB, encode_fakeA):
#         LabA = self.classifer(encode_A)
#         LabB = self.classifer(encode_B)
#         LabfakeB = self.classider(encode_fakeB)
#         LabfakeA = self.classider(encode_fakeA)
#         return LabA, LabB, LabfakeB, LabfakeA







