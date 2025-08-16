import os
import torch
import torch.nn.functional as  F
import numpy as np
import scipy.io as sio
import re
import random
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import cv2
import MyLib as ML
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.measure import compare_psnr as psnr
import warnings
import torch.optim as optim
warnings.filterwarnings('ignore')

from model_EGC import *

np.random.seed(19)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ==============================================================================#

valid_dir = './valid'
gt_image_path = r'{}/Dom{}{:03d}.nii.gz'
testResult_image_path = r'{}/Dom{}toDom{}_{:02d}_epoch{}.nii'


tedataSize = 5
teIdVec = np.arange(tedataSize) + 1

dom_name = ['A', 'B', 'C', 'D']
inputDomVec = np.arange(len(dom_name))


def psnr2(img1, img2):
   mse = np.mean( (img1/1. - img2/1.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def normalize(img, domain_id):

    if domain_id == 0:
        img = img / 3000. 
    elif domain_id == 1:
        img = img / 5000.
    elif domain_id == 2:
        img = img / 6000.
    else:
        img = img / 7000. 
    
    img[img > 1.] = 1.
    
    return img

# epochVec = np.arange(9,51) * 2 + 1
epochVec = np.arange(14,61) * 2 + 1

# ssimepoch = []
# maeepoch = []

for epoch in epochVec:

    ssimlist = []
    maelist = []
    psnrlist = []
    for teId in teIdVec:
        for input_domain_id in inputDomVec:

            input_domain_name = dom_name[input_domain_id]
            output_domain_list = np.delete(np.arange(len(dom_name)), input_domain_id)

            gt_path = gt_image_path.format(valid_dir,input_domain_name,teId)
            gt = nib.load(gt_path)
            
            for output_domain_id in output_domain_list:

                output_domain_name = dom_name[output_domain_id]
                
                testResult_path = testResult_image_path.format(valid_dir,output_domain_name,input_domain_name,teId,epoch)
                testResult = nib.load(testResult_path)

                # 获取图像数据
                img1 = gt.get_fdata().astype('single')
                img2 = testResult.get_fdata().astype('single')

                img1 = normalize(img1, input_domain_id)
                img2 = normalize(img2, input_domain_id)


                # print('SSIM: {:.4f}'.format(ssim(img1[:,:,:], img2[:,:,:])) )
                # print("PSNR: {:.4f}".format(psnr2(img1[:,:,:], img2[:,:,:])))
                # print("MAE: {:.4f}".format(np.mean(np.abs(img1[:,:,:] - img2[:,:,:]))))
                # print('---------------------------------')

                ssimlist.append(np.mean(ssim(img1[:,:,:], img2[:,:,:])))
                maelist.append(np.mean(np.mean(np.abs(img1[:,:,:] - img2[:,:,:]))))
                psnrlist.append(np.mean(psnr2(img1[:,:,:], img2[:,:,:])))

    print('epoch: {:2d}'.format(epoch))
    print("SSIM: {:.4f}".format(np.mean(ssimlist)))
    print("MAE: {:.4f}".format(np.mean(maelist)))
    print("PSNR: {:.4f}".format(np.mean(psnrlist)))
    # print("SSIM: mean {:.4f}, std {:.4f}".format(np.mean(ssimlist), np.std(ssimlist)))
    # print("MAE: mean {:.4f}, std {:.4f}".format(np.mean(maelist), np.std(maelist)))
    # print("PSNR: mean {:.4f}, std {:.4f}".format(np.mean(psnrlist),  np.std(psnrlist)))
    print('---------------------------------')





