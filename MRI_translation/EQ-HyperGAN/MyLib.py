# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:50:06 2018

@author: XieQi
"""

import numpy as np
import matplotlib.pyplot as plt
# import MyLib as ML
import os
import cv2
import nibabel as nib
import math

def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (np.maximum(maxX - minX, 0.0001))
    return X * 255


def normalized_Band(X):
    for i in range(3):
        maxX = np.max(X[:, :, i])
        minX = np.min(X[:, :, i])
        X[:, :, i] = np.minimum((X[:, :, i] - minX) / (np.maximum(maxX - minX, 0.0001)) * 1.5, 1)
    return X


def setRange(X, maxX=1, minX=0):
    X = (X - minX) / (maxX - minX + 0.0001)
    return X


def get3band_of_tensor(outX, nbanch=0, nframe=[0, 1, 2]):
    X = outX[:, :, :, nframe]
    X = X[nbanch, :, :, :]
    return X


def imshow(X):
    #    X = ML.normalized(X)
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imshow(X)
    plt.axis('off')
    plt.show()


# def imwrite(X, saveload='tempIm'):
#     X = np.maximum(X, 0)
#     X = np.minimum(X, 1)
#     plt.imsave(saveload, X)
#     plt.close()
#     # print("X shape",X.shape)
#     # cv2.imwrite(saveload,X)

def imwrite2(X, saveload='tempIm'):
    plt.imsave(saveload, X, cmap='gray')
    plt.close()

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There is " + path + " !  ---")

# file = "test/"
# mkdir(file)
def psnr(img1, img2):
   mse = np.mean( (img1/1. - img2/1.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

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

def getGT():
    DomA001 = nib.load('./valid/DomA001.nii.gz')
    DomA002 = nib.load('./valid/DomA002.nii.gz')
    DomA003 = nib.load('./valid/DomA003.nii.gz')
    DomA004 = nib.load('./valid/DomA004.nii.gz')
    DomA005 = nib.load('./valid/DomA005.nii.gz')

    DomB001 = nib.load('./valid/DomB001.nii.gz')
    DomB002 = nib.load('./valid/DomB002.nii.gz')
    DomB003 = nib.load('./valid/DomB003.nii.gz')
    DomB004 = nib.load('./valid/DomB004.nii.gz')
    DomB005 = nib.load('./valid/DomB005.nii.gz')

    DomC001 = nib.load('./valid/DomC001.nii.gz')
    DomC002 = nib.load('./valid/DomC002.nii.gz')
    DomC003 = nib.load('./valid/DomC003.nii.gz')
    DomC004 = nib.load('./valid/DomC004.nii.gz')
    DomC005 = nib.load('./valid/DomC005.nii.gz')

    DomD001 = nib.load('./valid/DomD001.nii.gz')
    DomD002 = nib.load('./valid/DomD002.nii.gz')
    DomD003 = nib.load('./valid/DomD003.nii.gz')
    DomD004 = nib.load('./valid/DomD004.nii.gz')
    DomD005 = nib.load('./valid/DomD005.nii.gz')

    return DomA001,DomA002,DomA003,DomA004,DomA005,\
           DomB001,DomB002,DomB003,DomB004,DomB005,\
           DomC001,DomC002,DomC003,DomC004,DomC005,\
           DomD001,DomD002,DomD003,DomD004,DomD005