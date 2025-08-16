from __future__ import division
import math
import pprint
import numpy as np
import copy
from skimage import transform
import imageio
import nibabel as nib
from glob import glob
import torch.utils.data as data
import torch

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

class MyDataset(data.Dataset):
    def __init__(self,args,device):
        self.args = args
        self.device = device
        self.data_domA = glob('../BraTS_new/SlicedData/TrainA/*.npy')
        self.data_domB = glob('../BraTS_new/SlicedData/TrainB/*.npy')
        self.data_domC = glob('../BraTS_new/SlicedData/TrainC/*.npy')
        self.data_domD = glob('../BraTS_new/SlicedData/TrainD/*.npy')
        
        self.A_size = len(self.data_domA)
        self.B_size = len(self.data_domB)
        self.C_size = len(self.data_domC)
        self.D_size = len(self.data_domD)
    def __getitem__(self, index):
        A_index = index % self.A_size
        B_index = index % self.B_size
        C_index = index % self.C_size
        D_index = index % self.D_size
        
        A_img = np.load(self.data_domA[A_index])
        B_img = np.load(self.data_domB[B_index])
        C_img = np.load(self.data_domC[C_index])
        D_img = np.load(self.data_domD[D_index])

        A_img = Aug(A_img, self.args.load_size0, self.args.load_size1, self.args.fine_size0, self.args.fine_size1)
        B_img = Aug(B_img, self.args.load_size0, self.args.load_size1, self.args.fine_size0, self.args.fine_size1)
        C_img = Aug(C_img, self.args.load_size0, self.args.load_size1, self.args.fine_size0, self.args.fine_size1)
        D_img = Aug(D_img, self.args.load_size0, self.args.load_size1, self.args.fine_size0, self.args.fine_size1)

        # A_img = np.array(A_img).astype(np.float32)
        # # print('A_img',A_img.shape) # (240,240)
        # B_img = np.array(B_img).astype(np.float32)
        # C_img = np.array(C_img).astype(np.float32)
        # D_img = np.array(D_img).astype(np.float32)
        A_img = torch.from_numpy(np.ascontiguousarray(A_img)).float()
        B_img = torch.from_numpy(np.ascontiguousarray(B_img)).float()
        C_img = torch.from_numpy(np.ascontiguousarray(C_img)).float()
        D_img = torch.from_numpy(np.ascontiguousarray(D_img)).float()

        # A_img = torch.FloatTensor(A_img).cuda().to(self.device)
        # B_img = torch.FloatTensor(B_img).cuda().to(self.device)
        # C_img = torch.FloatTensor(C_img).cuda().to(self.device)
        # D_img = torch.FloatTensor(D_img).cuda().to(self.device)
        
        return {'A': A_img, 'B': B_img, 'C': C_img, 'D': D_img}
        
    def __len__(self):
        return max(self.A_size,self.B_size,self.C_size,self.D_size)


def Aug(img_A, load_size0=256, load_size1=256, fine_size0=240, fine_size1=240):
    padA_size0 = load_size0 - img_A.shape[0]
    padA_size1 = load_size1 - img_A.shape[1]

    img_A = np.pad(img_A, ((int(padA_size0 // 2), int(padA_size0) - int(padA_size0 // 2)),
                            (int(padA_size1 // 2), int(padA_size1) - int(padA_size1 // 2))), mode='constant',
                    constant_values=-1)
    
    h1 = int(np.ceil(np.random.uniform(1e-2, load_size0 - fine_size0)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size1 - fine_size1)))
    img_A = img_A[h1:h1 + fine_size0, w1: w1 + fine_size1]

    return img_A


class Valid_Dataset(data.Dataset):
    def __init__(self):
        # self.args = args
        # self.device = device
        self.data_domA = glob('../BraTS_new/VolumeData/Valid_Slice/DomA/*.npy')
        self.data_domB = glob('../BraTS_new/VolumeData/Valid_Slice/DomB/*.npy')
        self.data_domC = glob('../BraTS_new/VolumeData/Valid_Slice/DomC/*.npy')
        self.data_domD = glob('../BraTS_new/VolumeData/Valid_Slice/DomD/*.npy')
        
        self.A_size = len(self.data_domA)
        self.B_size = len(self.data_domB)
        self.C_size = len(self.data_domC)
        self.D_size = len(self.data_domD)
    def __getitem__(self, index):
        A_index = index % self.A_size
        B_index = index % self.B_size
        C_index = index % self.C_size
        D_index = index % self.D_size
        
        A_img = np.load(self.data_domA[A_index])
        B_img = np.load(self.data_domB[B_index])
        C_img = np.load(self.data_domC[C_index])
        D_img = np.load(self.data_domD[D_index])
   
        return {'A': A_img, 'B': B_img, 'C': C_img, 'D': D_img}
        
    def __len__(self):
        return max(self.A_size,self.B_size,self.C_size,self.D_size)


def load_test_data(image_path, domain_id):

    imgAll = nib.load(image_path)
    img = imgAll.get_data().astype('single')
    # print('img',img.shape) # (240, 240, 160)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = torch.FloatTensor(img).cuda().to(device).unsqueeze(0)
    if domain_id == 0:
        img = img / 3000. * 2. - 1.
    elif domain_id == 1:
        img = img / 5000. * 2. - 1.
    elif domain_id == 2:
        img = img / 6000. * 2. - 1.
    else:
        img = img / 7000. * 2. - 1.
    
    img[img > 1.] = 1.
    
    return img

def load_test_data2(imgAll, domain_id):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = imgAll.get_data().astype('single')
    img = torch.FloatTensor(img).cuda().to(device).unsqueeze(0)
    # print('img',img.shape) # (240, 240, 160)
    if domain_id == 0:
        img = img / 3000. * 2. - 1.
    elif domain_id == 1:
        img = img / 5000. * 2. - 1.
    elif domain_id == 2:
        img = img / 6000. * 2. - 1.
    else:
        img = img / 7000. * 2. - 1.
    
    img[img > 1.] = 1.
    
    return img



def load_train_data(image_path, load_size0=256, load_size1=256, fine_size0=240, fine_size1=240, is_testing=False):
    
    img_A = np.load(image_path[0])
    img_B = np.load(image_path[1])
    
    
    if not is_testing:

        padA_size0 = load_size0 - img_A.shape[0]
        padA_size1 = load_size1 - img_A.shape[1]
        padB_size0 = load_size0 - img_B.shape[0]
        padB_size1 = load_size1 - img_B.shape[1]

        img_A = np.pad(img_A, ((int(padA_size0 // 2), int(padA_size0) - int(padA_size0 // 2)),
                               (int(padA_size1 // 2), int(padA_size1) - int(padA_size1 // 2))), mode='constant',
                       constant_values=-1)
        img_B = np.pad(img_B, ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
                               (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))), mode='constant',
                       constant_values=-1)
        
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size0 - fine_size0)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size1 - fine_size1)))
        img_A = img_A[h1:h1 + fine_size0, w1: w1 + fine_size1]
        img_B = img_B[h1:h1 + fine_size0, w1: w1 + fine_size1]

    else:
        
        padA_size = fine_size0 - img_A.shape[0]
        padB_size = fine_size0 - img_B.shape[0]

        img_A = np.pad(img_A, ((int(padA_size // 2), int(padA_size) - int(padA_size // 2)), (0, 0)), mode='constant',
                       constant_values=-1)
        img_B = np.pad(img_B, ((int(padB_size // 2), int(padB_size) - int(padB_size // 2)), (0, 0)), mode='constant',
                       constant_values=-1)
    # print('imageA',img_A.shape) # (240, 160)
    # print('imageB',img_B.shape) # (240, 160)

    img_AB = np.dstack((img_A, img_B))
    # print('imageAB',img_AB.shape) # (240, 160, 2)

    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

