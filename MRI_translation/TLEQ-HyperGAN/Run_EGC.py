import argparse
from glob import glob
import os
import torch
import torch.nn.functional as  F
import numpy as np
import scipy.io as sio
from utils import *
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
from CreatDataset import *
from torch.utils.data import DataLoader
from MyLib import normalize
from MyLib import psnr2
from MyLib import getGT

torch.manual_seed(3) # current cpu
torch.cuda.manual_seed(3) # current gpu
torch.cuda.manual_seed_all(3) # all gpu
np.random.seed(3) # numpy module
random.seed(3) # python random module
# 参数设置
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_dir', dest='dataset_dir', default='MRCT', help='path of the dataset')

parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=300, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--max_update_num', dest='max_update_num', type=int, default=3000, help='# updates at each epoch')

parser.add_argument('--n_domains', dest='n_domains', type=int, default=4, help='# domain numbers in multi-modal synthesis')

parser.add_argument('--load_size0', dest='load_size0', type=int, default=256, help='scale images to this size')
parser.add_argument('--load_size1', dest='load_size1', type=int, default=256, help='scale images to this size')
parser.add_argument('--fine_size0', dest='fine_size0', type=int, default=240, help='then crop to this size')
parser.add_argument('--fine_size1', dest='fine_size1', type=int, default=240, help='then crop to this size')

parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')

parser.add_argument("--milestone", type=int, default=[200,300], help="When to decay learning rate")
parser.add_argument('--lr', dest='lr', type=float, default=0.00045, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='BtoA', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, valid or test')

parser.add_argument('--save_freq', dest='save_freq', type=int, default=2,
                    help='save a model every save_freq epochs')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=300,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True,
                    help='if continue training, load the latest model: 1: true, 0: false')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--L2_lambda', dest='L2_lambda', type=float, default=0.2, help='weight on L2 term, i.e., classifier loss with grl')
parser.add_argument('--L3_lambda', dest='L3_lambda', type=float, default=0.5, help='weight on L3 term, i.e., common space loss')
parser.add_argument('--L4_lambda', dest='L4_lambda', type=float, default=10.0, help='weight on L4 term, i.e., self reconstruct loss')

parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True,
                    help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=0,
                    help='max size of image pool, 0 means do not use image pool')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# ==============================================================================#

def train(ED_gan,Discriminator,optimizerED,lr_shedulerED,optimizerD,lr_schedulerD):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if args.use_lsgan:
        criterionGAN = mae_criterion
    else:
        criterionGAN = sce_criterion


    start_time = time.time()
    train_data = MyDataset(args,device)
    train_loader = DataLoader(train_data,batch_size=args.batch_size, num_workers=8, shuffle=True)

    for epoch in range(args.epoch):

        lrED = optimizerED.param_groups[0]['lr']
        lrD = optimizerD.param_groups[0]['lr']
        print('epoch : %d, lr_ED = %f, lr_D = %f' % (epoch, lrED, lrD))

        Training_Loss = 0

        datakey = ['A','B','C','D']
        for ii, data in enumerate(train_loader):
            # domain_code DA,DB是两个数
            DA, DB = random.sample(range(args.n_domains), 2)
            domain_code_tmp = np.array(np.dstack((DA, DB))).astype(np.float32)
            domain_code = domain_code_tmp.reshape([1, args.input_nc + args.output_nc])
            domain_code = np.repeat(domain_code,args.batch_size,axis=0)
            # print(domain_code) # [[2. 3.] [2. 3.]]
            # input_code 建立one-hot编码
            code_domA = prod_input_code(args.n_domains, DA)
            code_domB = prod_input_code(args.n_domains, DB)
            input_code_tmp = np.array(np.hstack((code_domA, code_domB))).astype(np.float32)
            input_code = input_code_tmp.reshape([1, 1, 1, args.n_domains + args.n_domains])
            input_code = np.repeat(input_code,args.batch_size,axis=0)
            # 随机选取两个域（对比度）的数据
            # input image in two domains.
            real_A = data[datakey[DA]]
            # print(real_A.shape) ([2,240,240])
            real_B = data[datakey[DB]]

            # one hot code for distinguishing input domain, e.g., [1,0,0], [0,1,0], etc.
            real_codeA = input_code[:, :, :, :args.n_domains]
            # print('real_codeA',real_codeA.shape) # (1, 1, 1, 4)
            real_codeB = input_code[:, :, :, args.n_domains:args.n_domains + args.n_domains]
            # a number for denoting input domain, e.g., 1, 2, etc.
            DA = domain_code[:, :args.input_nc]
            DB = domain_code[:, args.input_nc:args.input_nc + args.output_nc]

            real_codeA = torch.FloatTensor(real_codeA).cuda().to(device)
            real_codeB = torch.FloatTensor(real_codeB).cuda().to(device)
            DA = torch.FloatTensor(DA).cuda().to(device)
            DB = torch.FloatTensor(DB).cuda().to(device)
            # real_A = torch.FloatTensor(real_A).cuda().to(device)
            # real_B = torch.FloatTensor(real_B).cuda().to(device)
            real_A = real_A.cuda().to(device)
            real_B = real_B.cuda().to(device)
            real_A = torch.unsqueeze(real_A,dim=1)
            # print(real_A.shape) # ([2,1,240,240])
            real_B = torch.unsqueeze(real_B,dim=1)
            # real_codeA = real_codeA.permute(0, 3, 1, 2)
            # real_codeB = real_codeB.permute(0, 3, 1, 2)

            #################################################################################
            # (1) Update ED_gan and Classifier
            ED_gan.train()
            ED_gan.zero_grad()

            encode_A,fake_B,encode_fakeB,fake_A_,encode_B,fake_A,encode_fakeA,fake_B_,recon_A,recon_B,LabA,LabB,LabfakeB,LabfakeA = ED_gan(real_codeA,real_codeB,real_A,real_B)
            DB_fake = Discriminator(fake_B, DB)
            # print('DB_fake', DB_fake.shape) # ([1, 1, 29, 19])
            DA_fake = Discriminator(fake_A, DA)

            # define generator loss
            g_loss_ads = criterionGAN(DB_fake, torch.ones_like(DB_fake)) + criterionGAN(DA_fake, torch.ones_like(DA_fake))
            g_loss_cycle = abs_criterion(real_A, fake_A_) + abs_criterion(real_B, fake_B_)
            # print('LabA',LabA.shape) # ([2, 4, 60, 60])
            # print('DA',DA.shape) # ([2,1])
            B,C,H,W = LabA.shape
            # print(DA[0:1,:].shape) # ([1,1])
            g_loss_class = sce_criterion(LabA, torch.ones(B,H,W).cuda().to(device)*DA[0:1,:]) \
                           + sce_criterion(LabB, torch.ones(B,H,W).cuda().to(device)*DB[0:1,:])
            g_loss_class_fake = sce_criterion(LabfakeA, torch.ones(B,H,W).cuda().to(device)*DA[0:1,:]) \
                                + sce_criterion(LabfakeB, torch.ones(B,H,W).cuda().to(device)*DB[0:1,:])


            # real_codeA = real_codeA.permute(0, 3, 1, 2)
            # real_codeB = real_codeB.permute(0, 3, 1, 2)

            # g_loss_class = sce_criterion(LabA, torch.mul(torch.ones_like(LabA), real_codeA)) \
            #                     + sce_criterion(LabB, torch.mul(torch.ones_like(LabB), real_codeB))
            # g_loss_class_fake = sce_criterion(LabfakeA, torch.mul(torch.ones_like(LabfakeA), real_codeA)) \
            #                          + sce_criterion(LabfakeB, torch.mul(torch.ones_like(LabfakeB), real_codeB))

            g_loss_common = abs_criterion(encode_A, encode_fakeB) + abs_criterion(encode_B, encode_fakeA)

            g_loss_recon = abs_criterion(real_A, recon_A) + abs_criterion(real_B, recon_B)

            g_loss = g_loss_ads \
                          + args.L1_lambda * g_loss_cycle \
                          + args.L2_lambda * g_loss_class \
                          + args.L2_lambda * g_loss_class_fake \
                          + args.L3_lambda * g_loss_common \
                          + args.L4_lambda * g_loss_recon

            g_loss.backward()
            optimizerED.step()
            #################################################################################
            # (2) Update Discriminator
            Discriminator.train()
            Discriminator.zero_grad()

            DB_real = Discriminator(real_B, DB)
            DA_real = Discriminator(real_A, DA)
            DB_fake_sample = Discriminator(fake_B.detach(), DB)
            DA_fake_sample = Discriminator(fake_A.detach(), DA)

            db_loss_real = criterionGAN(DB_real, torch.ones_like(DB_real))
            db_loss_fake = criterionGAN(DB_fake_sample, torch.zeros_like(DB_fake_sample))
            db_loss = (db_loss_real + db_loss_fake) / 2
            da_loss_real = criterionGAN(DA_real, torch.ones_like(DA_real))
            da_loss_fake = criterionGAN(DA_fake_sample, torch.zeros_like(DA_fake_sample))
            da_loss = (da_loss_real + da_loss_fake) / 2
            d_loss = da_loss + db_loss

            d_loss.backward()
            optimizerD.step()
            sum_loss = g_loss + d_loss
            Training_Loss += sum_loss
            # show
            if np.mod(ii, args.print_freq) == 0:
                CurLoss = Training_Loss / (ii + 1)
                # print(("Epoch: [%2d] [%4d/%4d] loss: [%.4f] time: %4.4f" % (epoch, idx, batch_idxs, CurLoss, (time.time() - start_time)/3600)))
                print(("Epoch: [%2d] [%4d] loss: [%.4f] time: %4.4f" % (epoch, ii, CurLoss, (time.time() - start_time)/3600)))
                inputB = real_B.permute(0,2,3,1).cpu().detach().numpy()
                inputB = inputB[0,:,:,0]
                # inputB = np.load(dataB_idxs[0])

                # print('inputB',inputB.shape) # (240, 160)
                # print('fake_A',fake_A.shape) # ([1, 1, 240, 160])
                fake_A = fake_A.cpu().detach().numpy()
                syn_A = fake_A[0,0,:,:]
                fake_B_ = fake_B_.cpu().detach().numpy()
                recon_B = fake_B_[0,0,:,:]

                inputA = real_A.permute(0,2,3,1).cpu().detach().numpy()
                inputA = inputA[0,:,:,0]
                # inputA = np.load(dataA_idxs[0])
                fake_B = fake_B.cpu().detach().numpy()
                syn_B = fake_B[0,0,:,:]
                fake_A_ = fake_A_.cpu().detach().numpy()
                recon_A = fake_A_[0,0,:,:]

                images_B2A = np.concatenate([inputB, syn_A, recon_B], axis=1)
                images_A2B = np.concatenate([inputA, syn_B, recon_A], axis=1)
                images_show = np.concatenate([images_B2A, images_A2B], axis=0)

                # imageio.imwrite('./{}/Epoch{:02d}_{:04d}.jpg'.format(args.sample_dir, epoch, idx), (images_show + 1.) * 127.5)
                ML.imwrite2((images_show + 1.) * 127.5,'./{}/Epoch{:02d}_{:04d}.jpg'.format(args.sample_dir, epoch, ii))

        # adjust the learning rate
        lr_shedulerED.step()
        lr_schedulerD.step()
        if epoch >= 90:
            # ED_gan.train()
            save_path_model = os.path.join(args.checkpoint_dir, 'EDgan_state_' + str(epoch + 1) + '.pth')
            torch.save(ED_gan.state_dict(), save_path_model)
        # if np.mod(epoch, args.save_freq) == 0:
        #     ED_gan.train()
        #     save_path_model = os.path.join(args.checkpoint_dir, 'EDgan_state_' + str(epoch + 1) + '.pth')
        #     torch.save(ED_gan.state_dict(), save_path_model)
        #     # save_path_model = os.path.join(args.checkpoint_dir, 'Discriminator_state_' + str(epoch + 1) + '.pth')
        #     # torch.save(Discriminator.state_dict(), save_path_model)


def valid():
    """valid cyclegan"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_path = r'../BraTS_new/VolumeData/Valid/Valid{}/Dom{}{:03d}.nii.gz'

    DomA001,DomA002,DomA003,DomA004,DomA005,\
    DomB001,DomB002,DomB003,DomB004,DomB005,\
    DomC001,DomC002,DomC003,DomC004,DomC005,\
    DomD001,DomD002,DomD003,DomD004,DomD005 = getGT()
 
    GTdic = {'A001':DomA001, 'A002':DomA002,'A003':DomA003,'A004':DomA004,'A005':DomA005,\
             'B001':DomB001,'B002':DomB002,'B003':DomB003,'B004':DomB004,'B005':DomB005,\
             'C001':DomC001,'C002':DomC002,'C003':DomC003,'C004':DomC004,'C005':DomC005,\
             'D001':DomD001,'D002':DomD002,'D003':DomD003,'D004':DomD004,'D005':DomD005}

    valNum = 5
    valVec = np.arange(valNum) + 1 # [1 2 3 4 5]

    # epochVec = np.arange(15,75) * 2 + 1  # [31 33 35 ... 121]
    epochVec = np.arange(91,201)

    dataInfoFile = open(r'../BraTS_new/dataInfo.txt', 'r')
    sourceInLines = dataInfoFile.readlines()
    dataInfoFile.close()
    dataInfo = []
    for line in sourceInLines:
        temp1 = line.strip('\n')
        temp2 = temp1.split(' ')
        dataInfo.append(temp2)

    dom_name = ['A', 'B', 'C', 'D']
    inputDomVec = np.arange(len(dom_name))
    start_time = time.time()
    for epoch in epochVec:
        print('########## epoch : %d ##########' % (epoch))
        ssimlist = []
        maelist = []
        psnrlist = []
        # 每隔2个epoch测一下
        # valid_dir = './valid'
        # if not os.path.exists(valid_dir):
        #     os.makedirs(valid_dir)

        valid_checkpoint = './checkpoint'

        # print(" [*] Reading checkpoint...")
        model_dir = 'EDgan_state_' + str(epoch) + '.pth'
        checkpoint_dir = os.path.join(valid_checkpoint, model_dir)
        with torch.no_grad():
            netGAN = ED_gan(args).to(device)
            netGAN.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))
            netGAN.eval()
            for input_domain_id in inputDomVec:
                # id依次取0，1，2，3
                input_domain_name = dom_name[input_domain_id] # 依次取A,B,C,D
                output_domain_list = np.delete(np.arange(len(dom_name)), input_domain_id)

                for output_domain_id in output_domain_list:
                    output_domain_name = dom_name[output_domain_id]
                    if output_domain_id == 0:
                        output_range = 3000
                    elif output_domain_id == 1:
                        output_range = 5000
                    elif output_domain_id == 2:
                        output_range = 6000
                    else:
                        output_range = 7000

                    # input_code
                    code_domA = prod_input_code(args.n_domains, input_domain_id)
                    code_domB = prod_input_code(args.n_domains, output_domain_id)
                    input_code_domA_tmp = np.array(code_domA).astype(np.float32)
                    input_code_domA = input_code_domA_tmp.reshape([1, 1, 1, args.n_domains])
                    input_code_domB_tmp = np.array(code_domB).astype(np.float32)
                    input_code_domB = input_code_domB_tmp.reshape([1, 1, 1, args.n_domains])
                    
                    input_code_domA = torch.FloatTensor(input_code_domA).cuda().to(device)
                    input_code_domB = torch.FloatTensor(input_code_domB).cuda().to(device)

                    # namehd = 'Dom{}toDom{}'.format(input_domain_name, output_domain_name)

                    for val_id in valVec:
                        input_image_path = image_path.format(input_domain_name, input_domain_name, val_id)
                        # input_imageAll = nib.load(input_image_path)
                        # gt_image_path = image_path.format(output_domain_name, output_domain_name, val_id)
                        # gt_imageAll = nib.load(gt_image_path)
                        gt_name = '{}{:03d}'.format(output_domain_name, val_id)
                        gt_image = GTdic[gt_name]

                        sliceNum = int(dataInfo[val_id - 1 + 100][2])
                        sliceVec = np.arange(sliceNum)

                        teResults = np.zeros(gt_image.shape, dtype=np.int16)
                        # inputImage = np.zeros(input_imageAll.shape, dtype=np.int16)
                        # gtImage = np.zeros(input_imageAll.shape, dtype=np.int16)
                        input_name = '{}{:03d}'.format(input_domain_name, val_id)
                        input_image = GTdic[input_name]
                        sample_vol = load_test_data2(input_image, input_domain_id)
                        # sample_vol = [load_test_data(input_image_path, input_domain_id)]
                        # sample_vol = np.array(sample_vol).astype(np.float32)
                        # 逐个slice处理
                        for iSlicet in sliceVec:

                            iSlice = iSlicet + int(dataInfo[val_id - 1 + 100][3]) 
                            # print(iSlice.type)
                            # print('Processing image: id ' + str(val_id) + ' slice' + str(iSlicet))
                            sample_image = sample_vol[:, :, :, int(iSlice)]
                            sample_image = sample_image.reshape([1, args.fine_size0, args.fine_size1, 1])
                            # sample_image = torch.FloatTensor(sample_image).cuda().to(device)
                            sample_image = sample_image.permute(0, 3, 1, 2)
                            # if epoch == epochVec[0]:
                            #     gt_image = gt_imageAll.get_data()[:, :, int(iSlice)].astype('int16')
                            #     gtImage[:, :, int(iSlice)] = np.array(gt_image).astype('int16')

                            #     input_image = input_imageAll.get_data()[:, :, int(iSlice)].astype('int16')
                            #     inputImage[:, :, int(iSlice)] = np.array(input_image).astype('int16')

                            _, fake_img, _, _, _, _, _, _, _, _,_,_,_,_ = netGAN(input_code_domA, input_code_domB, sample_image, sample_image)

                            temp = (fake_img + 1.) / 2. * output_range
                            temp = temp.cpu().detach()
                            teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
                        # nibabel保存新的nii文件
                        # head_output = gt_imageAll.get_header() # 获取头文件
                        # affine_output = gt_imageAll.affine # 获取仿射矩阵（物理坐标相关）

                        # saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
                        # nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, val_id, epoch))

                        # 计算指标
                        gt = gt_image.get_fdata().astype('single')
                        # result = saveResults.get_fdata().astype('single')
                        img1 = normalize(gt, output_domain_id)
                        img2 = normalize(teResults, output_domain_id)
                        # ssimlist.append(np.mean(ssim(img1[:,:,:], img2[:,:,:])))
                        maelist.append(np.mean(np.mean(np.abs(img1[:,:,:] - img2[:,:,:]))))
                        psnrlist.append(np.mean(psnr2(img1[:,:,:], img2[:,:,:])))

                        # if epoch == epochVec[0]:
                        #     gtResults = nib.Nifti1Image(gtImage, affine_output, head_output)
                        #     gt_path = os.path.join(valid_dir, '{}'.format(os.path.basename(gt_image_path)))
                        #     nib.save(gtResults, gt_path)

                        #     inputResults = nib.Nifti1Image(inputImage, affine_output, head_output)
                        #     input_path = os.path.join(valid_dir, '{}'.format(os.path.basename(input_image_path)))
                        #     nib.save(inputResults, input_path)
            print("SSIM: {:.4f}, MAE: {:.4f}, PSNR: {:.4f}".format(np.mean(ssimlist),np.mean(maelist),np.mean(psnrlist)))


# def valid():
#     """valid cyclegan"""
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     image_path = r'../BraTS_new/VolumeData/Valid/Valid{}/Dom{}{:03d}.nii.gz'

#     valNum = 5
#     valVec = np.arange(valNum) + 1 # [1 2 3 4 5]

#     epochVec = np.arange(14,61) * 2 + 1  # [31 33 35 ... 121]

#     dataInfoFile = open(r'../BraTS_new/dataInfo.txt', 'r')
#     sourceInLines = dataInfoFile.readlines()
#     dataInfoFile.close()
#     dataInfo = []
#     for line in sourceInLines:
#         temp1 = line.strip('\n')
#         temp2 = temp1.split(' ')
#         dataInfo.append(temp2)

#     dom_name = ['A', 'B', 'C', 'D']
#     inputDomVec = np.arange(len(dom_name))
#     start_time = time.time()
#     for epoch in epochVec:
#         # 每隔2个epoch测一下
#         valid_dir = './valid'
#         if not os.path.exists(valid_dir):
#             os.makedirs(valid_dir)

#         valid_checkpoint = './checkpoint'

#         print(" [*] Reading checkpoint...")
#         model_dir = 'EDgan_state_' + str(epoch) + '.pth'
#         checkpoint_dir = os.path.join(valid_checkpoint, model_dir)
#         netGAN = ED_gan(args).to(device)
#         netGAN.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))
#         netGAN.eval()
#         for input_domain_id in inputDomVec:
#             # id依次取0，1，2，3
#             input_domain_name = dom_name[input_domain_id] # 依次取A,B,C,D
#             output_domain_list = np.delete(np.arange(len(dom_name)), input_domain_id)

#             for output_domain_id in output_domain_list:
#                 output_domain_name = dom_name[output_domain_id]
#                 if output_domain_id == 0:
#                     output_range = 3000
#                 elif output_domain_id == 1:
#                     output_range = 5000
#                 elif output_domain_id == 2:
#                     output_range = 6000
#                 else:
#                     output_range = 7000

#                 # input_code
#                 code_domA = prod_input_code(args.n_domains, input_domain_id)
#                 code_domB = prod_input_code(args.n_domains, output_domain_id)
#                 input_code_domA_tmp = np.array(code_domA).astype(np.float32)
#                 input_code_domA = input_code_domA_tmp.reshape([1, 1, 1, args.n_domains])
#                 input_code_domB_tmp = np.array(code_domB).astype(np.float32)
#                 input_code_domB = input_code_domB_tmp.reshape([1, 1, 1, args.n_domains])
                
#                 input_code_domA = torch.FloatTensor(input_code_domA).cuda().to(device)
#                 input_code_domB = torch.FloatTensor(input_code_domB).cuda().to(device)

#                 namehd = 'Dom{}toDom{}'.format(input_domain_name, output_domain_name)

#                 for val_id in valVec:
#                     input_image_path = image_path.format(input_domain_name, input_domain_name, val_id)
#                     input_imageAll = nib.load(input_image_path)
#                     gt_image_path = image_path.format(output_domain_name, output_domain_name, val_id)
#                     gt_imageAll = nib.load(gt_image_path)

#                     sliceNum = int(dataInfo[val_id - 1 + 100][2])
#                     sliceVec = np.arange(sliceNum)

#                     teResults = np.zeros(input_imageAll.shape, dtype=np.int16)
#                     inputImage = np.zeros(input_imageAll.shape, dtype=np.int16)
#                     gtImage = np.zeros(input_imageAll.shape, dtype=np.int16)

#                     sample_vol = [load_test_data(input_image_path, input_domain_id)]
#                     sample_vol = np.array(sample_vol).astype(np.float32)
#                     # 逐个slice处理
#                     for iSlicet in sliceVec:

#                         iSlice = iSlicet + int(dataInfo[val_id - 1 + 100][3]) 
#                         print('Processing image: id ' + str(val_id) + ' slice' + str(iSlicet))

#                         sample_image = sample_vol[:, :, :, int(iSlice)]
#                         sample_image = sample_image.reshape([1, args.fine_size0, args.fine_size1, 1])
#                         sample_image = torch.FloatTensor(sample_image).cuda().to(device)
#                         sample_image = sample_image.permute(0, 3, 1, 2)
#                         if epoch == epochVec[0]:
#                             gt_image = gt_imageAll.get_data()[:, :, int(iSlice)].astype('int16')
#                             gtImage[:, :, int(iSlice)] = np.array(gt_image).astype('int16')

#                             input_image = input_imageAll.get_data()[:, :, int(iSlice)].astype('int16')
#                             inputImage[:, :, int(iSlice)] = np.array(input_image).astype('int16')

#                         _, fake_img, _, _, _, _, _, _, _, _,_,_,_,_  = netGAN(input_code_domA, input_code_domB, sample_image, sample_image)

#                         temp = (fake_img + 1.) / 2. * output_range
#                         temp = temp.cpu().detach()
#                         teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
#                     # nibabel保存新的nii文件
#                     head_output = input_imageAll.get_header() # 获取头文件
#                     affine_output = input_imageAll.affine # 获取仿射矩阵（物理坐标相关）

#                     saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
#                     nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, val_id, epoch))

#                     if epoch == epochVec[0]:
#                         gtResults = nib.Nifti1Image(gtImage, affine_output, head_output)
#                         gt_path = os.path.join(valid_dir, '{}'.format(os.path.basename(gt_image_path)))
#                         nib.save(gtResults, gt_path)

#                         inputResults = nib.Nifti1Image(inputImage, affine_output, head_output)
#                         input_path = os.path.join(valid_dir, '{}'.format(os.path.basename(input_image_path)))
#                         nib.save(inputResults, input_path)
#                 print(input_domain_name+'****'+'epoch: [%2d], time: %4.4f' % (epoch, (time.time() - start_time)/3600))


# def valid():
#     """valid cyclegan"""
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     valid_data = Valid_Dataset()
#     valid_loader = DataLoader(valid_data,batch_size=1, num_workers=8, shuffle=False, pin_memory=True)

#     datakey = ['A','B','C','D']

#     epochVec = np.arange(14,61) * 2 + 1  # [31 33 35 ... 121]

#     dataInfoFile = open(r'../BraTS_new/dataInfo.txt', 'r')
#     sourceInLines = dataInfoFile.readlines()
#     dataInfoFile.close()
#     dataInfo = []
#     for line in sourceInLines:
#         temp1 = line.strip('\n')
#         temp2 = temp1.split(' ')
#         dataInfo.append(temp2)

#     dom_name = ['A', 'B', 'C', 'D']
#     inputDomVec = np.arange(len(dom_name))
#     start_time = time.time()
#     for epoch in epochVec:
#         # 每隔2个epoch测一下
#         valid_dir = './valid'
#         if not os.path.exists(valid_dir):
#             os.makedirs(valid_dir)

#         valid_checkpoint = './checkpoint'
#         print(" [*] Reading checkpoint...")
#         model_dir = 'EDgan_state_' + str(epoch) + '.pth'
#         checkpoint_dir = os.path.join(valid_checkpoint, model_dir)
#         netGAN = ED_gan(args).to(device)
#         netGAN.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))
#         netGAN.eval()
#         for input_domain_id in inputDomVec:
#             # id依次取0，1，2，3
#             input_domain_name = dom_name[input_domain_id] # 依次取A,B,C,D
#             output_domain_list = np.delete(np.arange(len(dom_name)), input_domain_id)
#             # 获取nii头文件、仿射矩阵、shape等信息
#             shapeList, headList, affineList = get_nii_info(input_domain_name)
#             for output_domain_id in output_domain_list:
#                 output_domain_name = dom_name[output_domain_id]
#                 if output_domain_id == 0:
#                     output_range = 3000
#                 elif output_domain_id == 1:
#                     output_range = 5000
#                 elif output_domain_id == 2:
#                     output_range = 6000
#                 else:
#                     output_range = 7000

#                 # input_code
#                 code_domA = prod_input_code(args.n_domains, input_domain_id)
#                 code_domB = prod_input_code(args.n_domains, output_domain_id)
#                 input_code_domA_tmp = np.array(code_domA).astype(np.float32)
#                 input_code_domA = input_code_domA_tmp.reshape([args.batch_size, 1, 1, args.n_domains])
#                 input_code_domB_tmp = np.array(code_domB).astype(np.float32)
#                 input_code_domB = input_code_domB_tmp.reshape([args.batch_size, 1, 1, args.n_domains])
                
#                 input_code_domA = torch.FloatTensor(input_code_domA).cuda().to(device)
#                 input_code_domB = torch.FloatTensor(input_code_domB).cuda().to(device)

#                 namehd = 'Dom{}toDom{}'.format(input_domain_name, output_domain_name)
#                 for ii, data in enumerate(valid_loader):
#                     input_data = data[datakey[input_domain_id]]
#                     # 遍历指定模态的slice数据
#                     sample_image = input_data.reshape([1, args.fine_size0, args.fine_size1, 1])
#                     sample_image = torch.FloatTensor(sample_image).cuda().to(device)
#                     sample_image = sample_image.permute(0, 3, 1, 2)

#                     _, fake_img, _, _, _, _, _, _, _, _,_,_,_,_  = netGAN(input_code_domA, input_code_domB, sample_image, sample_image)
#                     temp = (fake_img + 1.) / 2. * output_range
#                     temp = temp.cpu().detach()
#                     if ii <= 141:
#                         # valid 1
#                         iSlice = ii + int(dataInfo[1 - 1 + 100][3]) 
#                         input_shape = shapeList[0]
#                         teResults = np.zeros(input_shape, dtype=np.int16)
#                         teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
#                         # nibabel保存新的nii文件
#                         head_output = headList[0] # 获取头文件
#                         affine_output = affineList[0] # 获取仿射矩阵（物理坐标相关）
#                         saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
#                         nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, 1, epoch))
#                     elif ii > 141 and ii <= 285:
#                         # valid 2
#                         iSlice = ii-142 + int(dataInfo[2 - 1 + 100][3]) 
#                         input_shape = shapeList[1]
#                         teResults = np.zeros(input_shape, dtype=np.int16)
#                         teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
#                         # nibabel保存新的nii文件
#                         head_output = headList[1] # 获取头文件
#                         affine_output = affineList[1] # 获取仿射矩阵（物理坐标相关）
#                         saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
#                         nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, 2, epoch))
#                     elif ii > 285 and ii <= 428:
#                         # valid 3
#                         iSlice = ii-286 + int(dataInfo[3 - 1 + 100][3]) 
#                         input_shape = shapeList[2]
#                         teResults = np.zeros(input_shape, dtype=np.int16)
#                         teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
#                         # nibabel保存新的nii文件
#                         head_output = headList[2] # 获取头文件
#                         affine_output = affineList[2] # 获取仿射矩阵（物理坐标相关）
#                         saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
#                         nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, 3, epoch))
#                     elif ii > 428 and ii <= 569:
#                         # valid 4
#                         iSlice = ii-429 + int(dataInfo[4 - 1 + 100][3]) 
#                         input_shape = shapeList[3]
#                         teResults = np.zeros(input_shape, dtype=np.int16)
#                         teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
#                         # nibabel保存新的nii文件
#                         head_output = headList[3] # 获取头文件
#                         affine_output = affineList[3] # 获取仿射矩阵（物理坐标相关）
#                         saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
#                         nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, 4, epoch))
#                     else:
#                         # valid 5
#                         iSlice = ii-570 + int(dataInfo[5 - 1 + 100][3]) 
#                         input_shape = shapeList[4]
#                         teResults = np.zeros(input_shape, dtype=np.int16)
#                         teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0, args.fine_size1])
#                         # nibabel保存新的nii文件
#                         head_output = headList[4] # 获取头文件
#                         affine_output = affineList[4] # 获取仿射矩阵（物理坐标相关）
#                         saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
#                         nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir, namehd, 5, epoch))
#                         print(input_domain_name+'****'+'epoch: [%2d], time: %4.4f' % (epoch, (time.time() - start_time)/3600))
                        

                    
def test():
    """Test cyclegan"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_path = r'../BraTS_new/VolumeData/Test/Test{}/Dom{}{:03d}.nii.gz'

    epoch = 181
    test_checkpoint = './checkpoint'
    model_dir = 'EDgan_state_' + str(epoch) + '.pth'
    checkpoint_dir = os.path.join(test_checkpoint, model_dir)
    # checkpoint_dir = model_dir
    netGAN = ED_gan(args).to(device)
    netGAN.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))
    netGAN.eval()

    test_base = './test'
    if not os.path.exists(test_base):
        os.makedirs(test_base)
    test_dir = '{}/epoch_{}'.format(test_base, epoch)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    tedataSize = 45
    teIdVec = np.arange(tedataSize) + 1  

    dataInfoFile = open(r'../BraTS_new/dataInfo.txt', 'r')
    sourceInLines = dataInfoFile.readlines()
    dataInfoFile.close()
    dataInfo = []
    for line in sourceInLines:
        temp1 = line.strip('\n')
        temp2 = temp1.split(' ')
        dataInfo.append(temp2)

    dom_name = ['A', 'B', 'C', 'D']
    inputDomVec = np.arange(len(dom_name))

    with torch.no_grad():
        for teId in teIdVec:
            sliceNum = int(dataInfo[teId - 1 + 105][2])
            # print('sliceNum',sliceNum) 
            sliceVec = np.arange(sliceNum)

            for input_domain_id in inputDomVec:
                # id依次取0，1，2，3
                input_domain_name = dom_name[input_domain_id] # 依次取A,B,C,D
                # print('input_domain_name',input_domain_name)
                output_domain_list = np.delete(np.arange(len(dom_name)), input_domain_id)

                for output_domain_id in output_domain_list:
                    output_domain_name = dom_name[output_domain_id]
                    # print('output_domain_name',output_domain_name)
                    if output_domain_id == 0:
                        output_range = 3000
                    elif output_domain_id == 1:
                        output_range = 5000
                    elif output_domain_id == 2:
                        output_range = 6000
                    else:
                        output_range = 7000

                    # input_code
                    code_domA = prod_input_code(args.n_domains, input_domain_id)
                    code_domB = prod_input_code(args.n_domains, output_domain_id)
                    input_code_domA_tmp = np.array(code_domA).astype(np.float32)
                    input_code_domA = input_code_domA_tmp.reshape([1, 1, 1, args.n_domains])
                    input_code_domB_tmp = np.array(code_domB).astype(np.float32)
                    input_code_domB = input_code_domB_tmp.reshape([1, 1, 1, args.n_domains])
                    
                    input_code_domA = torch.FloatTensor(input_code_domA).cuda().to(device)
                    input_code_domB = torch.FloatTensor(input_code_domB).cuda().to(device)

                    namehd = 'Dom{}toDom{}'.format(input_domain_name, output_domain_name)

                    # gt_image_path = image_path.format(output_domain_name, output_domain_name, teId)
                    # gt_imageAll = nib.load(gt_image_path)

                    input_image_path = image_path.format(input_domain_name, input_domain_name, teId)
                    input_imageAll = nib.load(input_image_path)

                    # 将数据处理到-1到1之间然后读入,有 img[img > 1.] = 1. 截断操作
                    sample_vol = load_test_data(input_image_path, input_domain_id)
                    # sample_vol = [load_test_data(input_image_path, input_domain_id)]
                    # sample_vol = np.array(sample_vol).astype(np.float32)

                    teResults = np.zeros(input_imageAll.shape, dtype=np.int16)
                    inputImage = np.zeros(input_imageAll.shape, dtype=np.int16)
                    # gtImage = np.zeros(input_imageAll.shape, dtype=np.int16)

                    # 逐个slice处理
                    for iSlicet in sliceVec:
                        iSlice = iSlicet + int(dataInfo[teId - 1 + 105][3]) 
                        print('Processing image: id ' + str(teId) + ' slice' + str(iSlicet))
                        # print('sample_vol',sample_vol.shape) # (1, 240, 240, 160)
                        sample_image = sample_vol[:, :, :, int(iSlice)]
                        # print('sample_image',np.max(sample_image))
                        sample_image = sample_image.reshape([1, args.fine_size0, args.fine_size1, 1])
                        # sample_image = torch.FloatTensor(sample_image).cuda().to(device)
                        sample_image = sample_image.permute(0, 3, 1, 2)

                        # gt_image = gt_imageAll.get_data()[:, :, int(iSlice)].astype('int16')
                        # gtImage[:, :, int(iSlice)] = np.array(gt_image).astype('int16')
                       
                        input_image = input_imageAll.get_data()[:, :, int(iSlice)].astype('int16')
                        inputImage[:, :, int(iSlice)] = np.array(input_image).astype('int16')
                        # inputImage[int(iSlice), :, :] = np.array(input_image).astype('int16')

                        _, fake_img, _, _, _, _, _, _, _, _,_,_,_,_ = netGAN(input_code_domA, input_code_domB, sample_image, sample_image)
                        # fake = fake_img.cpu().detach().numpy()
                        # print('fake_img_max',np.max(fake)) # 最大值都偏小：-0.14, -0.07, 0.06... 
                        # print('fake_img_min',np.min(fake)) # 最小值均为-1
                        temp = (fake_img + 1.) / 2. * output_range
                        temp = temp.cpu().detach().numpy()
                        teResults[:, :, int(iSlice)] = np.array(temp).astype('int16').reshape([args.fine_size0,args.fine_size1])

                    # nibabel保存新的nii文件
                    head_output = input_imageAll.get_header() # 获取头文件
                    affine_output = input_imageAll.affine # 获取仿射矩阵（物理坐标相关）

                    saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
                    nib.save(saveResults, '{}/{}_{:0>2d}.nii.gz'.format(test_dir, namehd, teId))

                    # gtResults = nib.Nifti1Image(gtImage, affine_output, head_output)
                    # gt_path = os.path.join(test_dir, '{}'.format(os.path.basename(gt_image_path)))
                    # nib.save(gtResults, gt_path)

                    inputResults = nib.Nifti1Image(inputImage, affine_output, head_output)
                    input_path = os.path.join(test_dir, '{}'.format(os.path.basename(input_image_path)))
                    nib.save(inputResults, input_path)

if __name__ == '__main__':
    if args.phase == 'train':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ED_gan = ED_gan(args).to(device)
        # ED_gan._initialize_weights()
        Discriminator = Discriminator(args).to(device)
        # Discriminator._initialize_weights()

        # lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)
        optimizerED = optim.Adam(ED_gan.parameters(), lr=args.lr, betas=(0.5,0.999))
        optimizerD = optim.Adam(Discriminator.parameters(), lr=args.lr, betas=(0.5,0.999))

        schedulerED = optim.lr_scheduler.MultiStepLR(optimizerED, args.milestone, gamma=0.5)
        schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, args.milestone, gamma=0.5)
        total = sum([param.nelement() for param in ED_gan.parameters()])
        print('Number of ED_gan_param',total)
        train(ED_gan,Discriminator,optimizerED,schedulerED,optimizerD,schedulerD)
    elif args.phase == 'valid':
        valid()
    else:
        test()




