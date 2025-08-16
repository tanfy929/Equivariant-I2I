# Equivariant-I2I
Code for " Image-to-Image Translation Framework Embedded with Rotation Symmetry Priors".

![](https://github.com/tanfy929/Equivariant-I2I/blob/main/TLEQ-I2I.png)

In this work, we have explored the transformation symmetry prior in image datasets, and focus on constructing I2I frameworks that preserve this domain-invariant feature. Specifically, we have introduced two key components to advance I2I methods: 1. rotation equivariant I2I framewok constructed with existing EQ-CNNs, which preserves strict rotation symmetry across the entire network flow, and 2. a new proposed transformation learnable equivariant convolution named TL-Conv,  which can adaptively learn more symmetric transformations in the dataset and ensure the corresponding equivariance in the I2I process. 

**Usage**
> 
> - **`Unpaired_I2I`**: Taking CycleGAN as an example, our implementation includes traditional CNN network, equivariant network (EQ), and transformation learnable equivariant networks (TLEQ), complete with both source codes and model weights. For execution commands and datasets, please refer to [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master).
> - **`MRI_translation`**: Codes for multi-contrast MR Image translation.
> - **`Image_Restoration`**: Codes for extended experiments on image restoration (single image rain removal, image denoising, and image super-resolution).  
**For DeRain**:  
--train--  
```python main.py --scale 2 --epochs 100 --batch_size 16 --patch_size 64 --n_threads 0 --stage 17 --lr_decay 25 --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 --loss 1*MSE --device 2```  
--test--  
```python main.py --data_test RainHeavyTest  --ext img --scale 2  --data_range 1-200/1-100 --pre_train ../experiment/RCDNet_syn_newfconv/model/model_best.pt --model newfconv_rcdnet --test_only --save_results --save RCDNet_syn_newfconv --device 2```  
**For DeNoising**:    
--train--  
```python main.py --model edsr_newbconv --scale 1 --save edsr_newbconv --n_resblocks 16 --n_feats 32 --res_scale 0.1 --epoch 150 --decay 100 --patch_size 48 --lr 0.00013 --device 6```  
--test--  
```python main_test.py --model edsr_newbconv --scale 1 --pre_train ../experiment/edsr_newbconv/model/model_best.pt --save ../experiment/edsr_newbconv/ --kernel_size 5 --n_resblocks 16 --n_feats 32 --res_scale 0.1 --device 6```  
**For SR**:  
--train--  
```python main_train.py --model edsr_newbconv --scale 2 --save edsr_fcnn_retrain_valid --n_resblocks 16 --n_feats 32 --res_scale 0.1 --epoch 150 --decay 100 --patch_size 96 --kernel_size 5 --tranNum 8 --device 4```  
--test--  
```python main.py --model edsr_newbconv --scale 2 --pre_train ../experiment/edsr_fcnn_retrain_valid/model/model_best.pt --save ../experiment/edsr_fcnn_retrain_valid/ --n_resblocks 16 --n_feats 32 --res_scale 0.1 --kernel_size 5 --tranNum 8 --test_only True --device 4```

We will continue to update and refine both the codes and corresponding usage instructions.

