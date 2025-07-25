# Roman's imports

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('/home/jonathak/VisualEncoder/DIP_decoder/GP-DIP/')
import numpy as np
from models import *
import torch
import torch.optim
import torch.nn.functional as F
import random
import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
import _pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sys.path.append('/home/romanb/PycharmProjects/BrainVisualReconst/')
    
# My imports and parameters

device = torch.device('cuda')
from create_stroke_fMRI import StrokeVoxelPredictor
stroke_predictor = StrokeVoxelPredictor()

voxel_paths_mean_fill = {
    'original_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/original_voxels.pt',
    'left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/half_mask_side-left_centerX-None_centerY-None_mask_mean_voxels.pt',
    'right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/half_mask_side-right_centerX-None_centerY-None_mask_mean_voxels.pt',
    'top_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/half_mask_side-top_centerX-None_centerY-None_mask_mean_voxels.pt',
    'bottom_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/half_mask_side-bottom_centerX-None_centerY-None_mask_mean_voxels.pt',
    'top_right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/quadrant_mask_quad-top_right_mask_mean_voxels.pt',
    'top_left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/quadrant_mask_quad-top_left_mask_mean_voxels.pt',
    'bottom_right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/quadrant_mask_quad-bottom_right_mask_mean_voxels.pt',
    'bottom_left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/quadrant_mask_quad-bottom_left_mask_mean_voxels.pt',
    'gaussian_inner_50_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/gaussian_inner_mask_x0.5_y0.5_w50_mask_mean_voxels.pt',
    'gaussian_outer_30_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/gaussian_outer_mask_x0.5_y0.5_w30_mask_mean_voxels.pt'
}
voxel_paths_blur_fill = {
'gaussian_inner_25_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/gaussian_inner_mask_x0.5_y0.5_w25_coarse_blur_voxels.pt',
'gaussian_inner_50_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/gaussian_inner_mask_x0.5_y0.5_w50_coarse_blur_voxels.pt',
'gaussian_inner_75_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/gaussian_inner_mask_x0.5_y0.5_w75_coarse_blur_voxels.pt',
'gaussian_outer_25_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/gaussian_outer_mask_x0.5_y0.5_w25_coarse_blur_voxels.pt',
'gaussian_outer_50_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/gaussian_outer_mask_x0.5_y0.5_w50_coarse_blur_voxels.pt',
'gaussian_outer_75_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/gaussian_outer_mask_x0.5_y0.5_w75_coarse_blur_voxels.pt',
'bottom_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/half_mask_side-bottom_centerX-None_centerY-None_coarse_blur_voxels.pt',
'left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/half_mask_side-left_centerX-None_centerY-None_coarse_blur_voxels.pt',
'right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/half_mask_side-right_centerX-None_centerY-None_coarse_blur_voxels.pt',
'top_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/half_mask_side-top_centerX-None_centerY-None_coarse_blur_voxels.pt',
'original_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/original_voxels.pt',
'bottom_left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/quadrant_mask_quad-bottom_left_coarse_blur_voxels.pt',
'bottom_right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/quadrant_mask_quad-bottom_right_coarse_blur_voxels.pt',
'top_left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/quadrant_mask_quad-top_left_coarse_blur_voxels.pt',
'top_right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/coarse_blur_all_masks_excluded_imgs/excluded_coarse_blur/voxels/quadrant_mask_quad-top_right_coarse_blur_voxels.pt'    
}
voxel_paths = voxel_paths_blur_fill

# Loading models

encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model = torch.load('/home/jonathak/VisualEncoder/Voxels_Prediction/model_ch128.pth').eval().cuda()

# Loading data

val_sub1 = True

from sklearn.model_selection import train_test_split

data_dir = '/net/mraid11/export/data/navvew/algonauts_2023_challenge_data'
num_voxels_subjects = np.load(data_dir + '/num_voxels_all_subjects.npy')

num_voxels_subjects = num_voxels_subjects.sum(1).astype(int)

data_dir = "/home/romanb/data/datasets/NVD/tutorial_data/subjects/subsuper/"
imgs = np.load(data_dir+"all_images.npy")

lvc_bitmap = np.load(data_dir+"lvc_bitmap.npy")
fmri_files = np.load(data_dir+"fmri.npz")

type_sample = fmri_files["type_sample"]
single_sub_fmri = fmri_files["single_sub_fmri"]
single_sub = fmri_files["single_sub"].astype(int)
multi_sub_fmri = fmri_files["multi_sub_fmri"]

imgs_single = imgs[type_sample==1]
imgs_multi = imgs[type_sample==2]

single_sub_fmri_train, single_sub_fmri_val, single_sub_train, single_sub_val, imgs_single_train, imgs_single_val = train_test_split(single_sub_fmri,single_sub, imgs_single , test_size=0.1, random_state = 10)

single_sub_fmri_train  = single_sub_fmri_train
single_sub_train       = single_sub_train
imgs_single_train    = imgs_single_train

if(val_sub1):
    select = (single_sub_val == 0)
    single_sub_fmri_val = single_sub_fmri_val[select]
    single_sub_val      = single_sub_val[select]
    imgs_single_val   = imgs_single_val[select]

num_vox = num_voxels_subjects[0]
NC = np.load("/home/romanb/data/datasets/NVD/tutorial_data/noise_ceiling/noise_ceiling.npy")
select = NC[:num_vox]>0.5

# get all indices
inds = np.arange(num_vox)
inds_torch = torch.from_numpy(inds)

# selects voxels with snr>0.5
inds_nc = np.where(NC[:num_vox]>0.5)[0]
inds_nc_torch = torch.from_numpy(inds_nc)

# Image processing functions

mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1,3,1,1).cuda()
std = torch.tensor((0.229, 0.224, 0.225)).reshape(1,3,1,1).cuda()

def trans_imgs(imgs):
    imgs = imgs/255.0
    imgs =  imgs.transpose([2,0,1])
    imgs =  torch.from_numpy(imgs.astype(float)).float()
    return imgs

def save_as_png(array, save_path):
    # If array is (3,224,224), transpose it
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    
    # Check if values are floats between 0-1 and scale accordingly
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array * 255).astype(np.uint8)
    elif array.dtype == np.uint8:
        # Already in correct range, no scaling needed
        pass
    else:
        raise ValueError(f"Unsupported array dtype: {array.dtype}. Expected float32/64 or uint8")
    
    # Save as PNG
    plt.imsave(save_path, array)
    
# DIP parameters

INPUT = 'noise'
pad = 'reflection'
OPT_OVER = 'net' # optimize over the net parameters only
c = 1./30.
reg_noise_std = 1./30.

learning_rate = LR = 0.001
exp_weight=0.99
input_depth = 32 
roll_back = True # to prevent numerical issues
num_iter = 1501 # max iterations
burnin_iter = 7000 # burn-in iteration for SGLD
weight_decay = 5e-8
show_every =  50
mse = torch.nn.MSELoss().type(dtype) # loss
iter_save = 600

# Defining subject

stroke_sub = 1

# Defining the desired masks

all_masks = list(voxel_paths.keys())[1:]
working_masks = ['top_right_mask_voxels_path', 'bottom_right_mask_voxels_path', 'bottom_left_mask_voxels_path', 'top_left_mask_voxels_path', 'gaussian_outer_30_voxels_path']
test_masks = all_masks[:1]

masks = []

# Defining thresholds

starting_threshold = 1.8
stroke_indices_counts = [600, 1000]
# stroke_indices_counts = [1000]

# Creating random image indices

# n_images = 60
# images_indices = [22]
# images_indices = random.sample(range(879), n_images)
# images_indices = np.load('/home/jonathak/VisualEncoder/DIP_decoder/images_indices.npy')
images_indices = [334,335,691]

# Results path

root_save_path = '/home/jonathak/VisualEncoder/DIP_decoder/stroke_experiment_results'

# The images loop

for img_idx in images_indices:
    
    save_path = f'{root_save_path}/img_{img_idx}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Saving the original image
    save_as_png(imgs_single_val[img_idx], f'{save_path}/img_{img_idx}_original_image.png')
    
    # Getting the stroke fill value
    min_val = single_sub_fmri_val[img_idx,inds].min()
    
    # Original target voxel maps
    original_target_all = torch.from_numpy(single_sub_fmri_val[img_idx,inds]).unsqueeze(0).float().to(device)
    original_target_nc = torch.from_numpy(single_sub_fmri_val[img_idx,inds_nc]).unsqueeze(0).float().to(device)

    # Step 1: Fitting the network to the image
            
    in_img = trans_imgs(imgs_single_val[img_idx])

    net = get_net(input_depth, 'skip', pad,
                skip_n33d=128, 
                skip_n33u=128,
                skip_n11=2,
                num_scales=3,
                upsample_mode='bilinear').type(dtype)

    ## Optimize
    net_input = get_noise(input_depth, INPUT, (224, 224),var=0.1).type(dtype).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    
    i = 0

    out = net(net_input)
    rec_img_np = out.detach().cpu().numpy()[0]

    def closure():

        global i, out_avg, net_input, out_avg_np
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        out = net(net_input)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        loss = F.mse_loss(out, in_img.cuda())

        loss.backward()

        out_avg_np = out_avg.detach().cpu().numpy()[0]

        i += 1
        return loss

    ## Optimizing 
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    for j in range(num_iter):
        optimizer.zero_grad()
        closure()
        optimizer.step()
        
    print(f'Finished fitting on image for image {img_idx}')
    
    # Saving the fitted image
    save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_fitted_image.png')
    
    # Saving the fitted network
    torch.save({
        'net_state': net.state_dict(),
        'out_avg': out_avg
    }, os.path.join(root_save_path, 'dip_on_original_image.pth'))

    torch.cuda.empty_cache()

    # Step 2: Decoding normal fMRI voxel map
    
    net = get_net(input_depth, 'skip', pad,
    skip_n33d=128, 
    skip_n33u=128,
    skip_n11=2,
    num_scales=3,
    upsample_mode='bilinear').type(dtype)

    checkpoint = torch.load(os.path.join(root_save_path, 'dip_on_original_image.pth'))
    net.load_state_dict(checkpoint['net_state'])

    out_avg = checkpoint['out_avg']

    reg_noise_std = 1./30.
    i = 0 

    def closure():

        global i, out_avg, net_input, out_avg_np
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        out = net(net_input)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
        
        enc_in = (out-mean)/std
        vox_pred = model(enc_in, inds_nc_torch.unsqueeze(0).cuda())
        
        loss = F.mse_loss(vox_pred, original_target_nc)

        loss.backward()

        out_avg_np = out_avg.detach().cpu().numpy()[0]
        
        # Saving intermediate image
        # if i == iter_save:
        #     np.save(f'{save_path}/img_{img_idx}_normal_fMRI_iter_{iter_save}.npy', out_avg_np)

        i += 1
        return loss

    ## Optimizing 

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    for j in range(num_iter):
        optimizer.zero_grad()
        closure()
        optimizer.step()
        
    print(f'Finished decoding original fMRI for image {img_idx}')
    
    # MAYBE REMOVE- start from fMRI 
    torch.save({
        'net_state': net.state_dict(),
        'out_avg': out_avg
    }, os.path.join(save_path, 'dip_on_fMRI_image.pth'))
    
    torch.cuda.empty_cache()
    
    # Saving the normal fMRI final image
    save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_normal_fMRI_final.png')
    
    # The masks loop    
    for mask in masks:
        
        masked_voxels_path = voxel_paths[mask]
        mask_name = mask.replace("_voxels_path", "")
        
        # The stroke counts loop
        for stroke_indices_count in stroke_indices_counts:
            
            # Setting stroke size name for saving 
            if stroke_indices_count >= 950:
                stroke_size = 'large'
            else:
                stroke_size = 'small'
            
            # Setting initial threshold and counts
            current_stroke_indices_count = 0
            threshold = starting_threshold
            
            # Finding the apropriate threshold and creating target voxel maps (normal and stroked)
            while current_stroke_indices_count < stroke_indices_count:
                
                stroke_indices = stroke_predictor.create_stroked_voxels(img_idx, stroke_sub, threshold, voxel_paths['original_voxels_path'], masked_voxels_path, only_indices=True)

                stroke_target_all = original_target_all.clone().detach()
                stroke_target_all[0, stroke_indices] = stroke_target_all.min()
                stroke_target_all = stroke_target_all.float()
                stroke_target_nc = stroke_target_all[:,inds_nc]

                # Counting stroke voxels in the nc case
                current_stroke_indices_count = torch.sum((original_target_nc != min_val) & (stroke_target_nc == min_val))

                # Moving to cuda
                stroke_target_all = stroke_target_all.to(device)
                stroke_target_nc = stroke_target_nc.to(device)
                
                # Stepping up the threshold
                if threshold > 1.4:
                    threshold -= 0.08
                else:
                    threshold -= 0.05
        
            # Step 3: Decoding stroke fMRI voxel map
            
            net = get_net(input_depth, 'skip', pad,
            skip_n33d=128, 
            skip_n33u=128,
            skip_n11=2,
            num_scales=3,
            upsample_mode='bilinear').type(dtype)

            # Starting from original image
            checkpoint = torch.load(os.path.join(root_save_path, 'dip_on_original_image.pth'))
            net.load_state_dict(checkpoint['net_state'])
            out_avg = checkpoint['out_avg']

            reg_noise_std = 1./30.
            i = 0 

            def closure():

                global i, out_avg, net_input, out_avg_np
                
                if reg_noise_std > 0:
                    net_input = net_input_saved + (noise.normal_() * reg_noise_std)
                out = net(net_input)

                if out_avg is None:
                    out_avg = out.detach()
                else:
                    out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                
                enc_in = (out-mean)/std
                vox_pred = model(enc_in, inds_nc_torch.unsqueeze(0).cuda())
                
                loss = F.mse_loss(vox_pred, stroke_target_nc)#+total_variation_loss(out)+0.01*norm_6(out)#- 0.1*torch.mean(F.cosine_similarity(vox_pred, target))

                loss.backward()

                out_avg_np = out_avg.detach().cpu().numpy()[0]
                
                # Saving intermediate image
                # if i == iter_save:
                #     np.save(f'{save_path}/img_{img_idx}_stroke_fMRI_mask_{mask_name}_size_{stroke_size}_iter_{iter_save}.npy', out_avg_np)

                i += 1
                return loss

            ## Optimizing 
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            for j in range(num_iter):
                optimizer.zero_grad()
                closure()
                optimizer.step()

            torch.cuda.empty_cache()
            
            # Saving the stroke fMRI final image
            save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_stroke_{mask_name}_size_{stroke_size}_start_original.png')
            
            # Starting from fMRI image
            checkpoint = torch.load(os.path.join(save_path, 'dip_on_fMRI_image.pth'))
            net.load_state_dict(checkpoint['net_state'])
            out_avg = checkpoint['out_avg']

            reg_noise_std = 1./30.
            i = 0 

            def closure():

                global i, out_avg, net_input, out_avg_np
                
                if reg_noise_std > 0:
                    net_input = net_input_saved + (noise.normal_() * reg_noise_std)
                out = net(net_input)

                if out_avg is None:
                    out_avg = out.detach()
                else:
                    out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                
                enc_in = (out-mean)/std
                vox_pred = model(enc_in, inds_nc_torch.unsqueeze(0).cuda())
                
                loss = F.mse_loss(vox_pred, stroke_target_nc)#+total_variation_loss(out)+0.01*norm_6(out)#- 0.1*torch.mean(F.cosine_similarity(vox_pred, target))

                loss.backward()

                out_avg_np = out_avg.detach().cpu().numpy()[0]
                
                # Saving intermediate image
                # if i == iter_save:
                #     np.save(f'{save_path}/img_{img_idx}_stroke_fMRI_mask_{mask_name}_size_{stroke_size}_iter_{iter_save}.npy', out_avg_np)

                i += 1
                return loss

            ## Optimizing 
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            for j in range(num_iter):
                optimizer.zero_grad()
                closure()
                optimizer.step()
                
            print(f'Finished decoding stroke fMRI for image {img_idx}, {mask_name}, stroke size {stroke_size}')
        
            torch.cuda.empty_cache()
            # Saving the stroke fMRI final image
            save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_stroke_{mask_name}_size_{stroke_size}_start_fMRI.png')    

print('Finished all experiments!')