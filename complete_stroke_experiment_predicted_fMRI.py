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

sys.path.append('/home/jonathak/VisualEncoder/Analysis/Brain_maps')
from NIPS_utils import get_hemisphere_indices

sys.path.append('/home/jonathak/VisualEncoder/Voxels_Prediction')
from predict_voxels_jonathan import get_images_for_prediction

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
'original_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/original_voxels.pt',
'left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/half_mask_side-left_centerX-None_centerY-None_coarse_blur_voxels.pt',
'right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/half_mask_side-right_centerX-None_centerY-None_coarse_blur_voxels.pt',
'top_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/half_mask_side-top_centerX-None_centerY-None_coarse_blur_voxels.pt',
'bottom_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/half_mask_side-bottom_centerX-None_centerY-None_coarse_blur_voxels.pt',
'top_right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/quadrant_mask_quad-top_right_coarse_blur_voxels.pt',
'top_left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/quadrant_mask_quad-top_left_coarse_blur_voxels.pt',
'bottom_right_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/quadrant_mask_quad-bottom_right_coarse_blur_voxels.pt',
'bottom_left_mask_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/quadrant_mask_quad-bottom_left_coarse_blur_voxels.pt',
'gaussian_inner_25_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/gaussian_inner_mask_x0.5_y0.5_w25_coarse_blur_voxels.pt',   
'gaussian_inner_50_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/gaussian_inner_mask_x0.5_y0.5_w50_coarse_blur_voxels.pt',
'gaussian_inner_75_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/gaussian_inner_mask_x0.5_y0.5_w75_coarse_blur_voxels.pt',
'gaussian_outer_25_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/gaussian_outer_mask_x0.5_y0.5_w25_coarse_blur_voxels.pt',   
'gaussian_outer_50_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/gaussian_outer_mask_x0.5_y0.5_w50_coarse_blur_voxels.pt',
'gaussian_outer_75_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/gaussian_outer_mask_x0.5_y0.5_w75_coarse_blur_voxels.pt',
}

voxel_paths = voxel_paths_blur_fill

# Loading models

encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model = torch.load('/home/jonathak/VisualEncoder/Voxels_Prediction/model_ch128.pth').eval().cuda()

# Defining subject
stroke_sub = 1

# Getting images

images = get_images_for_prediction(image_type='excluded', subjects=[stroke_sub])
images = images.permute(0, 2, 3, 1)


# Getting original predicted fMRI

original_predicted_fMRI = torch.load(voxel_paths['original_voxels_path']).cuda()

# Getting voxel indices

lh_start, lh_end = get_hemisphere_indices(stroke_sub, 'lh')
rh_start, rh_end = get_hemisphere_indices(stroke_sub, 'rh')
inds = np.arange(lh_start, rh_end)

NC = np.load("/home/romanb/data/datasets/NVD/tutorial_data/noise_ceiling/noise_ceiling.npy")

inds_nc = np.where(NC[inds]>0.5)[0]
inds_nc_torch = torch.from_numpy(inds_nc)

# Image processing functions

mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1,3,1,1).cuda()
std = torch.tensor((0.229, 0.224, 0.225)).reshape(1,3,1,1).cuda()

def trans_imgs(imgs):
    imgs = imgs/255.0
    imgs =  imgs.permute(2,0,1).float()
    return imgs

def save_as_png(array, save_path):
    
    # If the image is a troch tensor, convert it to a numpy array
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    
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
num_iter_no_stroke = 4001 # max iterations
num_iter_stroke = 2001 # max iterations
burnin_iter = 7000 # burn-in iteration for SGLD
weight_decay = 5e-8
mse = torch.nn.MSELoss().type(dtype) # loss

save_throughout = True
n_saves = 5
save_every = num_iter_stroke // n_saves

# Defining the desired masks

all_masks = list(voxel_paths.keys())[1:]
working_masks = ['top_right_mask_voxels_path', 'bottom_right_mask_voxels_path', 'bottom_left_mask_voxels_path', 'top_left_mask_voxels_path', 'gaussian_outer_30_voxels_path']
test_masks = ['top_right_mask_voxels_path', 'bottom_left_mask_voxels_path', 'gaussian_outer_50_voxels_path', 'gaussian_inner_75_voxels_path']
masks = test_masks

# Defining thresholds

starting_threshold = 1.8
stroke_indices_counts = [600,1000]

# Creating random image indices

images_indices = np.sort(np.array([0, 1, 5, 6, 9, 12, 13, 14, 22, 28, 27, 66, 119, 39, 109, 56, 44, 69]))

# Results path

root_save_path = '/home/jonathak/VisualEncoder/DIP_decoder/stroke_experiment_results_predicted_fMRI/debugging'

# The images loop

for img_idx in images_indices:
    
    save_path = f'{root_save_path}/img_{img_idx}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Saving the original image
    save_as_png(images[img_idx], f'{save_path}/img_{img_idx}_original_image.png')
    
    # Getting the stroke fill value
    min_val = original_predicted_fMRI[img_idx,inds].min()
    
    # Original target voxel maps
    original_target_all = original_predicted_fMRI[img_idx,inds].unsqueeze(0).float().to(device)
    original_target_nc = original_predicted_fMRI[img_idx,inds_nc].unsqueeze(0).float().to(device)

    # Step 1: Fitting the network to the image
    # =====================================
    in_img = trans_imgs(images[img_idx])

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
        
        # Saving intermediate image
        # if save_throughout and i % save_every == 0 and i != 0:
        #     save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_fitted_image_iter_{i}.png')
        
        i += 1
        return loss

    ## Optimizing 
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    for j in range(num_iter_no_stroke):
        optimizer.zero_grad()
        closure()
        optimizer.step()
        
    print(f'Finished fitting on image for image {img_idx}')
    
    # Saving the fitted image
    save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_fitted_image_final.png')
    
    # Saving the fitted network
    torch.save({
        'net_state': net.state_dict(),
        'out_avg': out_avg
    }, os.path.join(save_path, 'dip_on_original_image.pth'))

    torch.cuda.empty_cache()
    
    
    # Step 2: Decoding normal fMRI voxel map
    # ================================
    
    net = get_net(input_depth, 'skip', pad,
    skip_n33d=128, 
    skip_n33u=128,
    skip_n11=2,
    num_scales=3,
    upsample_mode='bilinear').type(dtype)

    checkpoint = torch.load(os.path.join(save_path, 'dip_on_original_image.pth'))
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
        
        # # Saving intermediate image
        # if save_throughout and i % save_every == 0 and i != 0:
        #     save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_normal_fMRI_iter_{i}.png')

        i += 1
        return loss

    ## Optimizing 

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    for j in range(num_iter_no_stroke):
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
            # =====================================
            
            net = get_net(input_depth, 'skip', pad,
            skip_n33d=128, 
            skip_n33u=128,
            skip_n11=2,
            num_scales=3,
            upsample_mode='bilinear').type(dtype)

            # Step 3.1: Starting from original image
            # =====================================
            
            checkpoint = torch.load(os.path.join(save_path, 'dip_on_original_image.pth'))
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
                if save_throughout and i % save_every == 0 and i != 0:
                    save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_stroke_{mask_name}_size_{stroke_size}_iter_{i}_start_original.png')
                i += 1
                return loss

            ## Optimizing 
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            for j in range(num_iter_stroke):
                optimizer.zero_grad()
                closure()
                optimizer.step()

            torch.cuda.empty_cache()
            
            # Saving the stroke fMRI final image
            save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_stroke_{mask_name}_size_{stroke_size}_start_original_final.png')
            
            # Step 3.2: Starting from fMRI image
            # =====================================
            
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
                if save_throughout and i % save_every == 0 and i != 0:
                    save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_stroke_{mask_name}_size_{stroke_size}_iter_{i}_start_fMRI.png')
                
                i += 1
                return loss

            ## Optimizing 
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            for j in range(num_iter_stroke):
                optimizer.zero_grad()
                closure()
                optimizer.step()
                
            print(f'Finished decoding stroke fMRI for image {img_idx}, {mask_name}, stroke size {stroke_size}')
        
            torch.cuda.empty_cache()
            # Saving the stroke fMRI final image
            save_as_png(out_avg_np, f'{save_path}/img_{img_idx}_stroke_{mask_name}_size_{stroke_size}_start_fMRI_final.png')    

print('Finished all experiments!')