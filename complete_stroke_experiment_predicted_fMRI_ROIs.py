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
from NIPS_utils import get_hemisphere_indices, get_roi_indices


sys.path.append('/home/jonathak/VisualEncoder/Voxels_Prediction')
from predict_voxels_jonathan import get_images_for_prediction

device = torch.device('cuda')
from create_stroke_fMRI import StrokeVoxelPredictor
stroke_predictor = StrokeVoxelPredictor()

# This is Matanya's config file - just to play with the script
sys.path.append('/home/matanyaw/DIP_decoder/voxel_embeddings_ROIs')
import ROI_actions
from datetime import timedelta
import argparse
from datetime import datetime
import requests
from PIL import Image, PngImagePlugin
from image_montage import create_montage

def get_roi_names(subset_rois):
    # Defining the desired ROI masks

    ROIs_bodies = ['EBA', 'FBA-1', 'FBA-2', 'mTL-bodies']
    ROIs_faces = ['OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces']
    ROIs_places = ['OPA', 'PPA', 'RSC']
    ROIs_words = ['OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words']

    ROIs = ROIs_bodies + ROIs_faces + ROIs_places + ROIs_words

    if subset_rois is not None:
        # Filter ROIs based on user input
        for roi in subset_rois:
            if roi not in ROIs:
                print(f"ROI '{roi}' is not a valid ROI. Available ROIs: {ROIs}")

        ROIs = [roi for roi in subset_rois if roi in ROIs]
        if not ROIs:
            raise ValueError("No valid ROIs provided in --roi_to_process argument.")
    return ROIs


def run_experiment(args_config):
    """ This function runs the complete stroke experiment on ROIs.
    """
    # with open('/home/matanyaw/DIP_decoder/matanyas_config.yaml', 'r') as f:
    #     matanyas_config = yaml.safe_load(f)

    # --------------------------------------------------------------
    start_time = time.time()
    print(f'\nExperiment run {args_config["run"]}')
    print(f'Experiment started at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
    print(f'Starting the experiment with config:')
    for key, value in args_config.items():
        print(f'  {key}: {value}')

    voxel_paths_blur_fill_excluded = {
        'original_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_excluded_sub_1/original_voxels.pt',
    }
    voxel_paths_blur_fill_shared = {
        'original_voxels_path': '/home/jonathak/VisualEncoder/Results/all_masks_coarse_blur_shared/original_voxels.pt',
    }
    voxel_paths = voxel_paths_blur_fill_shared 


    # Loading models

    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg') # The Dinov2 encoder
    model = torch.load('/home/jonathak/VisualEncoder/Voxels_Prediction/model_ch128.pth').eval().cuda() # The universal encoder model

    # Defining subject
    stroke_sub = 1
    img_meta = None

    # Getting images

    # images = get_images_for_prediction(image_type='excluded', subjects=[stroke_sub])
    # images = get_images_for_prediction(image_type='shared', subjects=[stroke_sub])
    images = get_images_for_prediction(image_type=args_config['image_type'], subjects=[stroke_sub])

    images = images.permute(0, 2, 3, 1)

    # Getting original predicted fMRI

    original_predicted_fMRI = torch.load(voxel_paths['original_voxels_path']).cuda()

    # Getting voxel indices

    lh_start, lh_end = get_hemisphere_indices(stroke_sub, 'lh')
    rh_start, rh_end = get_hemisphere_indices(stroke_sub, 'rh')
    inds = np.arange(lh_start, rh_end)

    # Testing voxel embeddings
    voxel_embeddings = model.voxel_embed # Has shape [315997, 256]
    voxel_embeddings = voxel_embeddings[inds]



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

    
    def save_as_png(array, save_path, metadata=None):   
        # 1) Convert torch tensor → numpy
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()

        # 2) Reorder channels if needed
        if array.ndim == 3 and array.shape[0] == 3:
            array = array.transpose(1, 2, 0)

        # 3) Convert dtype to uint8
        if array.dtype in (np.float32, np.float64):
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255).round().astype(np.uint8)
        elif array.dtype == np.uint8:
            pass
        else:
            raise ValueError(f"Unsupported array dtype: {array.dtype}")

         # 4) Build the PIL Image
        img = Image.fromarray(array)

        # 5) Attach metadata if provided
        if metadata:
            pnginfo = PngImagePlugin.PngInfo()
            for key, val in metadata.items():
                pnginfo.add_text(str(key), str(val))
            img.save(save_path, pnginfo=pnginfo)
        else:
            print('save_as_png was called with no metadata. Saving without metadata.')
            img.save(save_path)

    
    def save_as_png_old2(array, save_path, title=None, subtitle=None, figsize=(4,4)):
        # 1. Convert torch tensor to numpy
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()
        # 2. Reorder channels if needed
        if array.ndim == 3 and array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        # 3. Scale floats to [0–255]
        if array.dtype in (np.float32, np.float64):
            array = (array * 255).astype(np.uint8)
        elif array.dtype == np.uint8:
            pass
        else:
            raise ValueError(f"Unsupported dtype {array.dtype}")

        # 4. Create figure & axis
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(array)
        ax.axis('off')

        # 5. Main title
        if title is not None:
            ax.set_title(title, fontsize=14, pad=6)

        # 6. Subtitle (smaller, placed just below main title)
        if subtitle is not None:
            # Using `fig.text` so we can position it precisely
            fig.text(0.5, 0.92, subtitle,
                    ha='center', va='top',
                    fontsize=10, color='gray')

        # 7. Save & clean up
        plt.tight_layout(pad=0)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


    def save_as_png_old(array, save_path, title=None):
        
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
        if title is not None:
            print('Saving image with title:', title) 
            plt.title(title)
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
    num_iter_no_stroke = 4001 # 3201 max iterations
    num_iter_stroke = 601 # 3201 max iterations
    burnin_iter = 7000 # burn-in iteration for SGLD
    weight_decay = 5e-8
    mse = torch.nn.MSELoss().type(dtype) # loss

    save_throughout = args_config['save_throughout']
    n_saves = 5
    save_every = num_iter_stroke // n_saves

    ROIs = get_roi_names(args_config['roi_to_process'])  # Get the list of ROIs to process

    # Creating image indices

    # images_indices = np.sort(np.array([1, 5, 6, 9, 12, 13, 14, 22, 28, 27, 66, 119, 39, 109, 56, 44, 69]))
    # images_indices = [69,109,119]

    # images_indices = np.sort(np.array([1, 5, 6, 9, 12, 13, 14, 22, 28, 27, 66, 69, 39, 109, 56, 44])) # For excluded
    # images_indices = np.sort(np.array([1, 4, 7, 9, 15, 16, 18, 20, 21, 29, 51, 65, 69, 96, 99])) # For shared

    images_indices = np.sort(np.array(args_config['images_indices']))


    # Results path

    root_save_path = args_config['save_path']
    # The images loop
    print('\nStarting the images loop...\n')
    for image_counter, img_idx in enumerate(images_indices, start=1):
        
        if image_counter > 1:
            t = time.time()
            print(f'Image {image_counter-1} out of {len(images_indices)} finished in {timedelta(seconds=t - start_time)}')
        
        image_save_path = f'{root_save_path}/img_{img_idx}'
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path, exist_ok=True)
        
        img_meta = {
            'title': f'Original Image',
            'Image Index': img_idx,
            'Image Type': args_config['image_type'],
            'Run Name': args_config['run'],
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Saving the original image
        save_as_png(images[img_idx], f'{image_save_path}/img_{img_idx}_original_image.png', metadata=img_meta)

        # Getting the stroke fill value
        min_val = original_predicted_fMRI[img_idx,inds].min()
        
        # Original target voxel maps
        original_target_all = original_predicted_fMRI[img_idx,inds].unsqueeze(0).float().to(device)
        original_target_nc = original_predicted_fMRI[img_idx,inds_nc].unsqueeze(0).float().to(device)


        # Step 1: Fitting the network to the image
        # =====================================
        if 1 in args_config['steps_to_do']:

            in_img = trans_imgs(images[img_idx])

            # Initialize DIP network
            net = get_net(input_depth, 'skip', pad,
                        skip_n33d=128, 
                        skip_n33u=128,
                        skip_n11=2,
                        num_scales=3,
                        upsample_mode='bilinear').type(dtype)

            ## Optimize
            state_dict = {
                'i': 0,
                'out_avg': None,
                'out_avg_np': None,
                'net_input': get_noise(input_depth, INPUT, (224, 224),var=0.1).type(dtype).detach(),
            }
            # net_input = get_noise(input_depth, INPUT, (224, 224),var=0.1).type(dtype).detach()
            net_input_saved = state_dict['net_input'].detach().clone()
            noise = state_dict['net_input'].detach().clone()

            out = net(state_dict['net_input'])
            rec_img_np = out.detach().cpu().numpy()[0]

            def closure():

                # global i, out_avg, net_input, out_avg_np
                if reg_noise_std > 0:
                    state_dict['net_input'] = net_input_saved + (noise.normal_() * reg_noise_std)
                out = net(state_dict['net_input'])

                if state_dict['out_avg'] is None:
                    state_dict['out_avg'] = out.detach()
                else:
                    state_dict['out_avg'] = state_dict['out_avg'] * exp_weight + out.detach() * (1 - exp_weight)

                loss = F.mse_loss(out, in_img.cuda())

                loss.backward()

                state_dict['out_avg_np'] = state_dict['out_avg'].detach().cpu().numpy()[0]
                
                # Saving intermediate image
                # if save_throughout and state_dict['i'] % save_every == 0 and state_dict['i'] != 0:
                #     save_as_png(state_dict['out_avg_np'], f'{image_save_path}/img_{img_idx}_fitted_image_iter_{state_dict['i']}.png')
                
                state_dict['i'] += 1
                return loss

            ## Optimizing 
            
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            for j in range(num_iter_no_stroke):
                optimizer.zero_grad()
                closure()
                optimizer.step()
                
            print(f'Finished fitting on image for image {img_idx} (Step 1)')
            
            # Saving the fitted image
            img_meta = {
                'title': f'DIP Fitted Image',
                'Image Index': img_idx,
                'Image Type': args_config['image_type'],
                'Run Name': args_config['run'],
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_as_png(state_dict['out_avg_np'], f'{image_save_path}/img_{img_idx}_fitted_image_final.png', metadata=img_meta)

            # Saving the fitted network
            torch.save({
                'net_state': net.state_dict(),
                'out_avg': state_dict['out_avg']
            }, os.path.join(image_save_path, 'dip_on_original_image.pth'))

            torch.cuda.empty_cache()
        
        
        
        # Step 2: Decoding normal fMRI voxel map
        # ===================================
        if 2 in args_config['steps_to_do']:
            
            net = get_net(input_depth, 'skip', pad,
                            skip_n33d=128, 
                            skip_n33u=128,
                            skip_n11=2,
                            num_scales=3,
                            upsample_mode='bilinear').type(dtype)

            checkpoint = torch.load(os.path.join(image_save_path, 'dip_on_original_image.pth'))
            net.load_state_dict(checkpoint['net_state'])

            state_dict['out_avg'] = checkpoint['out_avg']

            reg_noise_std = 1./30.
            state_dict['i'] = 0 

            def closure():

                # global i, out_avg, net_input, out_avg_np
                
                if reg_noise_std > 0:
                    state_dict['net_input'] = net_input_saved + (noise.normal_() * reg_noise_std)
                out = net(state_dict['net_input'])

                if state_dict['out_avg'] is None:
                    state_dict['out_avg'] = out.detach()
                else:
                    state_dict['out_avg'] = state_dict['out_avg'] * exp_weight + out.detach() * (1 - exp_weight)
                
                enc_in = (out-mean)/std
                vox_pred = model(enc_in, inds_nc_torch.unsqueeze(0).cuda())
                
                loss = F.mse_loss(vox_pred, original_target_nc)     # How close we are to the original fMRI voxel map

                loss.backward()

                state_dict['out_avg_np'] = state_dict['out_avg'].detach().cpu().numpy()[0]
                
                # # Saving intermediate image
                # if save_throughout and state_dict['i'] % save_every == 0 and state_dict['i'] != 0:
                #     save_as_png(state_dict['out_avg_np'], f'{image_save_path}/img_{img_idx}_normal_fMRI_iter_{state_dict['i']}.png')

                state_dict['i'] += 1
                return loss

            ## Optimizing 

            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            for j in range(num_iter_no_stroke):
                optimizer.zero_grad()
                closure()
                optimizer.step()
                
            print(f'Finished decoding original fMRI for image {img_idx} (Step 2)')
            
            # MAYBE REMOVE- start from fMRI 
            torch.save({
                'net_state': net.state_dict(),
                'out_avg': state_dict['out_avg']
            }, os.path.join(image_save_path, 'dip_on_fMRI_image.pth'))
            
            torch.cuda.empty_cache()
            
            # Saving the normal fMRI final image

            img_meta = {
                'title': f'DIP Fitted Image - Normal fMRI',
                'Image Index': img_idx,
                'Image Type': args_config['image_type'],
                'Run Name': args_config['run'],
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_as_png(state_dict['out_avg_np'], f'{image_save_path}/img_{img_idx}_normal_fMRI_final.png', metadata=img_meta)

        # The ROI masks loop
        # ==================
        
        if 3 not in args_config['steps_to_do'] and 4 not in args_config['steps_to_do']:
            continue
        print(f'\nStarting the ROIs loop for image {img_idx}...\n')
        
        meanshift_center_dict = {}
        
        predefined_ROI_indices = {}

        # Creating a dictionary of ROI indices (iterating over copy because we remove ROIs that don't exist)
        for ROI in ROIs.copy():
            
            roi_indices = get_roi_indices(stroke_sub, ROI)
            
            if roi_indices is None:
                ROIs.remove(ROI)
                print(f"ROI {ROI} does not exist for subject {stroke_sub}. Removing from list.")
                # print('ROI indices:', roi_indices)
                continue

            else:
                predefined_ROI_indices[ROI] = roi_indices
            print("ROI:", ROI)
            if args_config['modify_roi']:
                meanshift_center_dict[ROI] = ROI_actions.infer_center_by_meanshift(roi_indices, voxel_embeddings)
                # print(f"{ROI} center shape: {tuple(center_dict[ROI].shape)}")

        if args_config['modify_roi']:
            labels, meanshift_roi_indices = ROI_actions.assign_voxels_to_rois(voxel_embeddings, meanshift_center_dict, ROIs)

        
        for ROI in ROIs:
            mega_roi_dict = dict()

            mega_roi_dict['Predefined_ROI'] = predefined_ROI_indices[ROI]

            if args_config['modify_roi']:
                mega_roi_dict['Top-K_ROI'] = ROI_actions.infer_by_center_of_mass(predefined_ROI_indices[ROI],
                                                                                        voxel_embeddings, 
                                                                                        threshold=None, 
                                                                                        top_k=len(predefined_ROI_indices[ROI]),
                                                                                        use_angle=False)
                if args_config['use_angle']:
                    mega_roi_dict['Top-K_ROI_angle'] = ROI_actions.infer_by_center_of_mass(predefined_ROI_indices[ROI],
                                                                                            voxel_embeddings, 
                                                                                            threshold=None, 
                                                                                            top_k=len(predefined_ROI_indices[ROI]),
                                                                                            use_angle=True)
                    
                    # avg_dist_from_center_of_mass = ROI_actions.get_average_distance(predefined_ROI_indices[ROI],
                    #                                                                 voxel_embeddings,
                    #                                                                 use_angle=args_config['use_angle'])
                    # mega_roi_dict['avg_dist_ROI'] = ROI_actions.infer_by_center_of_mass(predefined_ROI_indices[ROI], 
                    #                                                                     voxel_embeddings, 
                    #                                                                     threshold=avg_dist_from_center_of_mass, 
                    #                                                                     top_k=None,
                    #                                                                     use_angle=args_config['use_angle'])
                    # mega_roi_dict['Threshold_ROI'] = ROI_actions.infer_by_center_of_mass(predefined_ROI_indices[ROI], voxel_embeddings, threshold=0.5, top_k=None)
                mega_roi_dict['Meanshift_ROI'] = meanshift_roi_indices[ROI]
            
            roi_path = f'{image_save_path}/roi_{ROI}'
            if not os.path.exists(roi_path):
                os.makedirs(roi_path, exist_ok=True)
            
            stroke_target_all = original_target_all.clone().detach()
            
            for roi_version, roi_indices in mega_roi_dict.items():              # For each ROI version (Predefined or Modified)
                stroke_target_all[0, roi_indices] = stroke_target_all.min()
                stroke_target_all = stroke_target_all.float()
                stroke_target_nc = stroke_target_all[:,inds_nc]

                # Counting stroke voxels in the nc case
                current_stroke_indices_count = torch.sum((original_target_nc != min_val) & (stroke_target_nc == min_val))

                # Moving to cuda
                stroke_target_all = stroke_target_all.to(device)
                stroke_target_nc = stroke_target_nc.to(device)
                
                # Step 3.1: Decoding stroke fMRI voxel map
                # ===================================

                if 3 in args_config['steps_to_do']:
                    print(f'Starting decoding stroke fMRI for image {img_idx}, ROI: {roi_version} {ROI} (step 3.1)')
                    # print(f'ROI size: {roi_indices.shape[0]}, Stroke voxels in nc case: {current_stroke_indices_count}')
                    net = get_net(input_depth, 'skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128,
                    skip_n11=2,
                    num_scales=3,
                    upsample_mode='bilinear').type(dtype)

                    # Starting from original image
                    checkpoint = torch.load(os.path.join(image_save_path, 'dip_on_original_image.pth'))
                    net.load_state_dict(checkpoint['net_state'])
                    state_dict['out_avg'] = checkpoint['out_avg']

                    reg_noise_std = 1./30.
                    state_dict['i'] = 0 

                    def closure():

                        # global i, out_avg, net_input, out_avg_np
                        
                        if reg_noise_std > 0:
                            state_dict['net_input'] = net_input_saved + (noise.normal_() * reg_noise_std)
                        out = net(state_dict['net_input'])

                        if state_dict['out_avg'] is None:
                            state_dict['out_avg'] = out.detach()
                        else:
                            state_dict['out_avg'] = state_dict['out_avg'] * exp_weight + out.detach() * (1 - exp_weight)
                        
                        enc_in = (out-mean)/std
                        vox_pred = model(enc_in, inds_nc_torch.unsqueeze(0).cuda())
                        
                        loss = F.mse_loss(vox_pred, stroke_target_nc)#+total_variation_loss(out)+0.01*norm_6(out)#- 0.1*torch.mean(F.cosine_similarity(vox_pred, target))

                        loss.backward()

                        state_dict['out_avg_np'] = state_dict['out_avg'].detach().cpu().numpy()[0]

                        # Saving intermediate image 
                        if save_throughout and state_dict['i'] % save_every == 0 and state_dict['i'] != 0:
                            img_meta = {
                                'title': f'{roi_version} - start original',
                                'Image Index': img_idx,
                                'Image Type': args_config['image_type'],
                                'ROI Version': roi_version,
                                'ROI Name': ROI,
                                'ROI Size': roi_indices.shape[0],
                                    'Run Name': args_config['run'],
                                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Iteration': state_dict['i']
                            }

                            save_as_png(state_dict['out_avg_np'], 
                                        f"{roi_path}/img_{img_idx}_stroke_{roi_version}_{ROI}_iter_{state_dict['i']}_start_original.png",
                                        metadata=img_meta)
                        state_dict['i'] += 1
                        return loss

                    ## Optimizing 
                    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
                    for j in range(num_iter_stroke):
                        optimizer.zero_grad()
                        closure()
                        optimizer.step()

                    torch.cuda.empty_cache()
                    
                    print(f'Finished decoding stroke fMRI for image {img_idx}, ROI: {roi_version} {ROI} (step 3.1)')
                    img_meta = {
                        'title': f'{roi_version} - start original',
                        'Image Index': img_idx,
                        'Image Type': args_config['image_type'],
                        'ROI Version': roi_version,
                        'ROI Name': ROI,
                        'ROI Size': roi_indices.shape[0],
                        'Run Name': args_config['run'],
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    save_as_png(state_dict['out_avg_np'], f'{roi_path}/img_{img_idx}_stroke_{roi_version}_{ROI}_start_original.png', metadata=img_meta)

                # Step 3.2: Decoding stroke fMRI voxel map
                # ===================================

                if 4 not in args_config['steps_to_do']:
                    continue
                print(f'Starting decoding stroke fMRI for image {img_idx}, ROI: {roi_version} {ROI} (step 3.2)')
                # print(f'ROI size: {roi_indices.shape[0]}, Stroke voxels in nc case: {current_stroke_indices_count}')
                checkpoint = torch.load(os.path.join(image_save_path, 'dip_on_fMRI_image.pth'))
                net.load_state_dict(checkpoint['net_state'])
                state_dict['out_avg'] = checkpoint['out_avg']

                reg_noise_std = 1./30.
                state_dict['i'] = 0 

                def closure():

                    # global i, out_avg, net_input, out_avg_np
                    
                    if reg_noise_std > 0:
                        state_dict['net_input'] = net_input_saved + (noise.normal_() * reg_noise_std)
                    out = net(state_dict['net_input'])

                    if state_dict['out_avg'] is None:
                        state_dict['out_avg'] = out.detach()
                    else:
                        state_dict['out_avg'] = state_dict['out_avg'] * exp_weight + out.detach() * (1 - exp_weight)
                    
                    enc_in = (out-mean)/std
                    vox_pred = model(enc_in, inds_nc_torch.unsqueeze(0).cuda())
                    
                    loss = F.mse_loss(vox_pred, stroke_target_nc)#+total_variation_loss(out)+0.01*norm_6(out)#- 0.1*torch.mean(F.cosine_similarity(vox_pred, target))

                    loss.backward()

                    state_dict['out_avg_np'] = state_dict['out_avg'].detach().cpu().numpy()[0]
                    
                    # Saving intermediate image 
                    if save_throughout and state_dict['i'] % save_every == 0 and state_dict['i'] != 0:
                        img_meta = {
                            'title': f'{roi_version} - start fMRI',
                            'Image Index': img_idx,
                            'Image Type': args_config['image_type'],
                            'ROI Version': roi_version,
                            'ROI Name': ROI,
                            'ROI Size': roi_indices.shape[0],
                            'Run Name': args_config['run'],
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Iteration': state_dict['i']
                        }
                        save_as_png(state_dict['out_avg_np'], 
                                    f"{roi_path}/img_{img_idx}_stroke_{roi_version}_{ROI}_iter_{state_dict['i']}_start_fMRI.png", 
                                    metadata=img_meta)

                    state_dict['i'] += 1
                    return loss

                ## Optimizing 
                optimizer = torch.optim.Adam(net.parameters(), lr=LR)
                for j in range(num_iter_stroke):
                    optimizer.zero_grad()
                    closure()
                    optimizer.step()


                torch.cuda.empty_cache()
                print(f'Finished decoding stroke fMRI for image {img_idx}, ROI: {roi_version} {ROI} (step 3.2)')
                # Saving the stroke fMRI final image
                img_meta = {
                    'title': f'{roi_version} - start fMRI',
                    'Image Index': img_idx,
                    'Image Type': args_config['image_type'],
                    'ROI Version': roi_version,
                    'ROI Name': ROI,
                    'ROI Size': roi_indices.shape[0],
                    'Run Name': args_config['run'],
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                save_as_png(state_dict['out_avg_np'], f'{roi_path}/img_{img_idx}_stroke_{roi_version}_{ROI}_start_fMRI.png', metadata=img_meta)

                create_montage(input_dir=roi_path, input_dir2=image_save_path, main_title=f'Image {img_idx} - ROI {ROI} Subject {stroke_sub}')


    print('Finished all experiments!')
    

def argparse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the complete stroke experiment on ROIs.')

    parser.add_argument('--save_path', type=str, default=None,
                        help='Root path to save results.')
    parser.add_argument('--run', type=str, default=None, help='Name of the run for saving results.')
    parser.add_argument('--steps_to_do', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='List of steps to perform in the experiment.')
    parser.add_argument('--image_type', type=str, default='shared',
                        help='Type of images to use in the experiment (e.g., "shared", "excluded").')
    parser.add_argument('--images_indices', type=int, nargs='+', default=[1, 4],
                        help='List of image indices to process in the experiment.')
    parser.add_argument('--modify_roi', action='store_true',
                        help='Whether to modify the ROI during the experiment.')
    parser.add_argument('--save_throughout', action='store_true',
                        help='Whether to save intermediate results throughout the experiment.')
    parser.add_argument('--use_angle', action='store_true',
                        help='Whether to use angle-based distance for ROI inference.')
    parser.add_argument('--roi_to_process', type=str, nargs='+', default=None,
                        help='List of specific ROIs to process. If None, all predefined ROIs will be processed.')
    # parser.add_argument('--roi_indexes_file', type=str, required=True,
                        # help='Path to the file containing ROI indexes.')

    return parser.parse_args()

def small_script():
    a = 1 + 1
    a = a / 0
    return a

def send_notification(message):
    """Send a notification using ntfy."""
    try:
        requests.post(
            "https://ntfy.sh/complete_stroke_experiment_predicted_fMRI_ROIs",
            data=message.encode("utf-8")
        )
    except Exception as e:
        print(f"Failed to send ntfy notification: {e}")

if __name__ == "__main__":
    start_time = time.time()
    args = argparse_args()
    current_date = datetime.now().strftime("%y_%m_%d")
    default_save_path = f'/home/matanyaw/DIP_decoder/matanya_results/results_{current_date}'
    if args.save_path is None:
        save_path = default_save_path
    else:
        save_path = args.save_path
    
    if args.run:
        save_path = os.path.join(save_path, f'run_{args.run}')
        os.makedirs(save_path, exist_ok=True)

    args_config = {
        'images_indices': args.images_indices,  # List of image indices to process in the experiment
        'save_path': save_path,  # Root path to save results
        'steps_to_do': args.steps_to_do,  # List of steps to perform in the experiment
        'image_type': args.image_type,  # Type of images to use in the experiment
        'modify_roi': args.modify_roi,  # Whether to modify the ROI during the experiment
        'run': args.run,  # Name of the run for saving results
        'save_throughout': args.save_throughout,
        'use_angle': args.use_angle,  # Whether to use angle-based distance for ROI inference
        'roi_to_process': args.roi_to_process,  # List of specific ROIs to process
        
        # 'save_throughout': true,
        # 'n_saves': 5,
        # 'num_iter_no_stroke': 4001,
        # 'num_iter_stroke': 601,
        # 'learning_rate': 0.001,
    }

    # Just to see which GPU is being used
    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        print(f"Using CUDA device #{dev_id}: {torch.cuda.get_device_name(dev_id)}")
        print(f"Total GPUs available: {torch.cuda.device_count()}")
    else:
        print("No GPU detected!")


    try: 
        run_experiment(args_config)
        # small_script()
        end_time = time.time()
        time_taken = timedelta(seconds=end_time - start_time)
        print(f'Total time taken: {time_taken}')
        send_notification(f"✅ Experiment completed successfully\nRun: {args_config['run']} Time: {time_taken}\n")
    except Exception as e:
        end_time = time.time()
        time_taken = timedelta(seconds=end_time - start_time)
        print(f'Total time taken: {time_taken}')
        send_notification(f"❌ Experiment failed\nRun: {args_config['run']} with error: {e}\n Time: {time_taken}\n")
        raise
    

# python complete_stroke_experiment_predicted_fMRI_ROIs.py --images_indices 1 4  --steps_to_do 1 2 3 --run 12
