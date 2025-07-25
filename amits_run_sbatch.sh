#!/bin/bash
#SBATCH --job-name=sub3_vgg_trans # Job name
#SBATCH --output=run_logs/external_%j.out # Standard output and error log
#SBATCH --error=run_logs/external_%j.err  # Error log
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --mem=200G               # Job memory request
#SBATCH --gres=gpu:L40S:1  # Request one gpu
#SBATCH --time=100:00:00          # Time limit hrs:min:sec (adjust as needed)
#SBATCH --partition=irani_run.q      # Partition name

# Load necessary modules
eval "$(conda shell.bash hook)"
conda activate fmri-torch
# /home/romanb/data/miniconda3/envs/fmri-gpu
# cd /home/amitz/data/fmri_project/LoDa/scripts/benchmark/
# ./benchmark_loda_spaq.sh
srun python /home/amitz/data/roman_encoder/roman_encoder/icml2025_rebuttal/run_linear_regression.py --subject 3 --model 'vgg'
# srun python /home/amitz/data/roman_encoder/roman_encoder/icml2025_rebuttal/predictVoxels.py --subject 7 --imgs 'ext'
# srun python /home/amitz/data/decoding/generate_train_unclip_images.py
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'coffee mug' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'smartphone' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'sofa' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'tree' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'pasta' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'food' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'house' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'guitar' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'car' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'bicycle' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'tattoo' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'face' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'doll' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'shoe' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'forest' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'shop' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'beach' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'
# srun python /home/amitz/data/PerceptualTokens/main.py run --object 'elephant' --clf_path '/home/amitz/data/PerceptualTokens/emotion_model_Karlo_CLIP.pt' --init_token 'awe-inspiring' --emo 'awe' --task 'emo'

# srun python /home/amitz/data/clip_perceptual/main.py train_test --dataset_model 'lamem' --dataset_data 'lamem' --lora_model 'clip' --user 'amit' --alpha 1 --r 16 --lora_alpha 8 --train_all_data 'True' --trial 2
# srun python /home/amitz/data/clip_perceptual/main.py train_only_mlp --dataset_model 'EmoSet' 'emotionROI' --dataset_data 'EmoSet' --user 'amit' --lora_model 'clip_noOpen' --alpha 1 --r 16 --lora_alpha 8

# srun python /home/amitz/data/clip_perceptual/main.py test --dataset_model 'lamem' --dataset_data 'THINGS' --lora_model 'clip' --lr 0.00005 --user 'amit' --alpha 1 --trial 2 --r 16 --lora_alpha 8 --test_all_dataset 'True' --train_all_data 'True' 
# srun python /home/amitz/data/clip_perceptual/main.py test --dataset_model 'lamem' --dataset_data 'THINGS' --lora_model 'clip' --lr 0.0001 --user 'amit' --alpha 1 --trial 0 --r 16 --lora_alpha 8 --test_all_dataset 'True' --train_all_data 'True' 

# srun python /home/amitz/data/fmri_project/Inheritance/main.py aggregation --dataset 'lamem' --num_embeds 0 --user 'amit' --lora_model 'clip' --train_enc 'scratch' --init 'default' --enc_model 'clip' --trial 12 --alpha 1
# srun python /home/amitz/data/roman_encoder/roman_encoder/Final/test_copy_2.py
# srun python /home/amitz/data/fmri_project/Inheritance/experiments_test.py
# srun python /home/amitz/data/fmri_project/Inheritance/test_model.py
# srun python /home/amitz/roman_encoder/Project_N/regression.py
# srun python /home/amitz/data/fmri_project/Inheritance/test_model.py
# srun python /home/amitz/data/fmri_project/Project_N/main.py parser_emo --lr 1e-3 --feat "clip" --save_path '1e-3 batch 32' --num_embeds 2000 --head 'mlp' --init 'roi'
# srun python /home/amitz/data/fmri_project/Project_N/emoProj.py
# srun python /home/amitz/data/fmri_project/Inheritance/main.py train_test_no_optuna --dataset 'lamem' --feat 'clip'

# srun python /home/amitz/data/fmri_project/Inheritance/main.py run_optuna --dataset 'lamem' --embeds True --feat 'clip'

# srun python /home/amitz/data/fmri_project/Inheritance/main.py run_optuna --dataset 'EmoSet' --embeds True --num_emotions 8 --feat 'dino_regular'
# srun python /home/amitz/data/fmri_project/Inheritance/main.py test_model --dataset 'EmoSet' --num_embeds 2000 --lora_model 'dino' --lr 0.0002 --feat 'dino_lora' --batch_size 64 --loss 'cross_entropy' --init 'roi' --all_train True
# srun python /home/amitz/data/fmri_project/Inheritance/main.py test_model --dataset 'EmoSet' --num_embeds 2000 --lora_model 'dino' --lr 0.0002 --feat 'dino_lora' --batch_size 64 --loss 'cross_entropy' --init 'default' --all_train True

# srun python /home/amitz/data/fmri_project/Inheritance/extract_features.py
# srun python /home/amitz/data/fmri_project/Project_N/extract_features.py
# srun python /home/amitz/data/fmri_project/Project_N/main.py run_mlp --dataset "LiveC" --save_path '1e-3 batch 32 val15' --num_embeds 2000 --loda_split 'True' --feat 'loda' --all 'True' --avg_test 'True' --loss 'plcc' --head 'linear' --init 'roi' --val_15 'True'
# srun python /home/amitz/roman_encoder/Project_N/main.py run_mlp --dataset "kon10k" --save_path '1e-3 batch 32' --num_embeds 1000 --loda_split 'True' --feat 'loda' --all 'True' --avg_test 'True' --loss 'plcc' --head 'trained_linear' --init 'roi'

# srun python /home/amitz/roman_encoder/Project_N/main.py run_mlp --dataset "kon10k" --save_path '1e-3 batch 32' --num_embeds 1000 --loda_split 'True' --feat 'loda' --all 'True' --avg_test 'True' --loss 'mse' --head 'mlp' --init 'corr'

# srun python /home/amitz/roman_encoder/Project_N/main.py run_mlp --dataset "kon10k" --save_path '1e-3 batch 32' --num_embeds 1000 --loda_split 'True' --feat 'loda' --all 'True' --avg_test 'True' --loss 'mse' --head 'linear' --init 'roi'

# srun python /home/amitz/roman_encoder/Project_N/main.py run_mlp --dataset "kon10k" --save_path '1e-3 batch 32' --num_embeds 2000 --loda_split 'True' --feat 'loda' --all 'True' --avg_test 'True' --loss 'plcc' --head 'mlp'
# srun python /home/amitz/roman_encoder/Project_N/main.py run_mlp --dataset "kon10k" --save_path '1e-3 batch 32' --num_embeds 2000 --loda_split 'True' --feat 'loda' --all 'True' --avg_test 'True' --loss 'plcc' --head 'mlp'

# srun python /home/amitz/roman_encoder/Project_N/eval_mlp.py




# srun python /home/amitz/roman_encoder/temp.py

# ridge
# srun python /home/amitz/roman_encoder/Project/main.py run_ridge --dataset "kon10k"  --split 1 --clip 'True'
# pred voxels
# srun python /home/amitz/roman_encoder/Project/main.py predict_voxels --dataset "kon10k" --split 0 --loda_split 'True'

# preprocess
# srun python /home/amitz/roman_encoder/Project/main.py preprocess_data --dataset "kon10k" --split 0 --loda_split 'True'

# Run the experiment
# srun python /home/amitz/roman_encoder/Project/main.py run_mlp --dataset "kon10k" --save_path '1e-3 batch 32' --dino 'False' --num_embeds 1000 --loda_split 'True' --loda_feat 'True' --split 0
# srun python /home/amitz/roman_encoder/Project/main.py run_mlp --dataset "kon10k" --save_path '1e-3 batch 32' --dino 'True' --num_embeds 2000
# srun python /home/amitz/roman_encoder/Project/main.py run_mlp --dataset "SPAQ" --save_path '1e-3 batch 32' --dino 'True' --num_embeds 500
# srun python /home/amitz/roman_encoder/Project/main.py run_mlp --dataset "SPAQ" --save_path '1e-3 batch 32' --dino 'True' --num_embeds 2000

# Dino features
# srun python /home/amitz/roman_encoder/Project/main.py extract_features --dataset "kon10k" --which 'dino'
# srun python /home/amitz/roman_encoder/Project/main.py extract_features --dataset "SPAQ" --which 'dino'
# srun python /home/amitz/roman_encoder/Project/main.py extract_features --dataset "lamem" --which 'dino'
# srun python /home/amitz/roman_encoder/Project/main.py extract_features --dataset "LiveC" --which 'dino'

# srun python /home/amitz/roman_encoder/Project/main.py extract_features --dataset "kon10k" --which 'LODA' --split 0

# srun python /home/amitz/roman_encoder/Project/main.py run_mlp_feat_only --dataset "kon10k" --features 'LODA' --split 0
