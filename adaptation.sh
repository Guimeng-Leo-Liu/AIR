#!/usr/bin/env bash

export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python train.py --size 1024 --batch 2 --n_sample 16 --lambda_global 0.0 --lambda_direction 1.0 \
--output_dir ./output/ --lr 0.002 --source_model_type "ffhq" --frozen_gen_ckpt ./pretrained/stylegan2/stylegan2-ffhq-config-f.pt --iter 201 \
--source_class "human" --target_class "baby" --auto_layer_k 18 --auto_layer_iters 1 --auto_layer_batch 8 \
--output_interval 50 --clip_models "ViT-B/32" --clip_model_weights 1.0 --mixing 0 --save_interval 200  \
--n_ctx 4 --ctx_init "a photo of a" \
--adaptative_direction --adaptative_interval 20  --adaptative_lambda 1 --start_adaptive_iteration 100 \
--evaluation_in_training --evaluation_interval 0 100 150 200 250 300 --use_last_anchor --redefine_prompt

CUDA_VISIBLE_DEVICES=0 python train.py --size 1024 --batch 2 --n_sample 16 --lambda_global 0.0 --lambda_direction 1.0 \
--output_dir ./output/ --lr 0.002 --source_model_type "ffhq" --frozen_gen_ckpt ./pretrained/stylegan2/stylegan2-ffhq-config-f.pt --iter 301 \
--source_class "photo" --target_class "sketch" --auto_layer_k 18 --auto_layer_iters 1 --auto_layer_batch 8 \
--output_interval 50 --clip_models "ViT-B/32" --clip_model_weights 1.0 --mixing 0 --save_interval 200  \
--n_ctx 4 --ctx_init "a photo of a" \
--adaptative_direction --adaptative_interval 30  --adaptative_lambda 1 --start_adaptive_iteration 150 \
--evaluation_in_training --evaluation_interval 0 100 150 200 250 300 --use_last_anchor --redefine_prompt


