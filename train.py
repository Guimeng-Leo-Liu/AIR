'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    CUDA_VISIBLE_DEVICES=1 python train.py --size 1024
                                           --batch 2
                                           --n_sample 4
                                           --output_dir /path/to/output/dir
                                           --lr 0.002
                                           --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt
                                           --iter 301
                                           --source_class "photo"
                                           --target_class "sketch"
                                           --lambda_direction 1.0
                                           --lambda_patch 0.0
                                           --lambda_global 0.0
                                           --lambda_texture 0.0
                                           --lambda_manifold 0.0
                                           --phase None
                                           --auto_layer_k 0 # 1024->8, 512->7
                                           --auto_layer_iters 0
                                           --auto_layer_batch 8
                                           --output_interval 50
                                           --clip_models "ViT-B/32" "ViT-B/16"
                                           --clip_model_weights 1.0 1.0
                                           --mixing 0.0
                                           --save_interval 50
'''
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import numpy as np

import torch
import torchvision
from torchvision import utils
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import shutil
import json
import pickle
import copy
import random
import PIL.Image
from tensorboardX import SummaryWriter
import clip
from model.sg2_model import Generator, Discriminator
from criteria.clip_loss import CLIPLoss
import legacy as legacy
from copy import deepcopy

# from utils.file_utils import copytree, save_images, save_paper_image_grid, save_torch_img
from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions
import time


#TODO convert these to proper args
SAVE_SRC = False
SAVE_DST = True
save_img = False
write = False


class SG2Generator(torch.nn.Module):
    def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Generator, self).__init__()

        self.generator = Generator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator.load_state_dict(checkpoint["g_ema"], strict=False)

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])
        if phase == 'shape':
            # layers 1-2
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers()
        else:
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])

    def trainable_params(self):
        params = []
        for layer in self.get_training_layers():
            params.extend(layer.parameters())

        return params

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                # print(layer)
                self.requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                # print(layer)
                self.requires_grad(layer, True)

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''

        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)

    def modulation_layers(self):
        return self.generator.modulation_layers

    def forward(self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        input_is_s_code=False,
        noise=None,
        randomize_noise=True):
        return self.generator(styles, return_latents=return_latents, truncation=truncation, truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, input_is_latent=input_is_latent, input_is_s_code=input_is_s_code)

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))

def get_parameter_number(net, name= "un-named net"):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Name=', name, 'Total=', total_num, '# of Trainable Params.=', trainable_num, '# of Fixed Params.=', total_num-trainable_num)
    return trainable_num

def text_encoder(source_prompts, source_tokenized_prompts, clip_model):
    x = source_prompts + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(0, 2, 1, 3)  # NLD -> LND

    transformed_outputs = []
    for j in range(len(x)):
        transformed_output = clip_model.transformer(x[j])
        transformed_outputs.append(transformed_output)

    # Stack the transformed outputs along the appropriate dimension
    x = torch.stack(transformed_outputs, dim=0)

    # for j in range(len(x)):
    #     x[j] = clip_model.transformer(x[j])
    x = x.permute(0, 2, 1, 3)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    text_features = x[:, torch.arange(x.shape[1]), source_tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection

    return text_features

def compute_text_features(prompts, source_prefix, source_suffix, source_tokenized_prompts, clip_model, batch):
    source_ctx = prompts.unsqueeze(1)
    source_prefix = source_prefix.expand(batch, -1, -1, -1)
    source_suffix = source_suffix.expand(batch, -1, -1, -1)
    source_prompts = torch.cat(
        [
            source_prefix,  # (batch, n_cls, 1, dim)
            source_ctx,  # (batch, n_cls, n_ctx, dim)
            source_suffix,  # (batch, n_cls, *, dim)
        ],
        dim=2,
    )
    # print(source_prefix.shape)
    # print(source_ctx.shape)
    # print(source_suffix.shape)
    # print(source_prompts.shape)
    text_features = text_encoder(source_prompts, source_tokenized_prompts, clip_model)
    return text_features.squeeze(1)

def determine_opt_layers(args, generator_frozen, generator_trainable, clip_loss_models):
    z_dim = 512
    sample_z = torch.randn(args.auto_layer_batch, z_dim, device=device) # [8, 512]

    initial_w_codes = generator_frozen.style([sample_z])
    initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, generator_frozen.generator.n_latent, 1) # [8, 18, 512]

    w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(device)

    w_codes.requires_grad = True

    w_optim = torch.optim.SGD([w_codes], lr=0.01)

    for _ in range(args.auto_layer_iters):
        w_codes_for_gen = w_codes.unsqueeze(0)
        # with torch.no_grad():
        #     frozen_img = generator_frozen(w_codes_for_gen, input_is_latent=True)[0]
        generated_from_w = generator_trainable(w_codes_for_gen, input_is_latent=True)[0]

        # w_loss = clip_loss_models.clip_directional_loss(frozen_img, args.source_class, generated_from_w, args.target_class)
        w_loss = clip_loss_models.global_clip_loss(generated_from_w, args.target_class)
        w_optim.zero_grad()
        w_loss.backward()
        w_optim.step()

    # 用global CLIP loss训练auto_layer_iters轮找出参数变化最大的auto_layer_k个层
    layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
    chosen_layer_idx = torch.topk(layer_weights, args.auto_layer_k)[1].cpu().numpy()

    all_layers = list(generator_trainable.get_all_layers())
    conv_layers = list(all_layers[4])
    rgb_layers = list(all_layers[6])  # currently not optimized

    idx_to_layer = all_layers[2:4] + conv_layers  # add initial convs to optimization

    chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx]

    # uncomment to add RGB layers to optimization.
    # for idx in chosen_layer_idx:
    #     if idx % 2 == 1 and idx >= 3 and idx < 14:
    #         chosen_layers.append(rgb_layers[(idx - 3) // 2])

    # uncomment to add learned constant to optimization
    # chosen_layers.append(all_layers[1])

    return chosen_layers, chosen_layer_idx

def forward(z, clip_loss_models, generator_frozen, generator_trainable, generator_intermediate=None, require_grad=False, truncation=1):

    with torch.no_grad():
        w_styles = generator_frozen.style(z)
        # print(w_styles[0].shape)
        # if w_styles[0].shape[0] == 16:
        #     print(w_styles)
        frozen_img = generator_frozen(w_styles, input_is_latent=True, truncation=truncation)[0]
        if generator_intermediate is not None:
            adapted_img = generator_intermediate(w_styles, input_is_latent=True, truncation=truncation)[0]
        if not require_grad:
            trainable_img = generator_trainable(w_styles, input_is_latent=True, truncation=truncation)[0]
    if require_grad:
        trainable_img = generator_trainable(w_styles, input_is_latent=True, truncation=truncation)[0]

    clip_loss = clip_loss_models(frozen_img, args.source_class, trainable_img, args.target_class, write=False)
    if generator_intermediate is not None:
        adapted_encoding = clip_loss_models.get_image_features(adapted_img)
        target_encoding = clip_loss_models.get_image_features(trainable_img)
        edit_direction = (target_encoding - adapted_encoding)
        if edit_direction.sum() == 0:
            target_encoding = clip_loss_models.get_image_features(trainable_img + 1e-6)
            edit_direction = (target_encoding - adapted_encoding)

        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        clip_loss['adaptive_loss'] = args.adaptative_lambda * clip_loss_models.direction_loss(edit_direction, clip_loss_models.adapted_direction).mean()

    return frozen_img, trainable_img, clip_loss
# def create_print_gradient_norm_hook(layer_name, gradient_norms_list):
#     def print_gradient_norm_hook(module, grad_input, grad_output):
#         grad = grad_input[0]  # Assuming the interest is in the gradient w.r.t the output
#         norm = torch.norm(grad).item()
#         # print(f"{layer_name}: {norm:.4f}")
#         gradient_norms_list.append(norm)
#     return print_gradient_norm_hook

def train(args):

    # torch.autograd.set_detect_anomaly(True)
#################################### define src and trainable model ####################################
    # Set up networks, optimizers.
    if torch.cuda.is_available():
        # Get the current CUDA device index
        current_device = os.getenv('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"Current CUDA device index: {current_device}")
    print("Initializing networks...")
    generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier, device=device).to(device)
    generator_frozen.freeze_layers()
    generator_frozen.eval()

    generator_trainable = SG2Generator(args.train_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier, device=device).to(device)
    generator_trainable.freeze_layers()
    generator_trainable.unfreeze_layers(generator_trainable.get_training_layers(args.phase))
    generator_trainable.train()

    generator_intermediate = None


    get_parameter_number(generator_trainable, name=f'Generator-trainable')


    generator_ema = SG2Generator(args.train_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier).to(device)
    generator_ema.freeze_layers()
    generator_ema.eval()

    clip_loss_models = CLIPLoss(args, device)

    gradient_norms_G = []
    gradient_norms_CLIP = []
    directions = []
    start_adaptive = False

    # monitor gradient
    # # Gt_layer = generator_trainable.generator.convs[-1].conv
    # Gt_layer = generator_trainable.generator.to_rgbs[-1].conv
    # hook1 = Gt_layer.register_backward_hook(create_print_gradient_norm_hook("Generator Output Layer", gradient_norms_G))
    # CLIP_layer = clip_loss_models.model.visual.ln_post
    # hook2 = CLIP_layer.register_backward_hook(create_print_gradient_norm_hook("CLIP Output Layer", gradient_norms_CLIP))


    z_dim = 512

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )


    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")
    hist_dir = os.path.join(sample_dir, "hist")
    reference_dir = os.path.join('./reference_img', args.source_model_type, args.target_class)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    # os.makedirs(hist_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    random.seed(2)




    clip_model = clip_loss_models.model
    n_dim = clip_model.ln_final.weight.shape[0]

    with torch.no_grad():
        source_text_features = clip_loss_models.get_text_features(args.source_class)
        target_text_features = clip_loss_models.get_text_features(args.target_class)
        # for zs classification
        source_text_features_mean = source_text_features.mean(0, keepdim=True)
        source_text_features_mean /= source_text_features_mean.clone().norm(dim=-1, keepdim=True)
        target_text_features_mean = target_text_features.mean(0, keepdim=True)
        target_text_features_mean /= target_text_features_mean.clone().norm(dim=-1, keepdim=True)
        candidate_text_features = torch.cat((source_text_features_mean, target_text_features_mean), axis=0)

    print('source_text_features:', source_text_features.shape)
    # batch_prompt = source_text_features.shape[0]
    batch_prompt = args.batch_prompt
    if args.ctx_init != "":
        ctx_init = args.ctx_init.replace("_", " ")
        args.n_ctx = len(ctx_init.split(" "))
        prompt_prefix = ctx_init
    else:
        prompt_prefix = " ".join(["X"] * args.n_ctx)

    source_prompt_text = [prompt_prefix + " " + args.source_class]
    target_prompt_text = [prompt_prefix + " " + args.target_class]
    print("target prompts", target_prompt_text)
    source_tokenized_prompts = torch.cat(
        [clip.tokenize(p) for p in source_prompt_text]).to(device)
    target_tokenized_prompts = torch.cat(
        [clip.tokenize(p) for p in target_prompt_text]).to(device)
    source_token_embedding = clip_model.token_embedding(source_tokenized_prompts).type(clip_model.dtype)
    target_token_embedding = clip_model.token_embedding(target_tokenized_prompts).type(clip_model.dtype)
    prefix_token = target_token_embedding[:, :1, :].detach()
    suffix_token = target_token_embedding[:, 1 + args.n_ctx:, :].detach()
    prompt_token = target_token_embedding[:, 1 : 1 + args.n_ctx, :].detach()
    tokenized_prompts = target_tokenized_prompts
    # source_prefix = source_embedding[:, :1, :].detach()
    # source_suffix = source_embedding[:, -1:, :].detach()
    # source_prompt = source_embedding[:, 1: -1, :].detach()

    print(target_token_embedding.shape)
    print(prompt_token.shape)

    prompts = torch.nn.Parameter(prompt_token.clone().repeat(batch_prompt, 1, 1), requires_grad=True).to(device)
    print("Data type of prompts:", prompts.dtype)
    # prompts = torch.empty_like(target_prompt, device=device, requires_grad=True)
    # torch.nn.init.normal_(prompts, std=0.02)

    # 这里batch_mapper要和hard template的batch一样大

    print("learnable prompt shape:", prompts.shape)
    print(prompts.shape, prefix_token.shape, suffix_token.shape, tokenized_prompts.shape)
    prompt_features = compute_text_features(prompts, prefix_token, suffix_token, tokenized_prompts,
                                                            clip_model, batch_prompt)
    prompt_features = prompt_features / prompt_features.clone().norm(dim=1, keepdim=True)

    print('prompt_features:', prompt_features.shape)

    p_align_optim = torch.optim.Adam(
                    [prompts],
                    lr=args.prompt_lr,
                    betas=(0.9, 0.999),
                )
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    iterations = []
    fid = 'FID: '
    intraLPIPS = 'intraLPIPS: '

    fixed_z = torch.randn(args.n_sample, z_dim, device=device) # latent code for output
    val_z = torch.randn(50, z_dim, device=device) # latent code for evaluation
    # eval_z = val_z
    eval_z = torch.randn(500, z_dim, device=device) # latent code for evaluation
    update_z = torch.randn(1000, z_dim, device=device) # latent code for evaluation
    frozen_embeddings = torch.tensor([]).to(device)
    # eval_dist_mean, eval_dist_std, eval_dir_mean, eval_dir_std = [], [], [], []
    # eval_cos_dist_mean, eval_cos_dist_std = [], []
    # eval_gt_dist_mean, eval_gt_dist_std, eval_gt_dir_mean, eval_gt_dir_std = [], [], [], []

    # Training loop
    write_tensorboard = True
    writer = SummaryWriter(logdir=sample_dir)

    # resume training iter
    init_iter = args.init_iter or 0
    assert init_iter < args.iter

    pbar = tqdm(range(init_iter, args.iter), initial=0,
                     dynamic_ncols=True, smoothing=0.01)


    for i in pbar:
#################################### training ####################################
        sample_z = mixing_noise(args.batch, z_dim, args.mixing, device)
        layers_idx = [-1]
        if i > 0:
            generator_trainable.train()
            if i % args.LS_interval == 0 and args.auto_layer_iters > 0:
                generator_trainable.unfreeze_layers()
                train_layers, layers_idx = determine_opt_layers(args, generator_frozen, generator_trainable, clip_loss_models)

                if not isinstance(train_layers, list):
                    train_layers = [train_layers]

                generator_trainable.freeze_layers()
                generator_trainable.unfreeze_layers(train_layers)


            if generator_intermediate is not None and i>= args.start_adaptive_iteration:
                _, _, clip_loss = forward(sample_z, clip_loss_models, generator_frozen, generator_trainable, generator_intermediate, require_grad=True)
            else:
                _, _, clip_loss = forward(sample_z, clip_loss_models, generator_frozen, generator_trainable, require_grad=True)


            loss = torch.sum(torch.stack([v for v in clip_loss.values()]))

            # generator_trainable.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(g_optim)
            # scaler.update()
            generator_trainable.zero_grad()
            loss.backward()
            g_optim.step()
            ema(generator_trainable.generator, generator_ema.generator, args.ema_decay)
        else:
            generator_trainable.eval()

            _, _, clip_loss = forward(sample_z, clip_loss_models, generator_frozen, generator_trainable, require_grad=False)
            loss = torch.sum(torch.stack([v for v in clip_loss.values()]))

        if i % (args.output_interval//10) == 0:
            writer.add_scalar('loss', loss, i)
            writer.add_scalar('layer', layers_idx[0], i)
            # writer.add_scalars('D_cos',
            #                    {'P_source':D_cos[0],
            #                     'P_target':D_cos[1]},
            #                    i)
            # writer.add_histogram('trained_layer', layers_idx, i)

        bar_descrip = '; '.join([f"{k}:{v.item():3f}" for k, v in clip_loss.items()]) + ";"
        pbar.set_description(bar_descrip)

#################################### save grid image samples ####################################
        # 输出图片
        if i % args.output_interval == 0:
            generator_trainable.eval()
            # sampled_src, sampled_dst, clip_loss = forward([fixed_z], clip_loss_models, generator_frozen, generator_ema, require_grad=False, truncation=args.sample_truncation)
            sampled_src, sampled_dst, clip_loss = forward([fixed_z], clip_loss_models, generator_frozen, generator_ema, require_grad=False, truncation=args.sample_truncation)

            grid_rows = int(args.n_sample ** 0.5)




            # os.makedirs(f'{sample_dir}/{str(i).zfill(5)}/', exist_ok=True)
            # batch = 50
            # for ii in range(int(len(eval_z) / batch)):
            #     # w = g_target.style([sample_z[ii*batch: (ii+1)*batch]])
            #     sample_t, _ = generator_ema([eval_z[ii * batch: (ii + 1) * batch].data])
            #     for (num, img) in enumerate(sample_t):
            #         utils.save_image(
            #             transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(img),
            #             f'{sample_dir}/{str(i).zfill(5)}/img{str((ii * batch) + num).zfill(6)}.jpg',
            #             normalize=True,
            #         )
            # del sample_t





            if SAVE_SRC:
                save_images(sampled_src, sample_dir, "src", grid_rows, i)
            if SAVE_DST:
                save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

# #################################### write val loss ####################################
#         if write_tensorboard and i % (args.output_interval//50) == 0:
        if write_tensorboard and i % (args.output_interval // 10) == 0:
            generator_trainable.eval()
            with torch.no_grad():
                if save_img:
                    src_dir = os.path.join(args.output_dir, "sample/src")
                    out_dir = os.path.join(args.output_dir, f"sample/{iter:04d}")
                    os.makedirs(out_dir, exist_ok=True)
                val_loss = 0
                # cos_loss = 0

                sampled_src, sampled_dst, val_clip_loss = forward([val_z], clip_loss_models, generator_frozen,
                                                                  generator_trainable, require_grad=False,
                                                                  truncation=args.sample_truncation)
                loss = torch.sum(torch.stack([v for v in val_clip_loss.values()]))

                val_loss += loss

                if i == 0:
                    source_embedding = clip_loss_models.get_image_features(sampled_src)
                    prev_embeddings = source_embedding.detach().clone()



                current_embeddings = clip_loss_models.get_image_features(sampled_dst)
                text_embeddings = clip_loss_models.get_text_features(args.target_class).mean(axis=0, keepdim=True)
                text_embeddings /= text_embeddings.clone().norm(dim=-1, keepdim=True)
                distance_to_src = clip_loss_models.direction_loss(current_embeddings, source_embedding).mean()
                distance_to_prev = clip_loss_models.direction_loss(current_embeddings, prev_embeddings).mean()
                distance_to_text = clip_loss_models.direction_loss(current_embeddings, text_embeddings).mean()
                prev_embeddings = current_embeddings.detach().clone()

                direction2first = clip_loss_models.direction_loss_to_deltaT(sampled_src, sampled_dst)
                writer.add_scalar('direction loss with text direction', direction2first, i)
            del sampled_src, sampled_dst
                    # if save_img:
                    #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx + 1, len(seeds)))
                    #     if iter == 0:
                    #         src_img = (sampled_src + 1.0) * 126
                    #         save_torch_img(src_img.squeeze(0), src_dir, f'seed{seed:04d}.png')
                    #     img = (sampled_dst + 1.0) * 126
                    #     save_torch_img(img.squeeze(0), out_dir, f'seed{seed:04d}.png')

            for k, v in val_clip_loss.items():
                writer.add_scalar('val_'+k, v, i)
            writer.add_scalar('val_loss', val_loss, i)
            writer.add_scalar('distance_to_src', distance_to_src, i)
            writer.add_scalar('distance_to_prev', distance_to_prev, i)
            writer.add_scalar('distance_to_text', distance_to_text, i)
        # if i == 300:
        #     from evaluation.generate import generate_imgs_4FIDnISnLPIPS, generate_img_pairs, generate_imgs_SIFID
        #     evaluation_batch = 20
        #     args.SIFID_sample = 1000
        #     args.latent = 512
        #     args.save_target = './temp_img/imgs'
        #     temp_image_path = './temp_img/imgs'
        #     args.img_pth = temp_image_path
        #     os.makedirs(temp_image_path, exist_ok=True)
        #     generate_imgs_SIFID(args, generator_trainable, batch=evaluation_batch)





#################################### evaluation ####################################

        # if  args.evaluation_in_training and (i % args.evaluation_interval == 0 or (i < args.evaluation_interval and i % (args.evaluation_interval//4) == 0)):
        if args.evaluation_in_training and i in args.evaluation_interval:
            # print(i, args.evaluation_interval, i % args.evaluation_interval)
            from evaluation.generate import generate_imgs_4FIDnISnLPIPS, generate_img_pairs, generate_imgs_SIFID
            from evaluation.fid_score import calculate_fid_given_paths
            from evaluation.inception_score import calculate_is_given_paths
            from evaluation.IntraLPIPS_kmedoids import calculate_lpips_given_paths
            from evaluation.hed_model import HED_Network
            from evaluation.scs_score import SCS_eval
            from evaluation.sifid_score import calculate_sifid_given_paths
            from evaluation.CLIP_global import calculate_global_loss_given_paths
            from evaluation.CLIP_similarity import calculate_CLIPsim_given_paths
            import shutil

            if args.fid or args.globalCLIP or args.CLIPsim:
                args.FIDnISnLPIPS_sample = 5000
            if args.intraLPIPS:
                args.LPIPS_sample = 250
            args.eval_sample = 1000
            args.IS_sample=5000
            args.global_sample = 5000
            args.SCS_samples=500
            args.SIFID_sample=1000
            args.latent=512
            temp_image_path = f'./temp_img_{current_device}/imgs'
            args.save_target= f'./temp_img_{current_device}/imgs'
            args.center_path = "./evaluation/cluster_centers/"

            args.img_pth = temp_image_path
            evaluation_batch = 50
            if os.path.exists(temp_image_path):
                shutil.rmtree(temp_image_path)

            fp = open(os.path.join(args.output_dir,'sample/quantitative.txt'), 'a+')
            fp.write(f'iter {i}\n')
            fp.close()

            os.makedirs(temp_image_path, exist_ok=True)
            if args.fid or args.IS or args.globalCLIP or args.intraLPIPS or args.CLIPsim:
                generate_imgs_4FIDnISnLPIPS(args, generator_trainable, batch=evaluation_batch)
            if args.fid:
                fid_value = calculate_fid_given_paths([f'/home/user/Desktop/DiskSDA/dataset/{args.target_class}/{args.target_class}', args.save_target],
                                                      batch_size=evaluation_batch, device = device, dims = 2048, num_workers=8)
                print('FID: %.4f' % fid_value)
                writer.add_scalar('FID', fid_value, i)
                fid = fid + '%.4f, ' % fid_value
                fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
                fp.write('FID: %.4f \n' % fid_value)
                fp.close()

            if args.IS:
                IS_mean = calculate_is_given_paths(args)
                writer.add_scalar('IS', IS_mean, i)
                print('Inception score: %.4f' % IS_mean)
                fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
                fp.write('Inception score: %.4f \n' % IS_mean)
                fp.close()

            if args.intraLPIPS:
                lpips_mean = calculate_lpips_given_paths(args)
                print('intraLPIPS: %.4f' % lpips_mean)
                writer.add_scalar('intraLPIPS', lpips_mean, i)
                intraLPIPS = intraLPIPS + '%.4f, ' % lpips_mean
                fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
                fp.write('intraLPIPS: %.4f \n' % lpips_mean)
                fp.close()

            if args.globalCLIP:
                globalCLIP_mean, globalCLIP_std = calculate_global_loss_given_paths(args, text=args.target_class,
                                                                                    batch_size=evaluation_batch,
                                                                                    save_image_embeddings=True)
                print('globalCLIP: %.4f, %.4f' % (globalCLIP_mean, globalCLIP_std))
                writer.add_scalar('globalCLIP', globalCLIP_mean, i)
                fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
                fp.write('globalCLIP: %.4f, %.4f \n' % (globalCLIP_mean, globalCLIP_std))
                fp.close()

            if args.CLIPsim:
                CLIPsim_mean, CLIPsim_std = calculate_CLIPsim_given_paths(args, trg_img_dir=reference_dir,
                                                                              batch_size=evaluation_batch,
                                                                              save_image_embeddings=False)
                print('CLIPsim: %.4f, %.4f' % (CLIPsim_mean, CLIPsim_std))
                writer.add_scalar('CLIPsim', CLIPsim_mean, i)
                fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
                fp.write('CLIPsim: %.4f, %.4f \n' % (CLIPsim_mean, CLIPsim_std))
                fp.close()

            if args.sifid:
                generate_imgs_SIFID(args, generator_trainable, batch=evaluation_batch)
                sifid_values = calculate_sifid_given_paths(f'./reference/{args.target_class}', args.save_target)
                # sifid_values = np.asarray(sifid_values, dtype=np.float32)
                # print('SIFID: ', sifid_values)
                # print(type(sifid_values), len(sifid_values))
                writer.add_scalar('SIFID', np.array(sifid_values).mean(), i)
                fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
                for value in sifid_values:
                    fp.write('SIFID: %.4f \n' % value)
                fp.close()
            shutil.rmtree(temp_image_path)

            os.makedirs(temp_image_path, exist_ok=True)
            args.save_source = os.path.join(temp_image_path, 'source')
            args.save_target = os.path.join(temp_image_path, 'target')
            os.makedirs(args.save_source, exist_ok=True)
            os.makedirs(args.save_target, exist_ok=True)
            if args.scs:
                generate_img_pairs(args, generator_frozen, generator_trainable, batch=evaluation_batch)
                HED_net = HED_Network().cuda().eval()
                SCS_Score = SCS_eval(args, HED_net)
                writer.add_scalar('SCS', SCS.item(), i)
                print('SCS Score: %.4f' % SCS_Score.item())
                fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
                fp.write('SCS Score: %.4f \n' % SCS_Score.item())
                fp.close()
            shutil.rmtree(temp_image_path)

            torch.save(
                {
                    "g_ema": generator_ema.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )

            os.makedirs(f'{sample_dir}/{str(i).zfill(5)}/', exist_ok=True)
            batch = 50
            for ii in range(int(len(eval_z) / batch)):
                # w = g_target.style([sample_z[i*batch: (i+1)*batch]])
                sample_t, _ = generator_ema([eval_z[ii * batch: (ii + 1) * batch].data], truncation=args.sample_truncation)
                for (num, img) in enumerate(sample_t):
                    utils.save_image(
                        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(img),
                        f'{sample_dir}/{str(i).zfill(5)}/img{str((ii * batch) + num).zfill(6)}.jpg',
                        normalize=True,
                    )
            del sample_t


        if args.adaptative_direction and i!=0 and i % args.adaptative_interval==0:

            s_time = time.time()
            if i >= args.start_adaptive_iteration and not start_adaptive:
            # if not start_adaptive:
                print('Exceed iter threshold, start using adaptive direction ...')
                start_adaptive = True
                prev_generator_frozen = generator_frozen
                last_prompt_features = source_text_features

            batch_size = 50

            with torch.no_grad():
                frozen_embeddings = torch.tensor([]).to(device)
                adapted_embeddings = torch.tensor([]).to(device)
                # if len(frozen_embeddings) < 1:
                # #     print('both')

                for ii in range(0, len(update_z), batch_size):
                    if args.use_last_anchor and start_adaptive: # use previous anchor generator as source generator
                        frozen_img, adapted_img, clip_loss = forward([update_z[ii:ii + batch_size]], clip_loss_models, prev_generator_frozen,
                                                                      generator_trainable,
                                                                      require_grad=False, truncation=args.sample_truncation)
                    else:
                        frozen_img, adapted_img, clip_loss = forward([update_z[ii:ii + batch_size]], clip_loss_models,
                                                                     generator_frozen,
                                                                     generator_trainable,
                                                                     require_grad=False, truncation=args.sample_truncation)
                    frozen_encoding = clip_loss_models.get_image_features(frozen_img)
                    adapted_encoding = clip_loss_models.get_image_features(adapted_img)
                    frozen_embeddings = torch.cat((frozen_embeddings, frozen_encoding), dim=0)
                    adapted_embeddings = torch.cat((adapted_embeddings, adapted_encoding), dim=0)
                del frozen_img, adapted_img
                # else:
                #     for ii in range(0, len(update_z), batch_size):
                #         _, adapted_img, clip_loss = forward([update_z[ii:ii + batch_size]], clip_loss_models,
                #                                                      generator_frozen,
                #                                                      generator_trainable,
                #                                                      require_grad=False,
                #                                                      truncation=args.sample_truncation)
                #         adapted_encoding = clip_loss_models.get_image_features(adapted_img)
                #         adapted_embeddings = torch.cat((adapted_embeddings, adapted_encoding), dim=0)
                #     del adapted_img
            probs = (100.0 * adapted_embeddings @ candidate_text_features.T).softmax(dim=-1).mean(0)
            print('Adapted image zero-shot:', probs)
            if probs[-1] >= args.probs_threshold and not start_adaptive:
                print('Exceed prob threshold, start using adaptive direction ...')
                start_adaptive = True
                prev_generator_frozen = generator_frozen
                last_prompt_features = source_text_features


            # adapted delta_I
            if not start_adaptive:
                continue
            adapted_i_directions = adapted_embeddings - frozen_embeddings
            adapted_i_directions /= adapted_i_directions.clone().norm(dim=-1, keepdim=True)
            adapted_img_directions = adapted_i_directions.mean(0, keepdim=True).to(device)
            adapted_img_directions /= adapted_img_directions.clone().norm(dim=-1, keepdim=True)





            # redefine the prompt
            if args.redefine_prompt:
                print('Redefining new prompt ...')
                alpha = i/args.iter
                # print(source_tokenized_prompts)
                # print(target_tokenized_prompts)
                # tokenized_prompts = (1-alpha)*source_tokenized_prompts + alpha*target_tokenized_prompts
                # print(tokenized_prompts)
                # token_embedding = clip_model.token_embedding(tokenized_prompts.int()).type(clip_model.dtype)
                token_embedding = (1 - alpha) * source_token_embedding + alpha * target_token_embedding
                prefix_token = token_embedding[:, :1, :].detach()
                suffix_token = token_embedding[:, 1 + args.n_ctx:, :].detach()
                prompt_token = token_embedding[:, 1: 1 + args.n_ctx, :].detach()
                # source_prefix = source_embedding[:, :1, :].detach()
                # source_suffix = source_embedding[:, -1:, :].detach()
                # source_prompt = source_embedding[:, 1: -1, :].detach()

                # print(token_embedding.shape)
                # print(prompt_token.shape)

                prompts = torch.nn.Parameter(prompt_token.clone().repeat(batch_prompt, 1, 1), requires_grad=True).to(device)
                print("Data type of prompts:", prompts.dtype)
                print("learnable prompt shape:", prompts.shape)

                p_align_optim = torch.optim.Adam(
                    [prompts],
                    lr=args.prompt_lr,
                    betas=(0.9, 0.999),
                )






            # aligning direction
            pbar_align = tqdm(range(args.prompt_align_iter), initial=0,
                        dynamic_ncols=True, smoothing=0.01)
            for _ in pbar_align:
                # adapted delta_T
                # print(prompts.shape, prefix_token.shape, suffix_token.shape, tokenized_prompts.shape)
                prompt_features = compute_text_features(prompts, prefix_token, suffix_token, tokenized_prompts,
                                                        clip_model, batch_prompt)
                prompt_features = prompt_features / prompt_features.clone().norm(dim=1, keepdim=True)
                # print('prompt_features:', prompt_features.shape)
                if prompt_features.shape[0] != 1 and prompt_features.shape[0] != source_text_features.shape[0]:
                    prompt_features = prompt_features.mean(0, keepdim=True)
                    prompt_features = prompt_features / prompt_features.clone().norm(dim=1, keepdim=True)
                if args.use_last_anchor:
                    # print(prompt_features.shape)
                    adapted_t_directions = prompt_features - last_prompt_features
                else:
                    adapted_t_directions = prompt_features - source_text_features
                if adapted_t_directions.sum() == 0:
                    prompt_features = compute_text_features(prompts + 1e-6, prefix_token, suffix_token, tokenized_prompts,
                                                                   clip_model, batch_prompt)
                    prompt_features = prompt_features / prompt_features.clone().norm(dim=1,
                                                                                                          keepdim=True)
                    # print('target_prompt_features:', target_prompt_features.shape)
                    if prompt_features.shape[0] != 1 and prompt_features.shape[0] != \
                            source_text_features.shape[0]:
                        prompt_features = prompt_features.mean(0, keepdim=True)
                        prompt_features = prompt_features / prompt_features.clone().norm(dim=1, keepdim=True)
                    if args.use_last_anchor:
                        adapted_t_directions = prompt_features - last_prompt_features
                    else:
                        adapted_t_directions = prompt_features - source_text_features

                adapted_t_directions = adapted_t_directions / adapted_t_directions.clone().norm(dim=-1, keepdim=True)
                adapted_txt_directions = adapted_t_directions.mean(0, keepdim=True).to(device)
                adapted_txt_directions = adapted_txt_directions / adapted_txt_directions.clone().norm(dim=-1, keepdim=True)
                # print(adapted_img_directions)
                # print('prompts:', torch.isinf(prompts).any(), '\n', prompts)
                # print(target_prompt_features)
                # print(adapted_txt_directions)

                # prompt_align_loss = clip_loss_models.direction_loss(adapted_img_directions, adapted_txt_directions).mean()
                # prompt_distance_loss = clip_loss_models.direction_loss(target_text_features_mean, prompt_features).mean()
                # prompt_loss = prompt_align_loss + prompt_distance_loss
                # pbar_align.set_description('prompt_align_loss: %.4f, prompt_distance_loss: %.4f' % (prompt_align_loss, prompt_distance_loss))
                prompt_loss = clip_loss_models.direction_loss(adapted_img_directions, adapted_txt_directions).mean()
                pbar_align.set_description('prompt_align_loss: %.4f' % prompt_loss)

                # p_align_optim.zero_grad()
                # scaler.scale(prompt_align_loss).backward()
                # scaler.step(p_align_optim)
                # scaler.update()

                # print('prompt_align_loss: %.4f' % prompt_align_loss)
                p_align_optim.zero_grad()
                # clip_model.zero_grad()
                prompt_loss.backward()
                # torch.nn.utils.clip_grad_norm_(prompts, max_norm=10, norm_type=2)
                # print('prompts graident:', '\n', prompts.grad)
                p_align_optim.step()
            # # update direction
            with torch.no_grad():
                prompt_features = compute_text_features(prompts, prefix_token, suffix_token, tokenized_prompts,
                                                               clip_model, batch_prompt)
                prompt_features = prompt_features / prompt_features.clone().norm(dim=1, keepdim=True)
                # print(prompt_features)
                # print('target_prompt_features:', target_prompt_features.shape)
                if prompt_features.shape[0] != 1 and prompt_features.shape[0] != target_text_features.shape[0]:
                    prompt_features = prompt_features.mean(0, keepdim=True)
                    prompt_features = prompt_features / prompt_features.clone().norm(dim=1, keepdim=True)

                adapted_t_directions = target_text_features - prompt_features
                adapted_t_directions = adapted_t_directions / adapted_t_directions.clone().norm(dim=1, keepdim=True)
                adapted_txt_directions = adapted_t_directions.mean(0, keepdim=True).to(device)
                clip_loss_models.adapted_direction = adapted_txt_directions / adapted_txt_directions.clone().norm(dim=-1, keepdim=True)
                print("Distance to text direction %.4f" % clip_loss_models.direction_loss(clip_loss_models.first_target_direction, clip_loss_models.adapted_direction))

                # update the trainable generator to frozen
                # model1_state_dict = generator_trainable.state_dict()
                # model2_state_dict = {name: tensor.clone() for name, tensor in model1_state_dict.items()}
                # generator_frozen.load_state_dict(model2_state_dict)
                # generator_frozen.freeze_layers()
                # generator_frozen.eval()

                probs = (100.0 * prompt_features @ candidate_text_features.T).softmax(dim=-1).mean(0)
                print('Adapted text zero-shot:', probs)

                print('renewing intermediate generator')
                generator_intermediate = deepcopy(generator_trainable)
                generator_intermediate.freeze_layers()
                generator_intermediate.eval()

                if args.use_last_anchor:
                    prev_generator_frozen = deepcopy(generator_intermediate)
                    prev_generator_frozen.freeze_layers()
                    prev_generator_frozen.eval()
                    last_prompt_features = deepcopy(prompt_features)

            e_time = time.time()
            ss_time = e_time-s_time
            # print('%.3f'%ss_time)

                # generator_trainable = deepcopy(generator_ema)
                # generator_trainable.freeze_layers()
                # generator_trainable.unfreeze_layers(generator_trainable.get_training_layers(args.phase))
                # generator_trainable.train()
                # source_text_features = prompt_features.clone()



#################################### checkpoint ####################################
        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):

            torch.save(
                {
                    "g_ema": generator_ema.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )

        # if  i % args.eval_interval == 0 or (i < args.eval_interval and i % (args.eval_interval//4) == 0):
        # # if i % args.eval_interval == 0 and i != 0:
        #     os.makedirs(f'./eval_FIDnISnLPIPS/{args.target_class}/iter{str(i)}', exist_ok=True)
        #     os.makedirs(args.output_dir + f'/fid/{i}/', exist_ok=True)
        #     ## Evaluate direction and distance to mean
        #     iterations.append(i)
        #     ###
        #     batch_size = 20
        #     frozen_embeddings = torch.tensor([])
        #     adapted_embeddings = torch.tensor([])
        #     with torch.no_grad():
        #         for ii in range(0, len(eval_z), batch_size):
        #             w_styles = generator_frozen.style([eval_z[ii:ii + batch_size]])
        #             # frozen_img = generator_frozen(w_styles, input_is_latent=True, truncation=args.sample_truncation)[0]
        #             adapted_img = generator_ema(w_styles, input_is_latent=True, truncation=args.sample_truncation)[0]
        #             # frozen_encoding = clip_loss_models.get_image_features(frozen_img)
        #             # adapted_encoding = clip_loss_models.get_image_features(adapted_img)
        #             # frozen_embeddings = torch.cat((frozen_embeddings, frozen_encoding.cpu()), dim=0)
        #             # adapted_embeddings = torch.cat((adapted_embeddings, adapted_encoding.cpu()), dim=0)
        #             for (num, img) in enumerate(adapted_img):
        #                 utils.save_image(
        #                     img,
        #                     # f'./eval_FIDnISnLPIPS/{args.target_class}/iter{str(i)}/img{str(ii + num).zfill(5)}.png',
        #                     args.output_dir + f'/fid/{i}/{str(ii + num).zfill(5)}.jpg',
        #                     normalize=True,
        #                     range=(-1, 1),
        #                 )
    #         # save distance hist
    #         cosine_similarity = torch.mm(adapted_embeddings, adapted_embeddings.t())
    #         cosine_distance = 1 - cosine_similarity
    #         upper_triangle = torch.triu(cosine_distance, diagonal=1)
    #         eval_distances = upper_triangle[upper_triangle > 0].cpu()
    #         eval_dist_mean.append(eval_distances.mean().item())
    #         eval_dist_std.append(eval_distances.std().item())
    #
    #         eval_cos_distances = 1.0 - torch.nn.CosineSimilarity()(frozen_embeddings, adapted_embeddings)
    #         eval_cos_dist_mean.append(eval_cos_distances.mean().item())
    #         eval_cos_dist_std.append(eval_cos_distances.std().item())
    #
    #         # plt.figure()
    #         # sns.histplot(data=eval_distances.numpy(), bins=np.arange(100) / 100, stat="density", color='lightskyblue')
    #         # plt.xlabel(f"iter {i} in-domain distance", fontsize=20)
    #         # plt.ylabel("Frequency", fontsize=20)
    #         # plt.savefig(f"{hist_dir}/dis_iter{i:04d}.png")
    #
    #         # # save direction hist
    #         # direction = adapted_embeddings - frozen_embeddings
    #         # if clip_loss_models.target_direction is None:
    #         #     with torch.no_grad():
    #         #         clip_loss_models.clip_directional_loss(sampled_src, args.source_class, sampled_dst, args.target_class)
    #         # eval_dir = clip_loss_models.direction_loss(direction.cuda(), clip_loss_models.target_direction).cpu()
    #         # eval_dir_mean.append(eval_dir.mean().item())
    #         # eval_dir_std.append(eval_dir.std().item())
    #         #
    #         # # plt.figure()
    #         # # sns.histplot(data=eval_dir.numpy(), bins=np.arange(200) / 100, stat="density", color='#FFA500')
    #         # # plt.xlabel(f"iter {i} pairwise direction", fontsize=20)
    #         # # plt.ylabel("Frequency", fontsize=20)
    #         # # plt.savefig(f"{hist_dir}/dir_iter{i:04d}.png")
    #         #
    #         # # if not args.gt:
    #         # #
    #         # #     # save global distance
    #         # #     if clip_loss_models.global_target is None:
    #         # #         with torch.no_grad():
    #         # #             clip_loss_models.get_global_target(args.target_class)
    #         # #     eval_gt_dist = clip_loss_models.direction_loss(adapted_embeddings.cuda(), clip_loss_models.global_target).cpu()
    #         # #     eval_gt_dist_mean.append(eval_gt_dist.mean().item())
    #         # #     eval_gt_dist_std.append(eval_gt_dist.std().item())
    #         # #
    #         # #     if clip_loss_models.gt_direction is None:
    #         # #         with torch.no_grad():
    #         # #             clip_loss_models.get_gt_dir(args.target_class)
    #         # #     eval_gt_dir = clip_loss_models.direction_loss(direction.cuda(), clip_loss_models.gt_direction).cpu()
    #         # #     eval_gt_dir_mean.append(eval_gt_dir.mean().item())
    #         # #     eval_gt_dir_std.append(eval_gt_dir.std().item())
    #         # #
    #         # #     # plt.figure()
    #         # #     # sns.histplot(data=eval_gt_dir.numpy(), bins=np.arange(200) / 100, stat="density", color='lightseagreen')
    #         # #     # plt.xlabel(f"iter {i} pairwise direction", fontsize=20)
    #         # #     # plt.ylabel("Frequency", fontsize=20)
    #         # #     # plt.savefig(f"{hist_dir}/gt_dir_iter{i:04d}.png")
    #         # #     del eval_gt_dir, eval_gt_dist
    #         # # del frozen_embeddings, adapted_embeddings, frozen_img, adapted_img, frozen_encoding, adapted_encoding, \
    #         # #     cosine_similarity, cosine_distance, upper_triangle, eval_distances, direction, eval_dir
    #
    # ######### plot eval figure #########
    # np.save(f"{hist_dir}/eval_dist_mean.npy", eval_dist_mean)
    # np.save(f"{hist_dir}/eval_dist_std.npy", eval_dist_std)
    # np.save(f"{hist_dir}/eval_cos_dist_mean.npy", eval_cos_dist_mean)
    # np.save(f"{hist_dir}/eval_cos_dist_std.npy", eval_cos_dist_std)
    # # np.save(f"{hist_dir}/eval_dir_mean.npy", eval_dir_mean)
    # # np.save(f"{hist_dir}/eval_dir_std.npy", eval_dir_std)
    # # if not args.gt:
    # #     np.save(f"{hist_dir}/eval_gt_dir_mean.npy", eval_gt_dir_mean)
    # #     np.save(f"{hist_dir}/eval_gt_dir_std.npy", eval_gt_dir_std)
    # #     np.save(f"{hist_dir}/eval_gt_dist_mean.npy", eval_gt_dist_mean)
    # #     np.save(f"{hist_dir}/eval_gt_dist_std.npy", eval_gt_dist_std)
    #
    # # print(eval_dist_mean, '\n', eval_dist_std, '\n', eval_dir_mean, '\n', eval_dir_std, '\n',
    # #       eval_gt_dir_mean, '\n', eval_gt_dir_std, '\n', eval_gt_dist_mean, '\n', eval_gt_dist_std, '\n')
    #
    # plt.figure(figsize=(10, 6))
    # sns.set_theme(style='whitegrid')
    # sns.lineplot(x=iterations, y=eval_dist_mean, label='mean', color='deepskyblue', marker='o')
    # plt.fill_between(iterations, [m - s for m, s in zip(eval_dist_mean, eval_dist_std)], [m + s for m, s in zip(eval_dist_mean, eval_dist_std)],
    #                  color='turquoise', alpha=0.2, label='std')
    # plt.title("In-domain sample distance", fontsize=20)
    # plt.xlabel("iters", fontsize=20)
    # plt.ylabel("Cos Distance", fontsize=20)
    # plt.legend()
    # plt.xlim(iterations[0], iterations[-1])
    # plt.ylim(bottom=0)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"{hist_dir}/dist_iter.png")
    #
    # plt.figure(figsize=(10, 6))
    # sns.set_theme(style='whitegrid')
    # sns.lineplot(x=iterations, y=eval_cos_dist_mean, label='mean', color='darkorange', marker='o')
    # plt.fill_between(iterations, [m - s for m, s in zip(eval_cos_dist_mean, eval_cos_dist_std)], [m + s for m, s in zip(eval_cos_dist_mean, eval_cos_dist_std)],
    #                  color='orange', alpha=0.2, label='std')
    # plt.title("Cross-domain sample distance", fontsize=20)
    # plt.xlabel("iters", fontsize=20)
    # plt.ylabel("Cos Distance", fontsize=20)
    # plt.legend()
    # plt.xlim(iterations[0], iterations[-1])
    # plt.ylim(bottom=0)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"{hist_dir}/cos_dist_iter.png")
    #
    # # plt.figure(figsize=(10, 6))
    # # sns.set_theme(style='whitegrid')
    # # sns.lineplot(x=iterations, y=eval_dir_mean, label='mean', color='darkorange', marker='o')
    # # plt.fill_between(iterations, [m - s for m, s in zip(eval_dir_mean, eval_dir_std)], [m + s for m, s in zip(eval_dir_mean, eval_dir_std)],
    # #                  color='orange', alpha=0.2, label='std')
    # # plt.title("Cross domain directional loss", fontsize=20)
    # # plt.xlabel("iters", fontsize=20)
    # # plt.ylabel("Cos Distance", fontsize=20)
    # # plt.legend()
    # # plt.xlim(iterations[0], iterations[-1])
    # # plt.ylim(bottom=0)
    # # plt.grid(True)
    # # plt.tight_layout()
    # # plt.savefig(f"{hist_dir}/dir_iter.png")
    # #
    # # if not args.gt:
    # #     plt.figure(figsize=(10, 6))
    # #     sns.set_theme(style='whitegrid')
    # #     sns.lineplot(x=iterations, y=eval_gt_dist_mean, label='mean', color='darkslategrey', marker='o')
    # #     plt.fill_between(iterations, [m - s for m, s in zip(eval_gt_dist_mean, eval_gt_dist_std)],
    # #                      [m + s for m, s in zip(eval_gt_dist_mean, eval_gt_dist_std)],
    # #                      color='teal', alpha=0.2, label='std')
    # #     plt.title("Global loss (gt)", fontsize=20)
    # #     plt.xlabel("iters", fontsize=20)
    # #     plt.ylabel("Cos Distance", fontsize=20)
    # #     plt.legend()
    # #     plt.xlim(iterations[0], iterations[-1])
    # #     plt.ylim(bottom=0)
    # #     plt.grid(True)
    # #     plt.tight_layout()
    # #     plt.savefig(f"{hist_dir}/gt_dist_iter.png")
    # #
    # #     plt.figure(figsize=(10, 6))
    # #     sns.set_theme(style='whitegrid')
    # #     sns.lineplot(x=iterations, y=eval_gt_dir_mean, label='mean', color='darkgoldenrod', marker='o')
    # #     plt.fill_between(iterations, [m - s for m, s in zip(eval_gt_dir_mean, eval_gt_dir_std)], [m + s for m, s in zip(eval_gt_dir_mean, eval_gt_dir_std)],
    # #                      color='darkkhaki', alpha=0.2, label='std')
    # #     plt.title("Cross domain directional loss (gt)", fontsize=20)
    # #     plt.xlabel("iters", fontsize=20)
    # #     plt.ylabel("Cos Distance", fontsize=20)
    # #     plt.legend()
    # #     plt.xlim(iterations[0], iterations[-1])
    # #     plt.ylim(bottom=0)
    # #     plt.grid(True)
    # #     plt.tight_layout()
    # #     plt.savefig(f"{hist_dir}/gt_dir_iter.png")


    # with open(os.path.join(args.output_dir, 'sample/grad_G'), 'wb') as f:
    #     pickle.dump(gradient_norms_G, f)
    # with open(os.path.join(args.output_dir, 'sample/grad_CLIP'), 'wb') as f:
    #     pickle.dump(gradient_norms_CLIP, f)
    with open(os.path.join(args.output_dir, 'sample/directions.pkl'), 'wb') as f:
        pickle.dump(directions, f)
# writer.export_scalars_to_json("./loss.json")
    fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
    fp.write(fid)
    fp.close()
    fp = open(os.path.join(args.output_dir, 'sample/quantitative.txt'), 'a+')
    fp.write(intraLPIPS)
    fp.close()
    writer.flush()
    writer.close()
#################################### plot image same in the paper ####################################
    # default 0
    for i in range(args.num_grid_outputs):
        generator_trainable.eval()

        with torch.no_grad():
            sample_z = mixing_noise(16, z_dim, 0, device)
            generator_trainable.eval()
            with torch.no_grad():
                w_styles = generator_frozen.style(sample_z)
                sampled_dst = generator_trainable(w_styles, input_is_latent=True, truncation=args.sample_truncation)[0]

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

        save_paper_image_grid(sampled_dst, sample_dir, f"sampled_grid_{i}.png")


if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()

    if args.gt:
        gt = '_gt'
    else:
        gt = ''
    if args.adaptative_direction:
        adaptdir = '_adaptdir' + str(args.adaptative_lambda)
    else:
        adaptdir = ''

    if args.pca:
        pca = '_pca'
    else:
        pca = ''
    if args.onedir:
        onedir = '_onedir'
    else:
        onedir = ''
    if args.oneshot:
        oneshot = '_oneshot'
    else:
        oneshot = ''
    if args.tenshot:
        tenshot = '_tenshot'
    else:
        tenshot = ''

    lambda_dict = {
        '_gb': args.lambda_global,
        '_dir': args.lambda_direction,
        '_do': args.lambda_domain,
        '_con': args.lambda_const,
        '_KL': args.lambda_KL,
        '_pt': args.lambda_proto,
        '_dis': args.lambda_dis,
        '_prompt': args.lambda_prompt,
        '_adtvvintvl': args.adaptative_interval,
        '_probs_threshold': args.probs_threshold,
        '_sai': args.start_adaptive_iteration,
        '_#prompt': args.batch_prompt,
        '_mixing': args.mixing,
        '_use_last_anchor': args.use_last_anchor,
        'redefine_prompt': args.redefine_prompt,

    }
    lambda_loss = "".join(f"{key}{value}" for key, value in lambda_dict.items() if value != 0.0)

    args.output_dir = args.output_dir + args.source_model_type + '/' + args.source_class + '2' + args.target_class + '/' + \
                      oneshot + onedir + tenshot + gt + adaptdir + pca + \
                      lambda_loss + '_lf_iter' + str(args.auto_layer_iters) + '_layer' + str(args.auto_layer_k)

    # save snapshot of code / args before training.
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    copytree("criteria/", os.path.join(args.output_dir, "code", "criteria"), )
    # shutil.copy2("model/ZSSGAN.py", os.path.join(args.output_dir, "code", "ZSSGAN.py"))

    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    