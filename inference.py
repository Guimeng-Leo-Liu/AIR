import warnings
warnings.filterwarnings("ignore")

import argparse
import random

from matplotlib.image import imsave
import torch
import torch.nn as nn
from torchvision import utils
# from model import Generator, Projection_module, Projection_module_church
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import torchvision.transforms as transforms
import numpy as np
import math

def generate_gif(args, g_source, g_target):
    if args.load_noise:
        noise = torch.load(args.load_noise).cuda()
        
    else:
        noise = torch.randn(args.n_sample, args.latent).cuda()

    with torch.no_grad():

        n_steps = args.n_steps
        step = float(1)/n_steps
        n_paths = noise.size(0)
        for t in range(n_paths):
            print(t)
            if t != (n_paths - 1):
                z1, z2 = torch.unsqueeze(
                    noise[t], 0), torch.unsqueeze(noise[t+1], 0)
            else:
                z1, z2 = torch.unsqueeze(
                    noise[t], 0), torch.unsqueeze(noise[0], 0)

            for i in range(n_steps):
                alpha = step*i
                z = z2*alpha + (1-alpha)*z1
                sample_s, _ = g_source([z], randomize_noise=False)
                w = [g_target.module.style(z)]
                # w = [Proj_module.modulate(item) for item in w]
                sample_t, _= g_target(w, input_is_latent=True, randomize_noise=False)

                utils.save_image(
                    transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(sample_s),
                    f'%s/sample%d.jpg' % (args.save_source, (t*n_steps) + i) ,
                    normalize=True,
                )

                utils.save_image(
                    transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(sample_t),
                    f'%s/sample%d.jpg' % (args.save_target, (t*n_steps) + i),
                    normalize=True,
                )


def generate_imgs(args, g_source, g_target):

    with torch.no_grad():
        
        if args.load_noise:
            sample_z = torch.load(args.load_noise)
        else:
            sample_z = torch.randn(args.n_sample, args.latent).cuda()

        sample_s, _ = g_source([sample_z], input_is_latent=False, randomize_noise=False)
        w = [g_target.module.style(sample_z)]
        # w = [Proj_module.modulate(item) for item in w]
        sample_t, _= g_target(w, input_is_latent=True, randomize_noise=False)

        utils.save_image(
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(sample_s),
            f'%s/sample_s.jpg' % args.save_source,
            nrow=5,
            normalize=True,
        )

        utils.save_image(
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(sample_t),
            f'%s/sample_t.jpg' % args.save_target,
            nrow=5,
            normalize=True,
        )


def generate_img_pairs(args, g_source, g_target, batch=50):
    
    with torch.no_grad():
        sample_z = torch.randn(args.SCS_samples, args.latent).cuda()
        for i in range(int(args.SCS_samples / batch)):
            w = g_source.style([sample_z[i * batch: (i + 1) * batch]])
            sample_t, _ = g_target(w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)
            sample_s, _ = g_source(w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)

            for (num, (img_s, img_t)) in enumerate(zip(sample_s, sample_t)):
                utils.save_image(
                    transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(img_s),
                    f'%s/img%d.jpg' % (args.save_source, (i* batch) + num) ,
                    normalize=True,
                )

                utils.save_image(
                    transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(img_t),
                    f'%s/img%d.jpg' % (args.save_target, (i * batch) + num) ,
                    normalize=True,
                )


def generate_imgs_SIFID(args, g_target, batch=50):
    with torch.no_grad():

        sample_z = torch.randn(args.SIFID_sample, args.latent).cuda()
        for i in range(int(args.SIFID_sample / batch)):
            w = g_target.style([sample_z[i * batch: (i + 1) * batch]])
            sample_t, _ = g_target(w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)

            for (num, img) in enumerate(sample_t):
                utils.save_image(
                    transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(img),
                    f'%s/img%d.jpg' % (args.save_target, (i * batch) + num),
                    normalize=True,
                )

def generate_imgs_4FIDnISnLPIPS(args, g_target, batch=50, latent_code=None):
    
    with torch.no_grad():
        if not args.FIDnISnLPIPS_sample:
            if args.LPIPS_sample:
                args.FIDnISnLPIPS_sample = args.LPIPS_sample
            elif args.eval_sample:
                args.FIDnISnLPIPS_sample = args.eval_sample
        if latent_code is not None:
            sample_z = latent_code
        else:
            sample_z = torch.randn(args.FIDnISnLPIPS_sample, args.latent).cuda()
        for i in range(math.ceil(sample_z.shape[0] / batch)):
            # w = g_target.style([sample_z[0][i*batch: (i+1)*batch]])
            print(sample_z[i*batch: (i+1)*batch].shape)
            sample_t, _= g_target([sample_z[i*batch: (i+1)*batch]], input_is_latent=True, truncation=args.sample_truncation)

            for (num, img) in enumerate(sample_t):
                utils.save_image(
                    transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])(img),
                    f'%s/img%d.jpg' % (args.save_target, (i * batch) + num) ,
                    normalize=True,
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--SCS_samples', type=int, default=500, help='number of image pairs to eval SCS')
    parser.add_argument('--n_sample', type=int, default=25, help='number of fake images to be sampled')
    parser.add_argument('--SIFID_sample', type=int, default=1000, help='number of fake images to be sampled for SIFID')
    parser.add_argument('--FIDnISnLPIPS_sample', type=int, default=1000, help='number of fake images to be sampled for FID and IS and LPIPS')
    parser.add_argument('--n_steps', type=int, default=40, help="determines the granualarity of interpolation")
    parser.add_argument('--ckpt_source', type=str, default=None)
    parser.add_argument('--ckpt_target', type=str, default=None)
    parser.add_argument('--mode', type=str, default='viz_imgs', help='viz_imgs,viz_gif,eval_IS,eval_SCS')
    parser.add_argument('--method', type=str,)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--tar_model', type=str,)
    parser.add_argument('--load_noise', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--target', type=str, default='VanGogh', help='target domain')
    parser.add_argument('--task', type=int, default=10)
    parser.add_argument('--source', type=str, default='face', help='source domain')
    parser.add_argument('--latent_dir', type=str)
    parser.add_argument("--sample_truncation", default=0.7, type=float, help="Truncation value for sampled test images.")
    torch.manual_seed(10)
    random.seed(10)
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    args.exp_name = args.target

    latent_code = np.load(args.latent_dir, allow_pickle=True).item()
    # print(latent_code)
    noise = torch.tensor([]).cuda()
    for key in latent_code.keys():
        noise = torch.cat((noise, torch.tensor(latent_code[key][args.iteration]).unsqueeze(0).cuda()), axis=0)
    print(noise.shape)
        # print(len(latent_code[key]), latent_code[key][0].shape)
    # if args.source == 'church':
    #     Proj_module = Projection_module_church(args)
    # if args.source == 'face':
    #     Proj_module = Projection_module(args)
    print('############################# generating #############################')
    tar_model = args.tar_model
    method = args.method

    # method, tar_model = args.ckpt_target.split('/')[-2], args.ckpt_target.split('/')[-1].split('.')[:-1]

    if args.mode == 'viz_imgs' or args.mode == 'eval_SCS':

        temp_str = './inference/' + tar_model + '_' + method
        imsave_path_source = os.path.join('./', args.mode, temp_str, 'source')
        imsave_path_target = os.path.join('./', args.mode, temp_str, 'target')
        if not os.path.exists(imsave_path_source):
            os.makedirs(imsave_path_source)
        if not os.path.exists(imsave_path_target):
            os.makedirs(imsave_path_target)
        
        args.save_source = imsave_path_source
        args.save_target = imsave_path_target
    
    if args.mode == 'viz_gif':
        temp_str = f"%s" % args.source
        imsave_path_source = os.path.join('./', args.mode, temp_str, 'source')
        imsave_path_target = os.path.join('./', args.mode, temp_str, args.target)
        if not os.path.exists(imsave_path_source):
            os.makedirs(imsave_path_source)
        if not os.path.exists(imsave_path_target):
            os.makedirs(imsave_path_target)
        args.save_source = imsave_path_source
        args.save_target = imsave_path_target

    if args.mode == 'eval_FIDnISnLPIPS':
        temp_str = './inference/' + tar_model + '_' + method
        imsave_path_target = os.path.join('./', args.mode, temp_str, 'images')
        if not os.path.exists(imsave_path_target):
            os.makedirs(imsave_path_target)
        
        args.save_target = imsave_path_target

    if args.mode == 'eval_SIFID':
        temp_str = './inference/' + tar_model + '_' + method
        imsave_path_target = os.path.join('./', args.mode, temp_str, 'images')
        if not os.path.exists(imsave_path_target):
            os.makedirs(imsave_path_target)

        args.save_target = imsave_path_target

    if method != 'AdAM':
        from project.model.ZSSGAN import SG2Generator
        if args.ckpt_source is not None:
            g_source = SG2Generator(args.ckpt_source,
                                    img_size=args.size, channel_multiplier=args.channel_multiplier
                                    ).to(device)
        if args.ckpt_target is not None:
            g_target = SG2Generator(args.ckpt_target,
                                    img_size=args.size, channel_multiplier=args.channel_multiplier
                                    ).to(device)
    else:
        from project.model.model_adam import Generator
        if args.ckpt_source is not None:
            checkpoint = torch.load(args.ckpt_source)
            g_source = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
            g_source.load_state_dict(checkpoint['g_ema'], strict=False)
        if args.ckpt_target is not None:
            checkpoint = torch.load(args.ckpt_target)
            g_target = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
            g_target.load_state_dict(checkpoint['g_ema'], strict=False)
    
    if args.mode == 'viz_imgs':

        generate_imgs(args, g_source, g_target)
    
    if args.mode == 'eval_FIDnISnLPIPS':
    
        generate_imgs_4FIDnISnLPIPS(args, g_target, latent_code=noise)

    if args.mode == 'eval_SIFID':
        generate_imgs_SIFID(args, g_target)
    

    if args.mode == 'eval_SCS':
    
        generate_img_pairs(args, g_source, g_target)



    elif args.mode == 'viz_gif':
        generate_gif(args, g_source, g_target)

    print('############################# end of generation #############################')