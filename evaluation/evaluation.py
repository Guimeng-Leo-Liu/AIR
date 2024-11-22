import argparse
import os
import torch
import numpy as np
import random
from model.ZSSGAN import ZSSGAN, SG2Generator
from options.train_options import TrainOptions
from torch.utils import data
# from utils.file_utils import save_images
from torchvision import utils
from torchvision import transforms
import warnings
# import wandb
warnings.filterwarnings("ignore")

from gan_training import utils
from gan_training.eval import Evaluator
from dataset import MultiResolutionDataset



dataset_sizes = {
    "ffhq":   1024,
    "afhq":    512,
}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def eval(args):
    args.size = dataset_sizes[args.source_model_type]

    generator_ema = SG2Generator(args.adapted_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier).to(device)
    generator_ema.freeze_layers()
    generator_ema.eval()

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(256),
    #         transforms.CenterCrop(256),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    #     ]
    # )
    # real_dataset   = MultiResolutionDataset(args.data_path_real, transform, args.size)
    # real_loader    = data.DataLoader(
    #     real_dataset,
    #     batch_size=args.batch,
    #     sampler=data_sampler(real_dataset, shuffle=False, distributed=False),
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True,
    #     worker_init_fn=seed_worker
    # )
    # real_loader = sample_data(real_loader)
    #
    # x_real = utils.get_nsamples_lmdb(real_loader, args.n_sample_real, set_len=real_dataset.length)
    # to compute IS and FID
    evaluator = Evaluator(args, generator_ema,
                          batch_size=args.batch,
                          device=device,
                          fid_real_samples=None, # x_real
                          inception_nsamples=args.n_sample_real, # 5000
                          fid_sample_size=args.n_sample_real) # 5000
    with torch.no_grad():
        # eval metrics
        score = evaluator.compute_inception_score(kid=False, pr=False)
        # intra_lpips = evaluator.compute_intra_lpips(args=args).cpu().numpy()
    print(score)
    # if wandb and args.wandb:
    #     wandb.log(
    #         {
    #             "IS": score['IS'],
    #             "FID"    : score['fid'],
    #             # "intra-lpips": intra_lpips,
    #         }
    #     )

    # with torch.no_grad():
    #     sample_w = generator_ema.style([fixed_z])
    #     sample = generator_ema(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[0]
    #
    # utils.save_image(
    #     sample,
    #     f'{out_dir}/{idx:04d}.png',
    #     normalize=True,
    #     range=(-1, 1),
    # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default='./output')
    parser.add_argument("--data_path_real", type=str, default='../dataset/FFHQ/FFHQ')
    parser.add_argument("--source_model_type", type=str, default='ffhq')
    parser.add_argument("--adapted_gen_ckpt", type=str)
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1",)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--n_sample_real", type=int, default=5000)
    parser.add_argument("--n_sample_store", type=int, default=25, help="# of generated images using intermediate models")
    parser.add_argument("--sample_truncation", default=0.7, type=float, help="Truncation value for sampled test images.")
    parser.add_argument("--latent", type=int, default=512)
    # parser.add_argument("--wandb", action="store_true")
    # parser.add_argument("--wandb_project_name", type=str, default='debug')
    # parser.add_argument("--wandb_run_name", type=str, default='debug')

    args = parser.parse_args()
    # wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, reinit=True)

    os.makedirs(args.output_path, exist_ok=True)
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)

    # Step 1. Pre-experiment setups
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval(args)