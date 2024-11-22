#####
# Adapted from Ojha et al (https://github.com/utkarshojha/few-shot-gan-adaptation/blob/main/feat_cluster.py)
#####

import argparse
import random
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm
import sys
import random
import lpips
from torchvision import transforms, utils
from torch.utils import data
import os
from PIL import Image
import numpy as np

from sklearn_extra.cluster import KMedoids

from time import time


def get_img_paths(img_dir):
    return [os.path.join(img_dir, file_path) for file_path in os.listdir(img_dir) if
            file_path.endswith(".jpg") or file_path.endswith(".png")]


class LPIPS_Embedder(nn.Module):
    def __init__(self, device) -> None:
        super(LPIPS_Embedder, self).__init__()

        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)

        self.preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def forward(self, img1, img2):
        with torch.no_grad():
            img1_prep = self.preprocess(img1).cuda()
            img2_prep = self.preprocess(img2).cuda()

            return self.lpips_fn(img1_prep, img2_prep)

def cluster_images_kmed(lpips_embedder, gen_paths, LPIPS_sample):
    t_start = time()
    num_points = min(len(gen_paths), LPIPS_sample)
    pairwise_dists = np.zeros(shape=(num_points, num_points))

    for i in range(num_points):
        i_img = Image.open(gen_paths[i])
        pairwise_dists[i, i] = 0
        for j in range(i, num_points):
            j_img = Image.open(gen_paths[j])

            dist = lpips_embedder(i_img, j_img)
            pairwise_dists[i, j] = dist
            pairwise_dists[j, i] = dist

    kmed = KMedoids(n_clusters=10, metric="precomputed").fit(pairwise_dists)

    # print(f"total time: {time() - t_start}")

    all_dists = []
    cluster_dists = [[] for _ in range(10)]

    for idx in range(num_points):
        cluster_idx = kmed.medoid_indices_[kmed.labels_[idx]]
        dist = pairwise_dists[idx, cluster_idx]

        all_dists.append(dist)
        cluster_dists[kmed.labels_[idx]].append(dist)

    avg_dists = [np.mean(cluster_dist) for cluster_dist in cluster_dists]

    print("Final avg. %f/%f" % (np.mean(avg_dists), np.std(avg_dists)))
    print("Final avg on all: %f/%f" % (np.mean(all_dists), np.std(all_dists)))
    return np.mean(all_dists)

def calculate_lpips_given_paths(args):

    embedder = LPIPS_Embedder("cuda:0")
    # step 3. compute intra-lpips
    gen_paths = get_img_paths(args.save_target)

    intra_lpips_dist = cluster_images_kmed(embedder, gen_paths, args.LPIPS_sample)


    return intra_lpips_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_target', type=str, default=None)
    parser.add_argument('--center_path', type=str, default="./cluster_centers/")
    args = parser.parse_args()

    intra_lpips_dist = calculate_lpips_given_paths(args)

    print('Intra LPIPS score: %.4f' % intra_lpips_dist)
    fp = open(args.img_pth.split('/')[-2] + '/' + args.img_pth.split('/')[-1] + '_intraLPIPS.txt', 'a+')
    fp.seek(0)
    fp.truncate()
    fp.write(str(intra_lpips_dist))
    fp.close()