import torch
import numpy as np
import lpips
import shutil
import random
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image



def compute_intra_lpips(imgs, transform, device=None, batch_size=1):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size


    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    image_path = os.path.join("./oneshot_data", args.target_cls)
    image_name = os.listdir(image_path)[0]
    oneshot_path = os.path.join(image_path, image_name)
    oneshot_image = transform(Image.open(oneshot_path).convert('RGB')).to(device)

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    dists = []
    with torch.no_grad():
        for idx, (sample, _) in enumerate(dataloader, 0):
            dist = lpips_fn(oneshot_image, sample.to(device))
            dists.append(dist.cpu())

    dists = torch.tensor(dists)
    return dists.mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_pth', type=str, default=None)
    parser.add_argument('--LPIPS_sample', type=int, default=1000)
    parser.add_argument('--target_cls', type=str, default=None)
    args = parser.parse_args()

    preprocess = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    data = datasets.ImageFolder(args.img_pth, transform=preprocess)
    LPIPS_dataset = torch.utils.data.Subset(data, indices=range(args.LPIPS_sample))

    print("Calculating Inception Score...")
    intra_lpips_dist = compute_intra_lpips(LPIPS_dataset, transform=preprocess, device='cuda', batch_size=1)
    print('Intra LPIPS score: %.4f' % intra_lpips_dist)
    fp = open(args.img_pth.split('/')[-2] + '/' + args.img_pth.split('/')[-1] + '_intraLPIPS.txt', 'a+')
    fp.seek(0)
    fp.truncate()
    fp.write(str(intra_lpips_dist))
    fp.close()