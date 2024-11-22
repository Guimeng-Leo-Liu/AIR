import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.inception import inception_v3
import os
import numpy as np
from scipy.stats import entropy
import argparse


def inception_score(imgs, device=None, batch_size=64, resize=False, splits=1):

    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)


    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = inception_model.to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)

    def get_pred(x):
        with torch.no_grad():
            if resize:
                x = up(x)
            x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()
        #     out = F.softmax(x, dim=-1)
        # out = out.cpu().numpy()
        # return out

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch, _) in enumerate(dataloader, 0):
        batchv = batch.to(device)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_is_given_paths(args):
    data = datasets.ImageFolder(os.path.dirname(args.save_target), transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    IS_dataset = torch.utils.data.Subset(data, indices=range(args.IS_sample))
    print("Calculating Inception Score...")
    IS_mean, IS_std = inception_score(IS_dataset, device='cuda', batch_size=64, resize=True, splits=10)
    return IS_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_target', type=str, default=None)
    parser.add_argument('--IS_sample', type=int, default=1000)
    args = parser.parse_args()

    IS_mean = calculate_is_given_paths(args)

    print('Inception score: %.4f' % IS_mean)

    fp = open(args.save_target.split('/')[-2] + '/' + args.save_target.split('/')[-1] + '_IS.txt', 'a+')
    fp.seek(0)
    fp.truncate()
    fp.write(str(IS_mean))
    fp.close()




