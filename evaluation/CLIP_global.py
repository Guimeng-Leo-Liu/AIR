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
import clip


def get_image_features(model, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
    image_embeddings = model.encode_image(img).detach()

    if norm:
        image_embeddings /= image_embeddings.clone().norm(dim=-1, keepdim=True)

    return image_embeddings

def get_text_features(model, tokens: torch.Tensor, norm: bool = True) -> torch.Tensor:

    text_features = model.encode_text(tokens).detach() # 映射到CLIP Space中

    if norm:
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def global_score(args, imgs, text, model, device=None, batch_size=32, save_image_embeddings=False):

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    tokens = clip.tokenize([text]).to(device)
    text_embeddings = get_text_features(model, tokens)

    global_loss = torch.tensor([])
    embeddings = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # if i % 10 == 0:
            #     print('loading {}th batch'.format(i))
            images, label = data
            images = images.to(device)
            image_embeddings = get_image_features(model, images)
            # print(image_embeddings.shape, text_embeddings.shape)
            D_cos = 1.0 - torch.nn.CosineSimilarity()(image_embeddings, text_embeddings)
            if save_image_embeddings:
                embeddings = torch.cat((embeddings, image_embeddings.cpu().detach()), dim=0)
            global_loss = torch.cat((global_loss, D_cos.cpu().detach()), dim=0)
    print(global_loss.shape)
    if save_image_embeddings:
        print(embeddings.shape)
        np.save(args.output_dir + '/image embeddings.npy', embeddings.numpy())

    return torch.std_mean(global_loss)

def calculate_global_loss_given_paths(args, text, model_name = 'ViT-B/32', batch_size=50, save_image_embeddings=False):
    model, clip_preprocess = clip.load(model_name, device='cuda')
    data = datasets.ImageFolder(os.path.dirname(args.save_target), transform=clip_preprocess)
    global_dataset = torch.utils.data.Subset(data, indices=range(args.global_sample))
    print("Calculating Global Score...")
    std, mean = global_score(args, global_dataset, text, model, device='cuda', batch_size=batch_size, save_image_embeddings=save_image_embeddings)
    # print(global_loss)
    return mean, std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_target', type=str, default=None)
    parser.add_argument('--target_class', type=str, default=None)
    parser.add_argument('--global_sample', type=int, default=5000)
    args = parser.parse_args()

    global_loss = calculate_global_loss_given_paths(args)

    print('Global CLIP: %.4f' % global_loss)

    fp = open(args.save_target.split('/')[-2] + '/' + args.save_target.split('/')[-1] + '_globalCLIP.txt', 'a+')
    fp.seek(0)
    fp.truncate()
    fp.write(str(IS_mean))
    fp.close()




