from json.tool import main
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
from .hed_model import HED_Network

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

from PIL import Image
import numpy as np
from scipy.stats import entropy
import argparse

##########################################################

def HED_estimate(tenInput, HED_net):

	#assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	with torch.no_grad():
		return HED_net(tenInput.cuda()).cpu()
# end

##########################################################

def read_Image(img_name):
	return torch.from_numpy(numpy.ascontiguousarray(numpy.array(PIL.Image.open(img_name))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

def SCS_eval(args, HED_net):
    img_dir = args.img_pth 
    imgs_source = os.listdir(os.path.join(img_dir, 'source'))
    imgs_target = os.listdir(os.path.join(img_dir, 'target'))
    imgs_source_HED = os.path.join(img_dir, 'source_HED')
    imgs_target_HED = os.path.join(img_dir, 'target_HED')
    if not os.path.exists(imgs_source_HED):
        os.makedirs(imgs_source_HED)
    if not os.path.exists(imgs_target_HED):
        os.makedirs(imgs_target_HED)
    img_list = []
    for i in range(500):  
        image_name = os.path.join(os.path.join(img_dir, 'source'), imgs_source[i])
        img_list.append(read_Image(image_name).unsqueeze(0))
	#exit()
	#enInput = torch.FloatTensor()
    img_list = torch.cat(img_list, dim=0)
    tenOutput_list = []
    for i in range(50):
        tenOutput = HED_estimate(img_list[i*10:(i+1)*10], HED_net)
        tenOutput_list.append(tenOutput)
        tenOutput = torch.cat(tenOutput_list)
    for num, item in enumerate(tenOutput):
        PIL.Image.fromarray((item.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(f'%s/%s' % (imgs_source_HED, imgs_source[num]))

    img_list = []
    for i in range(500):  
        image_name = os.path.join(os.path.join(img_dir, 'target'), imgs_target[i])
        img_list.append(read_Image(image_name).unsqueeze(0))
    tenOutput_list = []
    img_list = torch.cat(img_list, dim=0)
    for i in range(50):
        tenOutput = HED_estimate(img_list[i*10:(i+1)*10], HED_net)
        tenOutput_list.append(tenOutput)
        tenOutput = torch.cat(tenOutput_list)
    for num, item in enumerate(tenOutput):
        PIL.Image.fromarray((item.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(f'%s/%s' % (imgs_target_HED, imgs_target[num]))
    score = 0
    for i in range(500):    
    
        img_s = np.array(Image.open(f'%s/img{i}.png' % imgs_source_HED)) 
        img_t = np.array(Image.open(f'%s/img{i}.png' % imgs_target_HED))
        img_s = torch.from_numpy(img_s).cuda().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) / 255
        img_t = torch.from_numpy(img_t).cuda().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) / 255

        sim = 2 * (img_s * img_t).sum() / (img_s**2 + img_t**2).sum()
        score += sim

    SCS_Score = score / 500

    return SCS_Score




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_pth', type=str, default=None)

    args = parser.parse_args()
    HED_net = HED_Network().cuda().eval()
    SCS_Score = SCS_eval(args, HED_net)

    print('SCS Score: %.4f' % SCS_Score.item())

    # fp = open(img_dir+'/SCS_score.txt', 'a+')
    fp = open(args.img_pth.split('/')[-2] + '/' + args.img_pth.split('/')[-1] + 'SCS_score.txt', 'a+')
    fp.seek(0)
    fp.truncate()
    fp.write(str(SCS_Score.item()))
    fp.close()

