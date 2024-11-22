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
import tqdm

def assign_fake_images_to_cluster_center(args):
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        center_path = os.path.join(f"../cluster_centers", args.data_path, args.method)
        files_list_samples = os.listdir(args.intra_lpips_path)

        # Step 1. compute lpips between sample and center
        for i in tqdm(range(len(files_list_samples))):  # all generated samples
            dists = []
            for k in range(10):  # cluster center
                cluster_center = os.path.join(center_path, "c%d" % (k), "center.png")
                input1_path = os.path.join(args.intra_lpips_path, files_list_samples[i])
                input2_path = os.path.join(cluster_center)

                input_image1 = Image.open(input1_path).convert('RGB')
                input_image2 = Image.open(input2_path).convert('RGB')

                input_tensor1 = preprocess(input_image1)
                input_tensor2 = preprocess(input_image2)

                input_tensor1 = input_tensor1.cuda()
                input_tensor2 = input_tensor2.cuda()

                dist = lpips_fn(input_tensor1, input_tensor2)
                dists.append(dist.cpu())
            dists = np.array(dists)

            # Step 2. Move images close to the best matched cluster
            idx = np.argmin(dists)
            shutil.move(input1_path, os.path.join(center_path, "c%d" % (idx)))


def intra_cluster_dist(args):
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        cluster_size = 50
        base_path = os.path.join(f"../cluster_centers", args.data_path, args.method)
        avg_dist = torch.zeros([10, ])  # placeholder for intra-cluster lpips
        for k in range(10):
            curr_path = os.path.join(base_path, "c%d" % (k))
            files_list = os.listdir(curr_path)
            files_list.remove('center.png')

            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            for i in range(len(files_list)):
                for j in range(i+1, len(files_list)):
                    input1_path = os.path.join(curr_path, files_list[i])
                    input2_path = os.path.join(curr_path, files_list[j])
                    # input2_path = os.path.join(curr_path, 'center.png')
                    input_image1 = Image.open(input1_path)
                    input_image2 = Image.open(input2_path)

                    input_tensor1 = preprocess(input_image1)
                    input_tensor2 = preprocess(input_image2)

                    input_tensor1 = input_tensor1.cuda()
                    input_tensor2 = input_tensor2.cuda()

                    dist = lpips_fn(input_tensor1, input_tensor2)

                    dists.append(dist.cpu())
            dists = torch.tensor(dists)
            print ("Cluster %d:  Avg. pairwise LPIPS dist: %f/%f" %
                   (k, dists.mean(), dists.std()))
            avg_dist[k] = dists.mean()

        # print ("Final avg. %f/%f" % (avg_dist[~torch.isnan(avg_dist)].mean(), avg_dist[~torch.isnan(avg_dist)].std()))
        return avg_dist[~torch.isnan(avg_dist)].mean()


def del_assigned_images(args):

    # remove images around cluster center
    base_path = os.path.join(f"../cluster_centers", args.data_path, args.method)
    for k in range(10):
        curr_path = os.path.join(base_path, "c%d" % (k))
        files_list = os.listdir(curr_path)
        files_list.remove('center.png')

        for i in range(len(files_list)):
            img = os.path.join(curr_path, files_list[i])
            if os.path.exists(img):
                os.remove(img)
            else:
                print("The file does not exist")
    print ("assigned images deleted" )

    # clear abundant generated images (if any left)
    files_list = os.listdir(args.intra_lpips_path)

    for i in range(len(files_list)):
        img = os.path.join(args.intra_lpips_path, files_list[i])
        if os.path.exists(img):
            os.remove(img)
        else:
            print("The file does not exist")
    print ("generated images deleted" )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intra_lpips_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--LPIPS_sample', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # del_assigned_images(args)
    # step 2. assign generated images to cluster centers
    assign_fake_images_to_cluster_center(args)

    # step 3. compute intra-lpips
    intra_lpips_dist = intra_cluster_dist(args)

    # step 4. delete abundant images of this checkpoint
    del_assigned_images(args)

    print('Intra LPIPS score: %.4f' % intra_lpips_dist)
    fp = open(args.img_pth.split('/')[-2] + '/' + args.img_pth.split('/')[-1] + '_intraLPIPS.txt', 'a+')
    fp.seek(0)
    fp.truncate()
    fp.write(str(intra_lpips_dist))
    fp.close()