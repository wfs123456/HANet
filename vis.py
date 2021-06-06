# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 11:32
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : vis.py
# @Software: PyCharm

import model
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as io
import torch
import os
import argparse

cfg = {
    "ShanghaiTech/part_A_final":
        [[0.410824894905, 0.370634973049, 0.359682112932],
         [0.278580576181, 0.26925137639, 0.27156367898]],
    "ShanghaiTech/part_B_final":
        [[0.452016860247, 0.447249650955, 0.431981861591],
         [0.23242045939, 0.224925786257, 0.221840232611]],
    "UCF-CC-50/folder1":
        [[0.403584420681,0.403584420681,0.403584420681],
         [0.268462955952,0.268462955952,0.268462955952]],
    "UCF-CC-50/folder2":
        [[0.403584420681, 0.403584420681, 0.403584420681],
         [0.268462955952, 0.268462955952, 0.268462955952]],
    "UCF-CC-50/folder3":
        [[0.403584420681, 0.403584420681, 0.403584420681],
         [0.268462955952, 0.268462955952, 0.268462955952]],
    "UCF-CC-50/folder4":
        [[0.403584420681, 0.403584420681, 0.403584420681],
         [0.268462955952, 0.268462955952, 0.268462955952]],
    "UCF-CC-50/folder5":
        [[0.403584420681, 0.403584420681, 0.403584420681],
         [0.268462955952, 0.268462955952, 0.268462955952]],
    "QNRF":
        [[0.413525998592, 0.378520160913, 0.371616870165],
         [0.284849464893, 0.277046442032, 0.281509846449]],
}


def vis(model,image_path):
    if not os.path.exists(image_path):
        print("not find image path!")
        exit(-1)

    model.eval()

    if image_path.split("/")[-4] != "QNRF":
        dataset = "/".join(image_path.split("/")[-5:-3])
    else:
        dataset = "QNRF"

    print("detect image '%s'..." % image_path)
    if not os.path.exists(image_path):
        print("not find image path!")
        exit(-1)

    if dataset == "QNRF":
        mat = io.loadmat(
            image_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('.', '_ann.').replace(
                "UCF-QNRF-Nor", "UCF-QNRF"))
        points = mat["annPoints"]
    else:
        mat = io.loadmat(image_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace("IMG", "GT_IMG"))
        points = mat["image_info"][0, 0][0, 0][0]

    gt_count = len(points)

    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image, dtype=np.float32)
    if len(image.shape) == 2:  # expand grayscale image to three channel.
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), 2)
    vis_img = image.copy()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=cfg[dataset][0], std=cfg[dataset][1])])
    img_tensor = transform(image)
    image = Variable(img_tensor.unsqueeze(0).cuda())

    gt_dmap_path = image_path.replace('.jpg', '.npy').replace('images', 'density_maps_constant4')
    gt_dmap = np.load(gt_dmap_path)

    dmap,atten1,atten2,atten3 = model(image,vis=True)

    dmap = dmap.squeeze(0).squeeze(0).cpu().data.numpy()
    atten1 = atten1.squeeze(0).squeeze(0).cpu().data.numpy()
    atten2 = atten2.squeeze(0).squeeze(0).cpu().data.numpy()
    atten3 = atten3.squeeze(0).squeeze(0).cpu().data.numpy()

    return dmap,atten1,atten2,atten3,vis_img,gt_dmap,gt_count


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="/home/zhongyuan/datasets/"
                             "ShanghaiTech/part_A_final0/test_data/images/IMG_2.jpg",
                        help="the image path to be detected.")
    parser.add_argument("--weight_path", type=str, default="models/ShanghaiTech/part_A_final/" \
                  "20200714after/WeightComp/pssloss1e-06_3AttenModule_NoAttenBN_AttenPath3x3_weight05/CRANet_epoch1178_mae5461_mse8930.pth",
                        help="the weight path to be loaded")
    opt = parser.parse_args()
    print(opt)

    model = model.Net().cuda()

    print("weight path: %s\nloading weights..." % opt.weight_path)
    weights = torch.load(opt.weight_path)
    model.load_state_dict(weights)

    dmap, atten1, atten2, atten3, vis_img, gt_dmap, gt_count = vis(model,opt.image_path)

    save_path = "vis/%s"%(opt.image_path.split("/")[-4]+"/"+opt.image_path.split("/")[-1][:-4])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("dmap count is %.2f, gt_dmap count is %.2f, gt count is %d"%(dmap.sum(),gt_dmap.sum(),gt_count))


    plt.imsave("%s/atten1.png" % save_path, atten1)
    plt.imsave("%s/atten2.png" % save_path, atten2)
    plt.imsave("%s/atten3.png" % save_path, atten3)
    plt.imsave("%s/dmap.png" % save_path, dmap)
    plt.imsave("%s/gt_dmap.png" % save_path, gt_dmap)
    plt.imsave("%s/image.png" % save_path, vis_img/vis_img.max())

    print("the visual result saved in %s"%save_path)


