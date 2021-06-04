# -*- coding: utf-8 -*-
# @Time    : 2/20/20 10:15 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : collate_fn.py
# @Software: PyCharm


import torch
import random
from config import *
import numpy as np
import cv2
import torchvision.transforms as transforms

cfg = {
            "part_A_final":
                [[0.410824894905, 0.370634973049, 0.359682112932],
                 [0.278580576181, 0.26925137639, 0.27156367898]],
            "part_B_final":
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
            "UCF-QNRF-Nor":
                [[0.413525998592, 0.378520160913, 0.371616870165],
                 [0.284849464893, 0.277046442032, 0.281509846449]],
        }

def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

transforms = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cfg[DATASET][0],std=cfg[DATASET][1])
                                              ])
def my_collect_fn(batch):
    imgs,labels = zip(*batch)
    imgs,labels = list(imgs),list(labels)

    if len(imgs)==1:
        res_labels = []
        for i,label in enumerate(labels):
            if GT_DOWNSAMPLE > 1:  # to downsample image and density-map to match deep-model.
                ds_rows = label.shape[0] // GT_DOWNSAMPLE
                ds_cols = label.shape[1] // GT_DOWNSAMPLE

                label = cv2.resize(label, (ds_cols, ds_rows), interpolation=cv2.INTER_CUBIC) * GT_DOWNSAMPLE * GT_DOWNSAMPLE
            label = label[np.newaxis,:,:]
            res_labels.append(torch.tensor(label,dtype=torch.float))
            imgs[i] = transforms(imgs[i])

        imgs = torch.stack(imgs, 0)
        labels = torch.stack(res_labels, 0)
        return imgs, labels

    batch_size = len(imgs)
    res_imgs = []
    res_labels = []

    for i in range(batch_size):
        img = imgs[i]
        label = labels[i]
        size = label.shape

        for j in range(CROP_NUM):
            x0 = random.randint(0,size[0]-CROP_SIZE)
            x1 = x0 + CROP_SIZE
            y0 = random.randint(0, size[1]-CROP_SIZE)
            y1 = y0 + CROP_SIZE

            crop_img = img[x0:x1,y0:y1,:].copy()
            crop_label = label[x0:x1,y0:y1].copy()

            if GT_DOWNSAMPLE > 1:  # to downsample image and density-map to match deep-model.
                ds_rows = crop_label.shape[0] // GT_DOWNSAMPLE
                ds_cols = crop_label.shape[1] // GT_DOWNSAMPLE
                crop_label = cv2.resize(crop_label, (ds_cols, ds_rows), interpolation=cv2.INTER_CUBIC) * GT_DOWNSAMPLE * GT_DOWNSAMPLE
            crop_label = crop_label[np.newaxis, :, :]
            crop_label = torch.tensor(crop_label, dtype=torch.float)
            
            crop_img = transforms(crop_img)

            res_imgs.append(crop_img)
            res_labels.append(crop_label)

    imgs = torch.stack(res_imgs,0, out=share_memory(res_imgs))
    labels = torch.stack(res_labels,0, out=share_memory(res_labels))
    return imgs,labels


if __name__ == "__main__":
    import crowddataset as Dataset
    import torch.utils.data.dataloader as Dataloader
    import matplotlib.pyplot as plt

    dataset = Dataset.CrowdDataset(dataset="part_A_final",phase="train")
    dataloader = Dataloader.DataLoader(dataset, batch_size=4, num_workers=0,
                                            shuffle=False, drop_last=False,
                                            collate_fn=my_collect_fn)

    # dataset = Dataset.Dataset(crop_factor=5)
    # dataloader = Dataloader.DataLoader(dataset, batch_size=BATCH_SIZE // CROP_NUM if BATCH_SIZE != 1 else BATCH_SIZE,
    #                                    num_workers=8, shuffle=True, drop_last=True,
    #                                    collate_fn=my_collect_fn)

    for i,(images,targets) in enumerate(dataloader):
        print(images.size(),targets.size())
        # images = images[-1].squeeze(0).transpose(0, 2).transpose(0, 1)
        # images = images.numpy()
        # images[:, :, 0], images[:, :, 2] = images[:, :, 2], images[:, :, 0]
        #
        # cv2.imwrite("../samples/image1.png", images * 255.0)
        #
        # targets = targets[-1].squeeze(0).squeeze(0)
        # print(images.shape, targets.size())
        # plt.imsave("../samples/dt_map1.png", targets)
        print("11111111111")
        exit(1)
