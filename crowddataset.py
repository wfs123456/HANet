# -*- coding: utf-8 -*-
# @Time    : 2020/10/6 9:21
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : crowddataset.py
# @Software: PyCharm


from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from config import *
import torchvision.transforms as transforms
from scripts.image import *


class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''

    def __init__(self, dataset=DATASET,phase="train", segma=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        crop_factor: how to crop in each epoch.
        '''
        super(CrowdDataset, self).__init__()

        self.cfg = {
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

        self.phase = phase
        self.name = dataset
        self.segma = segma
        self.img_root = os.path.join(HOME,self.name,"%s_data/images"%(phase))
        self.gt_dmap_root = os.path.join(HOME,self.name,"%s_data/density_maps_constant%s"%(phase,segma))

        self.img_names = [filename for filename in os.listdir(self.img_root) \
                          if os.path.isfile(os.path.join(self.img_root, filename))]
        random.shuffle(self.img_names)
        self.n_samples = len(self.img_names)

        self.transforms = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg[dataset][0],std=self.cfg[dataset][1])
                                              ])

    def __len__(self):
        return self.n_samples

    def dataAugument(self,img,gt_dmap):
        if self.phase == "train":
            if RANDOM_FLIP:
                img,gt_dmap = random_flip(img,gt_dmap,RANDOM_FLIP)
            if RANDOM_HUE:
                img, gt_dmap = random_hue(img, gt_dmap, RANDOM_HUE)
            if RANDOM_SATURATION:
                img, gt_dmap = random_saturation(img, gt_dmap, RANDOM_SATURATION)
            if RANDOM_BRIGHTNESS:
                img, gt_dmap = random_brightness(img, gt_dmap, RANDOM_BRIGHTNESS)
            if RANDOM_2GRAY:
                img,gt_dmap = random_2gray(img,gt_dmap,RANDOM_2GRAY)
            if RANDOM_CHANNEL:
                img,gt_dmap = random_channel(img,gt_dmap,RANDOM_CHANNEL)
            if RANDOM_NOISE:
                img,gt_dmap = random_noise(img,gt_dmap,RANDOM_NOISE)
        if PADDING:
            img, gt_dmap = paddingByfactor(img, gt_dmap, PADDING)
        if self.phase == "test":
            if DIVIDE:
                img,gt_dmap =  divideByfactor(img, gt_dmap, DIVIDE)
        return img,gt_dmap

    def preProcess(self, img):
        img_tensor = self.transforms(img)
        return img_tensor

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_name = self.img_names[index]

        img = Image.open(os.path.join(self.img_root, img_name)).convert("RGB")
        img = np.asarray(img, dtype=np.float32)
        gt_dmap = np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg', '.npy')))

        img, gt_dmap = self.dataAugument(img, gt_dmap)
        # img = self.preProcess(img)

        return img, gt_dmap


if __name__ == "__main__":
    import torch.utils.data.dataloader as Dataloader
    import matplotlib.pyplot as plt
    from scripts.collate_fn import my_collect_fn

    seed = 0
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # cudnn

    train_dataset = CrowdDataset(dataset="part_A_final", phase="train")
    train_dataloader = Dataloader.DataLoader(train_dataset, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False,
                                            )

    for i,(images,targets) in enumerate(train_dataloader):
        print(images.size(),targets.size())

        # images = images[0].squeeze(0).transpose(0, 2).transpose(0, 1)
        images = images.numpy()
        print(images)
        # images[:, :, 0], images[:, :, 2] = images[:, :, 2], images[:, :, 0]
        #
        # cv2.imwrite("samples/image.png", images * 255.0)
        #
        # targets = targets[0].squeeze(0).squeeze(0)
        # print(images.shape, targets.size())
        # plt.imsave("samples/dt_map.png", targets)
        print("11111111111")
        exit(1)
    print("length", len(train_dataloader))
