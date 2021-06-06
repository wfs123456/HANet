# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 9:54
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : eval.py
# @Software: PyCharm

import crowddataset as Dataset
import torch.utils.data.dataloader as Dataloader
import torch
from torch.autograd import Variable
import os
from scripts.collate_fn import my_collect_fn
import tqdm
from config import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval(model,dataset,isSave=True):
    model.eval()
    test_dataset = Dataset.CrowdDataset(dataset=dataset,phase="test")
    test_dataloader = Dataloader.DataLoader(test_dataset, batch_size=1, num_workers=8,
                                            shuffle=False, drop_last=False, collate_fn=my_collect_fn)

    with torch.no_grad():
        mae = 0
        mse = 0
        list_mae = []
        for _,(images,dt_targets) in enumerate(tqdm.tqdm(test_dataloader,desc="eval on %s"%dataset)):
            images, dt_targets = Variable(images.cuda()), Variable(dt_targets.cuda())

            densitymaps = model(images)

            mae += abs(densitymaps.data.sum()-dt_targets.data.sum()).item()
            mse += (densitymaps.data.sum()-dt_targets.data.sum()).item()**2

            list_mae.append(abs(densitymaps.data.sum()-dt_targets.data.sum()).item())

        mae = mae / len(test_dataloader)
        mse = (mse / len(test_dataloader)) **(1/2)

    # print("mae: ",mae, " mse: ",mse)
    if isSave:
        with open("eval.txt","w") as f:
            for index,i in enumerate(list_mae):
                f.write("index %d: "%index+str(i)+"\n")
            f.write("----------------------------------\n")
            f.write("mae: "+str(mae) + "\t" + "mse: "+str(mse))
    return mae, mse


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import model
    import argparse

    model = model.Net().cuda()

    parser = argparse.ArgumentParser()

    parser.add_argument("--weight_path", type=str, default="models/ShanghaiTech/part_A_final/" \
                        "20200714after/WeightComp/pssloss1e-06_3AttenModule_NoAttenBN_AttenPath3x3_weight05/CRANet_epoch1178_mae5461_mse8930.pth",
                        help="the weight path to be loaded")
    parser.add_argument("--dataset",type=str,default="ShanghaiTech/part_A_final",help="the dataset to be eval")
    opt = parser.parse_args()
    print(opt)

    weights = torch.load(opt.weight_path)
    model.load_state_dict(weights)

    eval(model,opt.dataset)



