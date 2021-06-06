# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 9:23
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : train.py
# @Software: PyCharm

from config import *
import model1
import crowddataset as Dataset
import torch.utils.data.dataloader as Dataloader
import torch.optim as optim
import time
import visdom
from torch.autograd import Variable
import os
import random
from scripts.loss import *
from scripts.log import my_print as myprint
from scripts.log import print_train_log
from scripts.collate_fn import my_collect_fn
import torch
import numpy as np
import eval

def train():
    config_log = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()) + \
        "\n-------------------------------------------------------------" \
        "\nconfig:\n%s" \
        "-------------------------------------------------------------"
    l_temp = ""
    for i in range(len(VAR_LIST)):
        l_temp += "\t%s\n" % VAR_LIST[i]
    config_log = config_log % l_temp
    myprint(config_log)

    net = model1.Net()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda:0')
    # net = net.to(device)
    net = net.cuda()
    print('Is model on gpu: ', next(net.parameters()).is_cuda)
    myprint("--------------------------net architecture------------------------------------")
    myprint(net)
    myprint("------------------------------------------------------------------------------")

    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    #if LR_DECAY:
    #    schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=STEPS,gamma=LR_DECAY)

    criterion = get_loss()
    # if LOSS_F == "MSE":
    #     criterion = nn.MSELoss(reduction='sum').cuda()
    # elif LOSS_F == "L1":
    #     criterion = nn.L1Loss(reduction='sum').cuda()
    # else:
    #     criterion = nn.MSELoss(reduction='sum').cuda()

    t0 = time.time()
    start_epoch = 0
    step_index = 1

    min_mae = 200.0
    min_mse = 200.0

    min_epoch = -1
    epoch_list = []
    train_loss_list = []
    epoch_loss_list = []
    test_mae_list = []

    if RESUME:
        path_list = os.listdir("models/%s"%SAVE_PATH)
        path_list.remove("log.txt")
        epoch_list = [int(i.split("_")[-3][5:]) for i in path_list]
        curr_index = epoch_list.index(max(epoch_list))

        weight_path = os.path.join("models/%s"%SAVE_PATH, path_list[curr_index])

        min_epoch = epoch_list[curr_index]
        min_mae = float(path_list[curr_index].split("_")[-2][3:]) / 100.0
        start_epoch = min_epoch + 1

        for i in STEPS:
            if start_epoch>=i:
                step_index += 1

        net.load_state_dict(torch.load(weight_path))
        myprint("resume weight %s, at %d\n" % (weight_path, min_epoch))

    

    for i in range(start_epoch, MAX_EPOCH):

        train_dataset = Dataset.CrowdDataset()
        train_dataloader = Dataloader.DataLoader(train_dataset,
                                             batch_size=BATCH_SIZE  if BATCH_SIZE != 1 else BATCH_SIZE,
                                             num_workers=8, shuffle=True, drop_last=True, collate_fn=my_collect_fn,
                                             worker_init_fn=worker_init_fn)

        #test_dataset = Dataset.CrowdDataset(phase="test")
        # test_dataloader = Dataloader.DataLoader(test_dataset, batch_size=1, num_workers=8,
        #                                     shuffle=False, drop_last=False, collate_fn=my_collect_fn,
        #                                     worker_init_fn=worker_init_fn)

        #if LR_DECAY:
        #    schedule.step()

        ## train ##
        epoch_loss = 0.0
        epoch_ssimloss = 0.0
        epoch_mseloss = 0.0

        net.train()
        for _,(images,dt_targets) in enumerate(train_dataloader):
            # images,dt_targets = images.to(device), dt_targets.to(device)
            images,dt_targets = images.type(torch.FloatTensor), dt_targets.type(torch.FloatTensor)
            images,dt_targets = Variable(images.cuda()),Variable(dt_targets.cuda())
            densitymaps = net(images)

            if densitymaps.size() != dt_targets.size():
                myprint("train error! densitymaps size: %s,dt_targets %s. densitymaps.size()!=dt_targets.size().input image size: %s" % (str(densitymaps.size()), str(dt_targets.size()), str(images.size())))
                exit(-1)

            loss, _, _ = criterion(densitymaps, dt_targets)
            epoch_loss += loss.item()
            #epoch_ssimloss += ssimloss.item()
            #epoch_mseloss += Mseloss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        epoch_loss_list.append(epoch_loss)
        train_loss_list.append(epoch_loss/len(train_dataloader))

        epoch_list.append(i)
        localdate = time.strftime("%Y/%m/%d %H:%M:%S",time.localtime())
        myprint(localdate)
        print_train_log(i,time.time()-t0,epoch_loss, epoch_ssimloss, epoch_mseloss, len(train_dataloader))

        t0 = time.time()

        ## eval ##
        # net.eval()
        # with torch.no_grad():
        #     mae = 0.0
        #     mse = 0.0
        #
        #     for _,(images,dt_targets) in enumerate(test_dataloader):
        #         images, dt_targets = Variable(images.cuda()), Variable(dt_targets.cuda())
        #
        #         densitymaps = net(images)
        #
        #         if densitymaps.size() != dt_targets.size():
        #             myprint("test error! densitymaps size: %s,dt_targets %s. densitymaps.size()!=dt_targets.size(). input image size: %s" % (str(densitymaps.size()), str(dt_targets.size()), str(images.size())))
        #             exit(-1)
        #
        #         mae += abs(densitymaps.data.sum() - dt_targets.data.sum()).item()
        #         mse += (densitymaps.data.sum() - dt_targets.data.sum()).item() ** 2
        #
        #     mae = mae / len(test_dataloader)
        #     mse = (mse / len(test_dataloader)) **(1/2)


        mae, mse = eval.eval(net,DATASET,isSave=False)
        if(mae<min_mae):
            min_mae = mae
            min_mse = mse
            min_epoch = i
            save_log = "save state, epoch: %d" % i
            myprint(save_log)
            torch.save(net.state_dict(), "models/%s/%s_epoch%d_mae%d_mse%d.pth" % (SAVE_PATH,MODEL,i,mae*100,mse*100))

        elif mse < min_mse:
            min_mse = mse
            save_log = "save state, epoch: %d" % i
            myprint(save_log)
            torch.save(net.state_dict(),"models/%s/%s_epoch%d_mae%d_mse%d.pth" % (SAVE_PATH, MODEL, i, mae * 100, mse * 100))
        test_mae_list.append(mae)

        eval_log = "eval [%d/%d] mae %.4f, mse %.4f, min_mae %.4f, min_epoch %d\n"%(i,MAX_EPOCH,mae,mse,min_mae, min_epoch)
        myprint(eval_log)
        # with torch.no_grad():
        #     ## vis ##
        #     if USE_VISDOM and not RESUME:
        #         if len(train_loss2_list) == 0:
        #             viz.line(win="1", X=epoch_list, Y=train_loss_list, opts=dict(title="train_loss",legend=[LOSS_F]))
        #         else:
        #             viz.line(win="1", X=epoch_list,
        #                      Y=np.column_stack((np.array(train_loss_list), np.array(train_loss2_list), np.array(train_loss3_list))),
        #                      opts=dict(title="train_loss",legend=["total_loss","pyssim_loss","mse_loss"]))
        #
        #         viz.line(win="2", X=epoch_list, Y=test_mae_list, opts=dict(title="test_mae"))
        #         index = random.randint(0,len(test_dataloader)-1)
        #         image,gt_map = test_dataset[index]
        #         image = test_dataset.preProcess(image)
        #
        #         img_show=image.detach().cpu().numpy()
        #
        #         viz.image(win="3",img=img_show,opts=dict(title="test_image"))
        #         viz.image(win="4",img=gt_map/(gt_map.max())*255,opts=dict(title="gt_map_%.4f"%(gt_map.sum())))
        #
        #
        #         image = Variable(image.unsqueeze(0).cuda())
        #         net.eval()
        #         densitymap,atten1,atten2,atten3 = net(image,True)
        #         densitymap = densitymap.squeeze(0).detach().cpu().numpy()
        #         viz.image(win="5",img=densitymap/(densitymap.max())*255,opts=dict(title="predictImages_%.4f"%(densitymap.sum())))
        #
        #         atten1 = atten1.squeeze(0).detach().cpu().numpy()
        #         atten2 = atten2.squeeze(0).detach().cpu().numpy()
        #         atten3 = atten3.squeeze(0).detach().cpu().numpy()
        #         viz.image(win="6", img=atten1 / (atten1.max()) * 255,opts=dict(title="attentionMap1"))
        #         viz.ShanghaiTechimage(win="7", img=atten2 / (atten2.max()) * 255, opts=dict(title="attentionMap2"))
        #         viz.image(win="8", img=atten3 / (atten3.max()) * 255, opts=dict(title="attentionMap3"))


def setup_seed(seed=19960715):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id): # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
    if not os.path.exists("models/%s" % SAVE_PATH):
        os.makedirs("models/%s" % SAVE_PATH)
    if USE_VISDOM:
        viz = visdom.Visdom(env=SAVE_PATH.replace("/", "_"))
    setup_seed(seed=SEED)
    train()
