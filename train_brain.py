import os
import time
import copy
import glob
import argparse

import torch
from torch import optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import losses
from dataset import data_generator_brain, load_vol, random_affine_augment

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="ours")
parser.add_argument("--train_data", type=str, default="/public/home/gyf/registration/data/OASIS/Train/Vols/")
parser.add_argument("--max_epoches", type=int, default=500)
parser.add_argument("--base_lr", type=float, default=0.0001)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--aug", type=bool, default=True)

args = parser.parse_args()
model_name = args.model_name
lr = args.base_lr
train_data = args.train_data
max_epoches = args.max_epoches
gpu_id = args.gpu_id
aug = args.aug


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def train():
    train_dataset = glob.glob(train_data + "*.nii.gz")
    steps_per_epoch = 100
    device = torch.device("cuda:" + gpu_id)
    criterions = [losses.NCC(device=device), losses.Grad3d(penalty='l2')]
    weights = [1.0, 1.0]
    vol_shape = (160, 192, 224)

    if model_name == "ours":
        from lscaregnet.lscaregnet import LSCARegNet
        model = LSCARegNet(vol_shape)

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    train_gen = data_generator_brain(train_dataset)

    for epoch in range(0, max_epoches):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, max_epoches, lr)
        for idx in range(steps_per_epoch):
            data, _ = next(train_gen)
            data = [torch.from_numpy(t).to(device) for t in data]
            moving = data[0]
            fixed = data[1]
            moving = moving[None, None, ...].float()
            fixed = fixed[None, None, ...].float()

            if aug:
                pass
                # 等待审稿结果上传

            model_in = torch.cat((moving, fixed), dim=1)
            warped, flow = model(model_in)
            loss = 0.0
            curr_loss = [0, 0]
            curr_loss[0] += criterions[0](warped, fixed) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_in = torch.cat((fixed, moving), dim=1)
            warped, flow = model(model_in)

            loss = 0.0
            curr_loss = [0, 0]
            curr_loss[0] += criterions[0](warped, moving) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_time = time.time()
            print(
                "{} of Epoch {}, Loss: {:.4f}, Sim: {:.4f}, Reg: {:.4f}. Cost Time: {:.2f}".format(epoch, max_epoches,
                                                                                                   loss.item(),
                                                                                                   curr_loss[0].item(),
                                                                                                   curr_loss[1].item(),
                                                                                                   end_time - start_time))

        if epoch % 1 == 0:
            torch.save(model, os.path.join("./save_model_pt/oasis", model_name + str(epoch) + ".pt"))


if __name__ == "__main__":
    train()
