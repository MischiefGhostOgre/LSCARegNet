import os
import time
import glob
import argparse

from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_vol
from utils import dice, jacobian_determinant
from lscaregnet.lscaregnet import SpatialTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", type=str, default="./save_model_pt/oasis/ours_best.pt")
parser.add_argument("--test_data", type=str, default="/public/home/gyf/registration/data/OASIS/Test/")
parser.add_argument("--gpu_id", type=str, default="0")

args = parser.parse_args()
load_model = args.load_model
test_data = args.test_data
gpu_id = args.gpu_id


def test():
    test_vol_dataset = glob.glob(test_data + "/Vols/*.nii.gz")
    test_seg_dataset = glob.glob(test_data + "/Segs/*.nii.gz")

    model_stn = SpatialTransformer((160, 192, 224), "nearest")
    model = torch.load(load_model)
    device = torch.device("cuda:" + gpu_id)

    model.to(device)
    model_stn.to(device)

    mean_dice_list = []
    mean_jac_list = []
    init_dice_list = []
    mean_time_list = []

    for moving_vol_name, moving_seg_name in zip(test_vol_dataset, test_seg_dataset):
        for fixed_vol_name, fixed_seg_name in zip(test_vol_dataset, test_seg_dataset):
            if moving_vol_name == fixed_vol_name:
                continue
            moving_vol, fixed_vol = load_vol(moving_vol_name), load_vol(fixed_vol_name)
            moving_seg, fixed_seg = load_vol(moving_seg_name), load_vol(fixed_seg_name)

            good_labels = np.intersect1d(moving_seg, fixed_seg)[1:32]

            dice_val = dice(fixed_seg, moving_seg, good_labels)
            init_dice_list.append(np.mean(dice_val))

            moving_vol = torch.from_numpy(moving_vol).to(device)[None, None, ...].float()
            moving_seg = torch.from_numpy(moving_seg).to(device)[None, None, ...].float()
            fixed_vol = torch.from_numpy(fixed_vol).to(device)[None, None, ...].float()
            fixed_seg = torch.from_numpy(fixed_seg).to(device)[None, None, ...].float()

            model_in = torch.cat((moving_vol, fixed_vol), dim=1)
            before_time = time.time()
            _, flow = model(model_in)
            after_time = time.time()
            warp_seg = model_stn(moving_seg, flow)

            fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
            warp_seg = warp_seg.detach().cpu().numpy().squeeze()
            flow = flow.squeeze().permute(1, 2, 3, 0).cuda().data.cpu().numpy()
            dice_val = dice(fixed_seg, warp_seg, good_labels)
            mean_dice_list.append(np.mean(dice_val))
            mean_jac_list.append(np.sum(jacobian_determinant(flow) <= 0))
            mean_time_list.append(after_time - before_time)

    print(np.mean(init_dice_list), np.std(init_dice_list))
    print(np.mean(mean_dice_list), np.std(mean_dice_list))
    print(np.mean(mean_jac_list), np.std(mean_jac_list))
    print(np.mean(mean_time_list), np.std(mean_time_list))


if __name__ == "__main__":
    with torch.no_grad():
        test()
