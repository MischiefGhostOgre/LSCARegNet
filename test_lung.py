import os
import time
import glob
import argparse

import nibabel as nib
from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_vol
from utils import point_spatial_transformer

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", type=str, default="./save_model_pt/dirlab/ours_best.pt")
parser.add_argument("--model_name", type=str, default="lcamorph")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--case_id", type=int, default=8)
parser.add_argument("--test_data", type=str, default="/public/home/gyf/registration/data/DIRLAB/Case")
args = parser.parse_args()
load_model = args.load_model
model_name = args.model_name
test_data = args.test_data
gpu_id = args.gpu_id
case_id = args.case_id


def test():
    global case_id
    alpha_list = [0.97, 0.97, 1.16, 1.15, 1.13, 1.10, 0.97, 0.97, 0.97, 0.97]
    alpha = alpha_list[case_id]
    case_id = str(case_id)
    beta = 0.625
    alpha /= beta

    mov_name = test_data + case_id + "/new_vols/case" + case_id + "_T00.nii.gz"
    fix_name = test_data + case_id + "/new_vols/case" + case_id + "_T50.nii.gz"
    mov_p_name = test_data + case_id + "/new_landmarks/case" + case_id + "_T00.txt"
    fix_p_name = test_data + case_id + "/new_landmarks/case" + case_id + "_T50.txt"

    device = "cuda:" + gpu_id
    avg_tre = 0
    std_tre = 0
    model = torch.load(load_model, map_location=device)
    #model.eval()
    with torch.no_grad():

        mov_img = nib.load(mov_name).get_fdata()
        fix_img = nib.load(fix_name).get_fdata()
        moving_image_landmarks = np.loadtxt(mov_p_name)
        fixed_image_landmarks = np.loadtxt(fix_p_name)
        init_tre_list = []
        before_distance = fixed_image_landmarks - moving_image_landmarks
        for ii in range(300):
            init_tre_list += [np.linalg.norm(before_distance[ii])]
        print(np.mean(init_tre_list) * alpha, np.std(init_tre_list) * alpha)

        mov_img = torch.from_numpy(mov_img).to(device)
        fix_img = torch.from_numpy(fix_img).to(device)

        mov_img = mov_img[None, None, ...].float()
        fix_img = fix_img[None, None, ...].float()

        model_in = torch.cat((mov_img, fix_img), dim=1)
        warped, flow = model(model_in)

        fixed_image_landmarks = fixed_image_landmarks[np.newaxis, ...]
        moving_image_landmarks = moving_image_landmarks[np.newaxis, ...]
        flow = flow.squeeze().permute(1, 2, 3, 0)
        dvf_Data = flow.cuda().data.cpu().numpy().squeeze()
        data = [torch.from_numpy(t).cuda() for t in [fixed_image_landmarks, dvf_Data]]
        moved_image_landmarks = point_spatial_transformer(data)

        moving_image_landmarks = torch.from_numpy(moving_image_landmarks).cuda()
        after_distance = moved_image_landmarks - moving_image_landmarks
        after_distance = after_distance.cuda().data.cpu().numpy().squeeze()

        reg_tre_list = []
        for i in range(300):
            reg_tre_list += [np.linalg.norm(after_distance[i])]
        print(np.mean(reg_tre_list) * alpha, np.std(reg_tre_list) * alpha)
        avg_tre += np.mean(reg_tre_list) * alpha
        std_tre += np.std(reg_tre_list) * alpha

        #######################################
        #######################################
        #######################################
        mov_img = nib.load(fix_name).get_fdata()
        fix_img = nib.load(mov_name).get_fdata()
        moving_image_landmarks = np.loadtxt(fix_p_name)
        fixed_image_landmarks = np.loadtxt(mov_p_name)
        init_tre_list = []
        before_distance = fixed_image_landmarks - moving_image_landmarks
        for ii in range(len(before_distance)):
            init_tre_list += [np.linalg.norm(before_distance[ii])]

        mov_img = torch.from_numpy(mov_img).to(device)
        fix_img = torch.from_numpy(fix_img).to(device)

        mov_img = mov_img[None, None, ...].float()
        fix_img = fix_img[None, None, ...].float()

        model_in = torch.cat((mov_img, fix_img), dim=1)
        warped, flow = model(model_in)

        fixed_image_landmarks = fixed_image_landmarks[np.newaxis, ...]
        moving_image_landmarks = moving_image_landmarks[np.newaxis, ...]
        flow = flow.squeeze().permute(1, 2, 3, 0)
        dvf_Data = flow.cuda().data.cpu().numpy().squeeze()
        data = [torch.from_numpy(t).cuda() for t in [fixed_image_landmarks, dvf_Data]]
        moved_image_landmarks = point_spatial_transformer(data)

        # np.savetxt('./save_txt/ans_' + model_name + "_" + str(case_id) + '.txt', moved_image_landmarks.cuda().data.cpu().numpy().squeeze(), delimiter=" ", fmt='%.3f')
        moving_image_landmarks = torch.from_numpy(moving_image_landmarks).cuda()
        after_distance = moved_image_landmarks - moving_image_landmarks
        after_distance = after_distance.cuda().data.cpu().numpy().squeeze()

        reg_tre_list = []
        for i in range(300):
            reg_tre_list += [np.linalg.norm(after_distance[i])]
        print(np.mean(reg_tre_list) * alpha, np.std(reg_tre_list) * alpha)
        avg_tre += np.mean(reg_tre_list) * alpha
        std_tre += np.std(reg_tre_list) * alpha

    print("mean", avg_tre / 2, "std", std_tre / 2)


if __name__ == "__main__":
    test()
