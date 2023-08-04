# Basically test.py from https://github.com/SenHe/Flow-Style-VTON, but with our options

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

# imports
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils

from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
from util import flow_util

import os
import numpy as np

from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader

from os import listdir
from os.path import isfile, join

# NEXT: based on: https://github.com/SenHe/Flow-Style-VTON

name = "demo"
resize_or_crop = None
batchSize = "1"
gpu_ids = "0"
warp_checkpoint = "checkpoints/ckp/non_aug/PFAFN_warp_epoch_101.pth"
gen_checkpoint = "checkpoints/ckp/non_aug/PFAFN_gen_epoch_101.pth"
dataroot = "dataset2"

testOptions = TestOptions()
testOptions.initialize()
testOptions.opt = testOptions.parser.parse_args(["--name",name,"--resize_or_crop",resize_or_crop,"--batchSize",batchSize,"--gpu_ids",gpu_ids,"--warp_checkpoint", warp_checkpoint, "--gen_checkpoint",gen_checkpoint,"--dataroot", dataroot])
testOptions.opt.isTrain = testOptions.isTrain   # train or test

str_ids = testOptions.opt.gpu_ids.split(',')
testOptions.opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        testOptions.opt.gpu_ids.append(id)

if len(testOptions.opt.gpu_ids) > 0:
    torch.cuda.set_device(testOptions.opt.gpu_ids[0])

# print(vars(options.opt))
for k, v in sorted(vars(testOptions.opt).items()):
    print('%s: %s' % (str(k), str(v)))

opt = testOptions.opt

def de_offset(s_grid):
    [b,_,h,w] = s_grid.size()

    x = torch.arange(w).view(1, -1).expand(h, -1).float()
    y = torch.arange(h).view(-1, 1).expand(-1, w).float()
    x = 2*x/(w-1)-1
    y = 2*y/(h-1)-1
    grid = torch.stack([x,y], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    offset = grid - s_grid

    offset_x = offset[:,0,:,:] * (w-1) / 2
    offset_y = offset[:,1,:,:] * (h-1) / 2

    offset = torch.cat((offset_y,offset_x),0)
    
    return  offset

f2c = flow_util.flow2color()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print("dataset_size =",dataset_size)

warp_model = AFWM(opt, 3)
# print(warp_model) # don't print model
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
#print(gen_model)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)

start_epoch, epoch_iter = 1, 0

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize


for epoch in range(1,2):
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        real_image = data['image']
        clothes = data['clothes']
        # edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = clothes * edge        

        #import ipdb; ipdb.set_trace()

        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')
        
        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        #sub_path = path + '/PFAFN'
        #os.makedirs(sub_path,exist_ok=True)
        print(i, data['p_name'])

        if step % 1 == 0:
            
            ## save try-on image only

            # utils.save_image(
            #     p_tryon,
            #     os.path.join('./our_t_results', data['p_name'][0]),
            #     nrow=int(1),
            #     normalize=True,
            #     range=(-1,1),
            # )
            
            # save try-on image only

            os.makedirs(path+"/try_on", exist_ok=True)
            utils.save_image(
               p_tryon,
               os.path.join(path+"/try_on", data['p_name'][0]),
               nrow=int(1),
               normalize=True,
               range=(-1,1),
            )
            
            # # save person image, garment, flow, warped garment, and try-on image
            
            # a = real_image.float().cuda()
            # b = clothes.cuda()
            # flow_offset = de_offset(last_flow)
            # flow_color = f2c(flow_offset).cuda()
            # c = warped_cloth.cuda()
            # d = p_tryon
            # combine = torch.cat([a[0],b[0], flow_color, c[0], d[0]], 2).squeeze()
            # utils.save_image(
            #    combine,
            #    os.path.join(path, data['p_name'][0]),
            #    nrow=int(1),
            #    normalize=True,
            #    range=(-1,1),
            # )
            
        step += 1
        if epoch_iter >= dataset_size:
            break