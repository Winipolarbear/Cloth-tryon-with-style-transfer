from __future__ import print_function
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.utils import save_image

import copy

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
# imsize = 256, 192 if torch.cuda.is_available() else 128  # use small size if no gpu

#
# followed: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
#
loader = transforms.Compose([
    transforms.Resize((256, 192)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()  # reconvert into PIL image

normalizer = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    # image = image[:,:, 80:100, 50:57]
    # print(image)
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
    
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
# importing vgg-19 pre-trained model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# given by tutorial
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default,
                               content_mask = None):
    cnn_copy = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn_copy.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # NOTE: masking the losses seems to make the result way worse
            if content_mask is not None:
              content_img = content_img * content_mask.detach()
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            if content_mask is not None:
              style_img = style_img * content_mask.detach()
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    del cnn_copy
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=250,
                       style_weight=1000000, content_weight=1, content_mask=None):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, content_mask=content_mask)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)
    
    del model
    return input_img

#
# end followed: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
#

def style_transfer_with_mask(style_img, content_img, content_mask=None, content_weight=1, style_weight=1000000):
    """
    Simple Segmented Style Transfer, if content mask is provided
    """
    if content_mask is not None:
        input_img = (style_img*(1-content_mask)+content_img).clone()
        content_img2 = input_img.clone()
    else:
        input_img = content_img.clone()
        content_img2 = content_img.clone()

    styled_output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img2, style_img, input_img, style_weight=style_weight, content_weight=content_weight, content_mask=content_mask)
        
    gc.collect()
    torch.cuda.empty_cache()
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
    return styled_output*content_mask


# NEXT: based on: https://github.com/SenHe/Flow-Style-VTON

name = "stylized-warp"
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

# print([sorted([f for f in listdir(f"{dataroot}/styles") if isfile(join(f"{dataroot}/styles", f))])])
for style_f in ["la_muse.jpg"]: # only uses la_muse.jpg style
    style_path = f"{dataroot}/styles/{style_f}"
    print("\n-----\nStyle with:",style_path)
    
    for epoch in range(1,2): # 1 epoch
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            real_image = data['image']
            clothes = data['clothes']
            ##edge is extracted from the clothes image with the built-in function in python
            edge = data['edge']
            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
            clothes = clothes * edge        

            #import ipdb; ipdb.set_trace()

            flow_out = warp_model(real_image.cuda(), clothes.cuda())
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                            mode='bilinear', padding_mode='zeros')
            normalizer = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)) # this is the default normalizer used by the flow-style-vton
            style_img = image_loader(style_path).cuda()
            
            masked_style_clothes = style_transfer_with_mask(style_img, warped_cloth.detach().cuda(), content_mask = warped_edge.detach().cuda())
            masked_style_clothes = warped_edge * masked_style_clothes
            normalized_masked_style_clothes = normalizer(masked_style_clothes.squeeze(0)).unsqueeze(0)
            normalized_style = normalizer(style_img.squeeze(0)).unsqueeze(0)

            cloth_style_background = (normalized_style*(1-warped_edge.detach().cuda())+warped_cloth.detach().cuda())

            gen_inputs = torch.cat([real_image.cuda(), normalized_masked_style_clothes, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = normalized_masked_style_clothes * m_composite + p_rendered * (1 - m_composite)

            path = 'results/' + opt.name
            os.makedirs(path, exist_ok=True)
            #sub_path = path + '/PFAFN'
            #os.makedirs(sub_path,exist_ok=True)
            print(i, data['p_name'], time.time()-iter_start_time)

            if step % 1 == 0:
                
                ## save try-on image only

                # utils.save_image(
                #     p_tryon,
                #     os.path.join('./our_t_results', data['p_name'][0]),
                #     nrow=int(1),
                #     normalize=True,
                #     range=(-1,1),
                # )
                
                
                os.makedirs(path+"/try_on"+f"/{style_f}", exist_ok=True)
                utils.save_image(
                    p_tryon,
                    os.path.join(path+"/try_on"+f"/{style_f}", data['p_name'][0]),
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
                # d = normalized_style.cuda()
                # e = normalized_masked_style_clothes.cuda()
                # f = p_tryon
                # os.makedirs(path+"/detailed", exist_ok=True)
                # combine = torch.cat([a[0],b[0], flow_color, c[0], d[0], e[0], f[0]], 2).squeeze()
                # utils.save_image(
                #     combine,
                #     os.path.join(path+"/detailed", data['p_name'][0]),
                #     nrow=int(1),
                #     normalize=True,
                #     range=(-1,1),
                # )

                # combine2 = torch.cat([a[0], b[0], d[0], c[0], e[0], f[0]], 2).squeeze()
                # os.makedirs(path+"/important", exist_ok=True)
                # utils.save_image(
                #     combine2,
                #     os.path.join(path+"/important", data['p_name'][0]),
                #     nrow=int(1),
                #     normalize=True,
                #     range=(-1,1),
                # )

                # combine3 = torch.cat([a[0], b[0], d[0]], 2).squeeze()
                # os.makedirs(path+"/inputs", exist_ok=True)
                # utils.save_image(
                #     combine3,
                #     os.path.join(path+"/inputs", data['p_name'][0]),
                #     nrow=int(1),
                #     normalize=True,
                #     range=(-1,1),
                # )


                # os.makedirs(path+"/background", exist_ok=True)
                # utils.save_image(
                #     cloth_style_background,
                #     os.path.join(path+"/background", data['p_name'][0]),
                #     nrow=int(1),
                #     normalize=True,
                #     range=(-1,1),
                # )
                
            step += 1
            if epoch_iter >= dataset_size:
                break

# stylized_vton_on_warp()