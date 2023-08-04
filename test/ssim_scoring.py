# pip install pytorch-msssim

import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# desired size of the output image
loader = transforms.Compose([
    transforms.Resize((256, 192)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image

from pytorch_msssim import ssim
import os
# from os import listdir
# from os.path import isfile, join

origin = "results/eval_origin"
origin_list = sorted([os.path.join(origin, f) for f in os.listdir(origin) if os.path.isfile(os.path.join(origin, f))])
no_nst = "results/demo/try_on"
no_nst_list = sorted([os.path.join(no_nst, f) for f in os.listdir(no_nst) if os.path.isfile(os.path.join(no_nst, f))])
on_warp_clothes = "results/stylized-warp/try_on/la_muse.jpg"
on_warp_clothes_list = sorted([os.path.join(on_warp_clothes, f) for f in os.listdir(on_warp_clothes) if os.path.isfile(os.path.join(on_warp_clothes, f))])
before_clothes_transfer = "results/stylized-before-warp/try_on/la_muse.jpg"
before_clothes_transfer_list = sorted([os.path.join(before_clothes_transfer, f) for f in os.listdir(before_clothes_transfer) if os.path.isfile(os.path.join(before_clothes_transfer, f))])
after_clothes_transfer = "results/stylized-after-warp/try_on/la_muse.jpg"
after_clothes_transfer_list = sorted([os.path.join(after_clothes_transfer, f) for f in os.listdir(after_clothes_transfer) if os.path.isfile(os.path.join(after_clothes_transfer, f))])
on_warp_no_mask_transfer = "results/stylized-warp-no-backgroundmask/try_on/la_muse.jpg"
on_warp_no_mask_transfer_list = sorted([os.path.join(on_warp_no_mask_transfer, f) for f in os.listdir(on_warp_no_mask_transfer) if os.path.isfile(os.path.join(on_warp_no_mask_transfer, f))])

base_files_list = [origin_list, no_nst_list]
eval_files_list = [on_warp_clothes_list, before_clothes_transfer_list, after_clothes_transfer_list]
# base_files_list = [origin_list]
# eval_files_list = [no_nst_list]
for base_files in base_files_list:

    for eval_files in eval_files_list:
        print(base_files[0], eval_files[0])
        num_files = 500.0
        total_ssim = 0.0

        for i,(b_file, e_file) in enumerate(zip(base_files, eval_files)):
            # print(b_file,e_file)
            base = image_loader(b_file)
            eval = image_loader(e_file)
            ssim_val = ssim( base, eval, data_range=255, size_average=False)
            # print(ssim_val.numpy())
            # if i%100 == 0:
            #     print(i, ssim_val.numpy())
            total_ssim += ssim_val.numpy()
        avg_ssim = total_ssim/num_files
        print(avg_ssim)
        print()

