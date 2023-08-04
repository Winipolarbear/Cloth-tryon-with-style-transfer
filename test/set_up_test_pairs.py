# set up test_pairs.txt

import os
from os import listdir
from os.path import isfile, join

dataroot = "dataset2" # TODO: Change as needed

import random
random.seed(42)

with open("test_pairs.txt", "w") as f:
    person_imgs = sorted([f for f in listdir(f"{dataroot}/test_img") if isfile(join(f"{dataroot}/test_img", f))])
    # person_imgs = sorted(os.listdir(f"{dataroot}/test_img"))
    clothes_imgs = sorted([f for f in listdir(f"{dataroot}/test_clothes") if isfile(join(f"{dataroot}/test_clothes", f))])
    # clothes_imgs = sorted(os.listdir(f"{dataroot}/test_clothes"))
    random.shuffle(person_imgs)
    random.shuffle(clothes_imgs)
    person_imgs = person_imgs[:500] # 500 random people
    clothes_imgs = clothes_imgs[:500] # 500 random clothes
    print(person_imgs)
    print(clothes_imgs)

    for i in range(len(person_imgs)):
        if person_imgs[i].endswith("jpg") and clothes_imgs[i].endswith("jpg"):
            write_line = f"{person_imgs[i]} {clothes_imgs[i]}\n"
            # write_line = pic.split("_")[0]
            f.write(write_line)