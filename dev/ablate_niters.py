import os,re

import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange
import torchvision.utils as tv_utils

from pathlib import Path

from run_eval import read_video,read_seg,get_video_names

def run_command(cmd):
    import subprocess
    print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    olines = output.split("\n")
    print(olines[-3:])
    match = re.search(r'Mean Time:\s([0-9]*\.?[0-9]+)', olines[-2])
    runtime = float(match.group(1))
    return output,runtime

def run_exp(method):
    vname = "dance-twirl"
    vmode = 1 if method == "bist" else 0
    niters_list = [4,8,12,20]
    for niters in niters_list:
        cmd = "./build/demo -n 20 -d /home/gauenk/Documents/data/davis/DAVIS/JPEGImages/480p/%s/ -f /home/gauenk/Documents/data/davis/DAVIS/JPEGImages/480p/%s/BIST_flows/ -o result/ablate/niters%d/%s/ --read_video %d --img_ext jpg --sigma_app 0.009 --potts 10.00 --alpha 0.0 --split_alpha 2.000 --iperc_coeff 2.0 --tgt_nspix 0 --niters %d --vid_niters %d" % (vname,vname,niters,method,vmode,niters,niters)
        run_command(cmd)


def crop_and_save(imgs,crop,root):

    # -- path --
    root = Path(root)
    if not root.exists(): root.mkdir(parents=True)

    # -- .. --
    def apply_crops(hs,he,ws,we,*vids):
        crops = []
        for vid in vids:
            crops.append(vid[...,hs:he,ws:we])
        return crops

    # -- run crops --
    crops = apply_crops(*crop,*imgs)
    print(crops[0].shape,crops[0].shape[-2]/(1.*crops[0].shape[-1]))

    # -- save images --
    for ix in range(len(imgs)):
        fn_ix = root / ("%05d.png" % ix)
        tv_utils.save_image(crops[ix],fn_ix)

def format_results(method):

    # frame_index = 5 # good one
    frame_index = 10
    niters = [4,8,12,20]
    fname_fmt = "/home/gauenk/Documents/packages/st_spix_refactor/result/ablate/niters%d/%s/pooled_%05d.png"

    imgs = []
    for niter in niters:
        fname = fname_fmt % (niter,method,frame_index)
        img = np.array(Image.open(fname).convert("RGB"))
        img = rearrange(img,'h w c -> c h w')/255.
        imgs.append(th.from_numpy(img))
    print(imgs[0].shape)

    # -- read video --
    img = read_video("davis","dance-twirl")[frame_index]
    img = rearrange(img,'h w c -> c h w')
    imgs.append(th.from_numpy(img))


    # -- crop and save --
    root = "output/ablate_niters/%s/crop0"%method
    # crop = [80,180,560-20,660+20+20+20+20] # nice bigger crop
    # crop = [85,170,630,680]
    crop = [85,170,628,683]
    crop_and_save(imgs,crop,root)


    root = "output/ablate_niters/%s/crop1"%method
    crop = [95,195,535,640-40]
    crop_and_save(imgs,crop,root)

def main():
    # run_exp("bass")
    # run_exp("bist")

    format_results("bass")
    format_results("bist")


if __name__ == "__main__":
    main()
