"""

   Vizualize the missing superpixel density

"""


import tqdm
import os,shutil,random
import subprocess
import colorsys

import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange,repeat
import torchvision.io as tvio
import torchvision.utils as tv_utils

import glob
from pathlib import Path
# from run_eval import read_video,read_seg,get_video_names
from st_spix.utils import rgb2lab
from st_spix.spix_utils.updated_io import read_video,read_seg
from st_spix.spix_utils.evaluate import computeSummary,scoreSpixPoolingQualityByFrame,count_spix,read_spix
from st_spix.spix_utils.evaluate import get_video_names

import bist_cuda

import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def read_image(fname):
    return tvio.read_image(fname)/255.
def csv_to_th(fname):
    return th.from_numpy(pd.read_csv(str(fname),header=None).to_numpy())
def save_image(img,fname):
    tv_utils.save_image(img,fname)
def crop_image(img,crop):
    hs,he,ws,we = crop
    return img[...,hs:he,ws:we]
def show_only_spixid(img,spix,spix_id,alpha):
    if isinstance(spix_id,int): spix_id = th.tensor([spix_id])
    print(img.shape,spix.shape)
    # we want a border included
    spix = spix[0]
    H,W = spix.shape
    mask = th.isin(spix,spix_id)
    expanded_mask = mask.clone()
    for di in [-1, 1]:  # Shift up/down
        for dj in [-1, 1]:  # Shift left/right
            shifted_mask = th.zeros_like(mask)
            if di == -1:
                shifted_mask[:H-1, :] = mask[1:, :]
            elif di == 1:
                shifted_mask[1:, :] = mask[:H-1, :]
            if dj == -1:
                shifted_mask[:, :W-1] |= mask[:, 1:]
            elif dj == 1:
                shifted_mask[:, 1:] |= mask[:, :W-1]
            expanded_mask |= shifted_mask
    return th.where(expanded_mask.unsqueeze(0),img,alpha*img)

def fill_invalid(tensor,bkg,spix):
    C,H,W = tensor.shape
    bkg = repeat(bkg,'1 h w -> c (hr h) (wr w)',hr=2,wr=2,c=3)[:,:H,:W].cpu()
    bkg[0] = 1.
    bkg[1] = 0.
    bkg[2] = 1.
    assert bkg.shape[1] == H and bkg.shape[2] == W
    tensor = th.where(spix==-1,bkg,tensor)
    return tensor

def get_marked_video(vid,spix,color):
    vid = vid.cuda().contiguous().float()
    spix = spix.cuda().contiguous().int()
    color = color.cuda().contiguous().float()
    vid = rearrange(vid,'t c h w -> t h w c').contiguous()
    marked = bist_cuda.get_marked_video(vid,spix,color)
    marked = rearrange(marked,'t h w c -> t c h w')
    return marked

def main():

    dname = "davis"
    pname = "param0"
    vname = "kid-football"


    offset = 0 if dname == "davis" else 1
    save_root = Path("output/plots/viz_crop_result/%s"%vname)
    if not save_root.exists(): save_root.mkdir(parents=True)

    frames = [0,3,5]
    crop = [0,485,110,710]

    # -- save crops --
    for frame in frames:
        img = read_image("result/davis/bist/param0/%s/border_%05d.png"%(vname,frame))
        img = crop_image(img,crop)
        save_image(img,save_root/("border_%05d.png"%frame))


if __name__ == "__main__":
    main()
