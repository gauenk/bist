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


def get_split_info(root,frame,split_ix):

    # -- read --
    split_root = root/"log/split/" / ("%05d" % frame)
    proposed_spix = csv_to_th(split_root / ("%05d.csv" % (4*split_ix)))[None,].cuda()
    init_spix = csv_to_th(split_root / ("%05d.csv" % (4*split_ix+1)))[None,].cuda()
    accepted_spix = csv_to_th(split_root / ("%05d.csv" % (4*split_ix+3)))[None,].cuda()

    # -- ensure the proposed side is the accepted side [it can swap in the code] --
    cond0 = accepted_spix != proposed_spix
    cond1 = accepted_spix != init_spix
    args = th.where(th.logical_and(cond0,cond1))
    old_spix = th.minimum(accepted_spix[args],proposed_spix[args])
    new_spix = th.maximum(accepted_spix[args],proposed_spix[args])
    pairs = th.stack([old_spix,new_spix],-1)
    pairs = th.unique(pairs,dim=0)
    for old,new in pairs:
        args_old = th.where(accepted_spix == old)
        args_new = th.where(accepted_spix == new)
        proposed_spix[args_old] = old
        proposed_spix[args_new] = new

    return proposed_spix.cpu(),init_spix.cpu(),accepted_spix.cpu()

def main():

    dname = "davis"
    pname = "param0"
    vname = "kid-football"

    # crop = [320-25,320-25+200,280-50,280-50+200]
    # frame = 3
    # crop = [65,65+200,185+20,185+200+20]

    frame = 4
    crop = [105,105+200,220,220+200]
    # frame = 2
    # crop = [45,45+200,220,220+200]

    offset = 0 if dname == "davis" else 1
    img = th.from_numpy(read_video(dname,vname)[frame-offset])
    img = rearrange(img,'h w c -> c h w')

    save_root = Path("output/plots/poster/viz_new_split/%s"%vname)
    if not save_root.exists(): save_root.mkdir(parents=True)

    ishifted = read_image("result/davis/bist/logged/param0/%s/anim_saf/%05d/00007.png"%(vname,frame))
    split0 = read_image("result/davis/bist/logged/param0/%s/anim_split/%05d/00000.png"%(vname,frame))
    # save_image(crop_image(split0,crop),save_root/"split0.png")
    split1 = read_image("result/davis/bist/logged/param0/%s/anim_split/%05d/00002.png"%(vname,frame))
    # save_image(crop_image(split1,crop),save_root/"split1.png")
    shifted = csv_to_th("result/davis/bist/logged/param0/%s/log/shifted/%05d/00006.csv"%(vname,frame))

    # -- read split info --
    root = Path("result/davis/bist/logged/%s/%s/"%(pname,vname))
    prop,init,acc = get_split_info(root,frame,0)

    # -- crop info with text header --
    split1 = crop_image(split1,crop)
    split0 = crop_image(split0,crop)
    ishifted = crop_image(ishifted,crop)

    # -- crop info without text header --
    crop[0],crop[1] = crop[0]-45,crop[1]-45
    img = crop_image(img,crop)
    shifted = crop_image(shifted,crop)
    prop,init,acc = crop_image(prop,crop),crop_image(init,crop),crop_image(acc,crop)

    # -- overlay invalid --
    max_spix = init.max().item()
    alpha = 0.4
    color = th.tensor([0.0,0.0,1.0])*0.7
    print(shifted.shape,ishifted.shape,img.shape)

    # -- nice background --
    bkg = tvio.read_image("data/transparent_png.jpg").cuda()/255.
    split0_m = fill_invalid(img,bkg,shifted)
    # split0_m = th.where((shifted==-1).unsqueeze(0),ishifted,img)
    # print(split0_m.shape,img.shape,prop.shape)
    # print(split0_m.device,img.device,prop.device)
    split0_m = th.where(max_spix>prop,split0_m,(1-alpha)*split0_m+alpha*color.view(3,1,1))
    # split0_m = th.where((shifted==-1).unsqueeze(0),ishifted,split0)

    color = th.tensor([1.0,1.0,1.0]).cuda()*0.7 # boundary color
    print(split0_m.shape)
    split0_m = get_marked_video(split0_m[None,:].cuda(),init.cuda(),color.cuda()).cpu()
    save_image(split0_m,save_root/"split0_m.png")
    save_image(split0,save_root/"split0.png")
    save_image(ishifted,save_root/"ishifted.png")

    # crop = [50,120,40,110]
    alpha = 0.3
    crop = [50,90,33,73]
    split0_m_c = crop_image(split0_m,crop)
    init_m = crop_image(init,crop)
    # spix_ids = th.unique(init_m)
    spix_id = 380
    split0_m_c = show_only_spixid(split0_m_c,init_m,spix_id,alpha)
    save_image(split0_m_c,save_root/"split0_m_c0.png")

    crop = [148,148+50,92,92+50]
    split0_m_c = crop_image(split0_m,crop)
    init_m = crop_image(init,crop)
    # spix_ids = th.unique(init_m)
    # print(spix_ids)
    spix_id = 1023
    split0_m_c = show_only_spixid(split0_m_c,init_m,spix_id,alpha)
    save_image(split0_m_c,save_root/"split0_m_c1.png")

    crop = [8,35+8,15,50]
    split0_m_c = crop_image(split0_m,crop)
    init_m = crop_image(init,crop)
    spix_id = 590#th.unique(init_m)[3]
    split0_m_c = show_only_spixid(split0_m_c,init_m,spix_id,alpha)
    save_image(split0_m_c,save_root/"split0_m_c2.png")

    spix_ids = th.tensor([380,1023,590])
    split0_m_c = show_only_spixid(split0_m,init,spix_ids,alpha)
    save_image(split0_m_c,save_root/"split0_m.png")

    # print(shifted.shape)
    # print(split0.shape)
    # print(shifted)


if __name__ == "__main__":
    main()
