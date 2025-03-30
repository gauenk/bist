"""

Get some cropped seq

"""

import tqdm
import os,shutil,random
import subprocess
import colorsys
from pathlib import Path

import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange,repeat
import torchvision.io as tvio
import torchvision.utils as tv_utils


def main():

    vname = "kid-football"
    def read_image(fname):
        return tvio.read_image(fname)/255.
    def save_image(img,fname):
        tv_utils.save_image(img,fname)
    def crop_image(img,crop):
        hs,he,ws,we = crop
        return img[:,hs:he,ws:we]

    # crop = [320-25,320-25+200,280-50,280-50+200]
    # frame = 3
    # crop = [65,65+200,185,185+200]
    frame = 4
    crop = [105,105+200,220,220+200]
    save_root = Path("output/plots/poster/split_merge_relabel/%s"%vname)
    if not save_root.exists(): save_root.mkdir(parents=True)

    split0 = read_image("result/davis/bist/logged/param0/%s/anim_split/%05d/00000.png"%(vname,frame))
    save_image(crop_image(split0,crop),save_root/"split0.png")
    split1 = read_image("result/davis/bist/logged/param0/%s/anim_split/%05d/00002.png"%(vname,frame))
    save_image(crop_image(split1,crop),save_root/"split1.png")

    merge0 = read_image("result/davis/bist/logged/param0/%s/anim_merge/%05d/00000.png"%(vname,frame))
    save_image(crop_image(merge0,crop),save_root/"merge0.png")
    merge1 = read_image("result/davis/bist/logged/param0/%s/anim_merge/%05d/00002.png"%(vname,frame))
    save_image(crop_image(merge1,crop),save_root/"merge1.png")

    if frame == 1:
        relabel0 = read_image("result/davis/bist/logged/param0/%s/anim_bndy/%05d/00400.png"%(vname,frame-1))
    else:
        relabel0 = read_image("result/davis/bist/logged/param0/%s/anim_bndy/%05d/00128.png"%(vname,frame-1))
    # relabel0 = read_image("result/davis/bist/logged/param0/%s/anim_saf/%05d/00006.png"%(vname,frame))
    # crop[0] -= 50
    # crop[2] -= 50
    save_image(crop_image(relabel0,crop),save_root/"relabel0.png")
    relabel1 = read_image("result/davis/bist/logged/param0/%s/anim_saf/%05d/00007.png"%(vname,frame))
    save_image(crop_image(relabel1,crop),save_root/"relabel1.png")
    relabel2 = read_image("result/davis/bist/logged/param0/%s/anim_relabel/%05d/00000.png"%(vname,frame))
    save_image(crop_image(relabel2,crop),save_root/"relabel2.png")

if __name__ == "__main__":
    main()
