import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange


import glob
from pathlib import Path
from run_eval import read_video,read_seg
from st_spix.utils import vid_rgb2lab_th
from st_spix.spix_utils.evaluate import computeSummary,scoreSpixPoolingQualityByFrame,count_spix,read_spix,read_anno_video

import numpy as np
import matplotlib.pyplot as plt
import argparse


# Argument parser to get image index from command line
parser = argparse.ArgumentParser(description="Hover over a segmented image and get the class ID.")
parser.add_argument("--index", type=int, default=0, help="Index of the image to load")
args = parser.parse_args()
index = args.index

def get_info(method,vname):
    dname = "davis"
    root = Path("result_davis/%s/param0/%s/"%(method,vname))
    vid = read_video(dname,vname)
    anno = read_anno_video(root,vname,0)
    spix = read_spix(root,vname,0)
    return vid,anno,spix

vid,anno,spix = get_info("bist","breakdance")
# vid = vid.cpu().numpy()
# spix = spix.cpu().numpy()
# index = 6
# vid = vid[:,:,:480]
# anno = anno[:,:,:480]
img = vid[index]
anno = anno[index]

img = img.astype(np.double)
img_lab = vid_rgb2lab_th(rearrange(img,'h w c -> 1 c h w'),False)[0]
img_lab = rearrange(img_lab,'c h w -> h w c').double().clone().numpy()
print("img_lab.shape: ",img_lab.shape)
print("img.shape: ",img.shape)
spix = spix[index]
print("spix.shape: ",spix.shape)
print(img[0,28])
print(img_lab[0,28])
# print(spix[35:45,330:340])
# exit()
for class_id in range(20):
# for class_id in [1]:
    args = np.where(spix==class_id)
    num = len(args[0])
    spix_mean = [img_lab[...,i][args].mean().item() for i in range(3)]
    # spix_mean = [img_lab[...,i][args].sum().item() for i in range(3)]
    loc_mean = np.flip(np.mean(np.stack(args),-1))
    # print(img_lab[...,1][args])
    print(class_id,num,["%2.4f"%x for x in spix_mean],["%2.2f"%x for x in loc_mean])

# exit()
fig, ax = plt.subplots()
ax.imshow(anno)
title = ax.set_title("Hover over the image to see class ID")

def on_hover(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < spix.shape[1] and 0 <= y < spix.shape[0]:
            class_id = spix[y, x]  # Row first in NumPy
            args = np.where(spix==class_id)
            spix_mean = [img_lab[...,i][args].mean().item() for i in range(3)]
            title.set_text(f"Class ID: {class_id} "+str(spix_mean))
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)
plt.show()
