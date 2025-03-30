"""

   Collect the grid of results to show the value of the boundary update terms

"""

import numpy as np
import pandas as pd
from einops import rearrange

import torch as th
import torchvision.io as tvio
import torchvision.utils as tv_utils

import bist_cuda

import glob
from pathlib import Path
from run_eval import read_video,read_seg,get_video_names


# -- mp4 --
import imageio


# def collect_results(vid,method,dname,vname,pname,frames,color):
#     root = Path("result/%s/%s/%s/"%(dname,method,pname))/vname
#     vid = []
#     for frame_index in frames:
#         frame_name = "border_%05d.png" % frame_index
#         fname = root / frame_name
#         img = tvio.read_image(fname)/255.
#         vid.append(img)
#     vid = th.from_numpy(np.stack(vid))
#     return vid

def read_spix(root,frames):
    spix = []
    for frame in frames:
        fn = root / ("%05d.csv"%frame)
        spix_f = np.asarray(pd.read_csv(fn,header=None))
        spix.append(th.from_numpy(spix_f))
    spix = th.stack(spix)
    return spix

def collect_results(vid,method,dname,vname,pname,frames,color):

    # -- manage color --
    if color == "grey":
        color = th.tensor([1.0,1.0,1.0])*0.25
    elif color == "blue":
        color = th.tensor([0.0,0.0,1.0])
    elif color == "red":
        color = th.tensor([1.0,0.0,0.0])*1.0

    # -- get marked sequence --
    root = Path("result/%s/%s/%s/"%(dname,method,pname))/vname
    spix = read_spix(root,frames).to(vid.device).int()
    marked = bist_cuda.get_marked_video(vid,spix,color)
    marked = rearrange(marked,'t h w c -> t c h w')

    return marked


def collect_and_save(dname,vname,frames,fmt,save_root):

    # -- read video --
    offs = 0 if dname == "davis" else 1
    vid = read_video(dname,vname)
    vid = th.stack([th.from_numpy(vid[t-offs]) for t in frames])
    vid = vid.contiguous().float().cuda()
    print(vid.shape)
    # vid = rearrange(vid,'t c h w -> t h w c').contiguous().cuda()

    # -- save a reference image --
    if not save_root.exists(): save_root.mkdir(parents=True)
    save_name = save_root/ "ref.png"
    tv_utils.save_image(vid[0].permute(2,0,1),save_name)

    # -- bist --
    method = "bist"
    # pnames = ["param%d"%i for i in range(10,10+4)]
    # pnames = ["param%d"%i for i in range(60,60+4)]
    pnames = ["param60","param63"]
    vids = []
    # cols = ["red","red","red","blue"]
    cols = ["red","blue"]
    for pname,col in zip(pnames,cols):
        vids.append(collect_results(vid,method,dname,vname,pname,frames,col))

    # -- bass --
    method = "bass"
    pname = "param0"
    col = "grey"
    vids = vids + [collect_results(vid,method,dname,vname,pname,frames,col)]


    # -- ensure save root --
    save_root = save_root / ("fmt%d"%fmt)
    if not save_root.exists(): save_root.mkdir(parents=True)

    # -- save grid --
    vids = th.stack(vids)
    vids = vids[...,-450:,-450:]
    # print(vids.shape)
    seq = []
    def prepare_for_seq(img):
        img = np.clip(255.*img.cpu().numpy(),0.,255.).astype(np.uint8)
        img = rearrange(img,'c h w -> h w c')
        return img

    for ix,t in enumerate(frames):

        # -- save name --
        save_name = save_root/ ("%05d.png"%t)


        if fmt == 0:

            #
            # -- format 0 [straight across] --
            #


            # -- padded grid across --
            grid_ix = tv_utils.make_grid(vids[:,ix].flip(dims=[0]),nrow=4)
            _,grid_h,grid_w = grid_ix.shape

            # -- add padding to bass --
            side_pad = 20
            bass = vids[-1,ix]
            _, H, W = bass.shape
            top_pad = (grid_h - H) // 2
            bottom_pad = grid_h - (H + top_pad)
            padded_bass = th.nn.functional.pad(bass, (0, side_pad, top_pad, bottom_pad), value=0)

            # final_image = th.cat([padded_bass, grid_ix], dim=2)
            # tv_utils.save_image(final_image,save_name)

            # -- save name --
            save_name = save_root/ ("bass_%05d.png"%t)
            tv_utils.save_image(bass,save_name)
            save_name = save_root/ ("bist_%05d.png"%t)
            tv_utils.save_image(grid_ix,save_name)

        else:

            #
            # -- format 1 [1 + 2x2]--
            #

            # -- 2x2 grid --
            grid_ix = tv_utils.make_grid(vids[:4,ix].flip(dims=[0]),nrow=2)
            _, grid_h, grid_w = grid_ix.shape

            # -- add padding to bass --
            bass = vids[-1,ix]
            _, H, W = bass.shape
            top_pad = (grid_h - H) // 2
            bottom_pad = grid_h - (H + top_pad)
            side_pad = 10  # Space between grid and img5
            padded_bass = th.nn.functional.pad(bass, (0, side_pad, top_pad, bottom_pad), value=0)

            # -- combine 2x2 and extra image --
            final_image = th.cat([padded_bass, grid_ix], dim=2)

            # -- save --
            seq.append(prepare_for_seq(final_image))
            tv_utils.save_image(final_image,save_name)

    print(f"Saved to {str(save_root)}")

    # -- make the gif --
    if len(seq) != 0:
        fname = save_root / ("%s.mp4"%vname)
        print("Saving movie: ",fname)
        fps = 2.5
        imageio.mimsave(fname, seq, fps=fps, format='mp4', codec='libx264')


def main():

    # # -- example --
    # dname = "davis"
    # vname = "judo"
    # frames = np.arange(0,30+1)
    # save_root = Path("output/viz_results/bndry_updates/")/dname/vname
    # collect_and_save(dname,vname,frames,0,save_root)
    # return

    # -- example --
    dname = "davis"
    vname = "kite-surf"
    frames = np.arange(0,32)
    save_root = Path("output/viz_results/bndry_updates/")/dname/vname
    # collect_and_save(dname,vname,frames,0,save_root)
    # collect_and_save(dname,vname,frames,1,save_root)

    # -- example --
    dname = "davis"
    vname = "soapbox"
    frames = np.arange(35,55+1)
    save_root = Path("output/viz_results/bndry_updates/")/dname/vname
    # collect_and_save(dname,vname,frames,0,save_root)
    # collect_and_save(dname,vname,frames,1,save_root)

    # -- example --
    dname = "davis"
    vname = "horsejump-high"
    frames = np.arange(0,15)
    save_root = Path("output/viz_results/bndry_updates/")/dname/vname
    # collect_and_save(dname,vname,frames,0,save_root)
    # collect_and_save(dname,vname,frames,1,save_root)

    # -- example --
    dname = "davis"
    vname = "parkour"
    frames = np.arange(10,20+1)
    save_root = Path("output/viz_results/bndry_updates/")/dname/vname
    # collect_and_save(dname,vname,frames,0,save_root)
    # collect_and_save(dname,vname,frames,1,save_root)

    # -- example --
    dname = "segtrackerv2"
    vname = "frog_2"
    frames = np.arange(1,40+1)
    save_root = Path("output/viz_results/bndry_updates/")/dname/vname
    collect_and_save(dname,vname,frames,1,save_root)

if __name__ == "__main__":
    main()
