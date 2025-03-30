"""

   Collect the grid of results to show the value of the boundary update terms

"""

import numpy as np
import pandas as pd
from einops import rearrange

import torch as th
import torchvision.io as tvio
import torchvision.utils as tv_utils
import torchvision.transforms.functional as F

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

def color_spix(vid,spix,anno,color,alpha,method="bist"):

    if method == "bist":
        spix_ids = th.unique(spix[0][th.where(anno[0]>0)])
        mask = th.zeros_like(spix)
        args = th.where(th.isin(spix,spix_ids))
        mask[args] = 1
        mask = mask[:,None]
        mask_rgb = th.tensor(color, device=vid.device).view(1, 3, 1, 1) * mask
        vid = th.where(mask_rgb.any(1,keepdim=True) > 0, (1 - alpha) * vid + alpha * mask_rgb, vid)
    else:
        for t in range(vid.shape[0]):
            spix_ids = th.unique(spix[t][th.where(anno[t]>0)])
            mask = th.zeros_like(spix[[t]])
            args = th.where(th.isin(spix[[t]],spix_ids))
            mask[args] = 1
            mask = mask[:,None]
            mask_rgb = th.tensor(color, device=vid.device).view(1, 3, 1, 1) * mask
            vid[[t]] = th.where(mask_rgb.any(1,keepdim=True) > 0,(1-alpha)*vid[[t]]+alpha*mask_rgb,vid[[t]])

    return vid

def read_spix(root,frames):
    spix = []
    for frame in frames:
        fn = root / ("%05d.csv"%frame)
        spix_f = np.asarray(pd.read_csv(fn,header=None))
        spix.append(th.from_numpy(spix_f))
    spix = th.stack(spix)
    return spix

def add_text(img,name):
    img = th.nn.functional.pad(img,(0,0,45,0),value=1.0)
    text_img = get_text_img_from_name(name,img.shape)
    print(img.shape,text_img.shape)
    img[:,:,3:3+41] = text_img
    return img

def get_text_img_from_name(name,ishape):
    fname = Path("./data/text_images")/("%s.png"%name)
    text = tvio.read_image(fname)/255.
    _,_H,_W = text.shape
    _,_,H,W = ishape

    # -- ensure height 41 --
    new_width = int(_W * (41 / _H))
    text = F.resize(text, size=[41, new_width])


    topad = W - text.shape[2]
    left = topad//2
    right = topad - left
    text = th.nn.functional.pad(text,(left,right,0,0),value=1)
    text = text[:3]
    return text

def collect_results(vid,anno,method,dname,vname,pname,frames,color,fill_color):

    # -- manage boundary color --
    if color == "grey":
        color = th.tensor([1.0,1.0,1.0])*0.25
    elif color == "blue":
        color = th.tensor([0.0,0.0,1.0])
    elif color == "red":
        color = th.tensor([1.0,0.0,0.0])*1.0
    elif color == "purple":
        color = th.tensor([1.0,0.0,1.0])*0.8

    # -- get marked sequence --
    root = Path("result/%s/%s/%s/"%(dname,method,pname))/vname
    spix = read_spix(root,frames).to(vid.device).int()
    marked = bist_cuda.get_marked_video(vid,spix,color)
    marked = rearrange(marked,'t h w c -> t c h w')

    # -- manage fill color --
    if color == "grey":
        color = th.tensor([1.0,1.0,1.0])*0.5
    elif color == "blue":
        color = th.tensor([0.0,0.0,1.0])
    elif color == "red":
        color = th.tensor([1.0,0.0,0.0])
    elif color == "purple":
        color = th.tensor([1.0,0.0,1.0])*0.8

    # -- color by superpixel label --
    alpha = 0.5
    marked = color_spix(marked,spix,anno,color,alpha,method)

    return marked


def collect_and_save(dname,vname,frames,save_root):

    # -- read video --
    offs = 0 if dname == "davis" else 1
    vid = read_video(dname,vname)
    anno = read_seg(dname,vname)
    vid = th.stack([th.from_numpy(vid[t-offs]) for t in frames])
    anno = th.stack([th.from_numpy(anno[t-offs]) for t in frames])
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
    pnames = ["param209","param210","param211"]
    vids = []
    cols = ["blue","purple","red"]
    # cols = ["red","blue"]
    for pname,col in zip(pnames,cols):
        vids.append(collect_results(vid,anno,method,dname,vname,pname,frames,col,col))

    # -- bass --
    method = "bass"
    pname = "param0"
    col = "grey"
    vids = [collect_results(vid,anno,method,dname,vname,pname,frames,col,col)] +  vids

    # -- add text --
    titles = ["bass_ref","bist_relabel_5p2", "bist_relabel_1p2", "bist_relabel_1p3"]
    _vids = []
    for t in range(len(vids)):
        _vids.append(add_text(vids[t],titles[t]))
    vids = _vids

    # -- ensure save root --
    # save_root = save_root / ("fmt%d"%fmt)
    save_root = save_root / "fmt1"
    if not save_root.exists(): save_root.mkdir(parents=True)

    # -- save grid --
    vids = th.stack(vids)
    # vids = vids[...,-450:,-450:]
    # print(vids.shape)
    seq = []
    def prepare_for_seq(img):
        img = np.clip(255.*img.cpu().numpy(),0.,255.).astype(np.uint8)
        img = rearrange(img,'c h w -> h w c')
        return img

    for ix,t in enumerate(frames):

        # -- save name --
        save_name = save_root/ ("%05d.png"%t)

        #
        # -- format 1 [1 + 2x2]--
        #

        # -- 2x2 grid --
        grid_ix = tv_utils.make_grid(vids[:,ix],nrow=2)
        _, grid_h, grid_w = grid_ix.shape

        # # -- add padding to bass --
        # bass = vids[-1,ix]
        # _, H, W = bass.shape
        # top_pad = (grid_h - H) // 2
        # bottom_pad = grid_h - (H + top_pad)
        # side_pad = 10  # Space between grid and img5
        # padded_bass = th.nn.functional.pad(bass, (0, side_pad, top_pad, bottom_pad), value=0)

        # # -- combine 2x2 and extra image --
        # final_image = th.cat([padded_bass, grid_ix], dim=2)

        # -- save --
        seq.append(prepare_for_seq(grid_ix))
        tv_utils.save_image(grid_ix,save_name)


    # -- make the gif --
    print(f"Saved to {str(save_root)}")
    if len(seq) != 0:
        fname = save_root / ("%s.mp4"%vname)
        print("Saving movie: ",fname)
        fps = 2.5
        imageio.mimsave(fname, seq, fps=fps, format='mp4', codec='libx264')


def create_poster_pair(dname,vname,frames,save_root):

    # -- read video --
    offs = 0 if dname == "davis" else 1
    vid = read_video(dname,vname)
    anno = read_seg(dname,vname)
    vid = th.stack([th.from_numpy(vid[t-offs]) for t in frames])
    anno = th.stack([th.from_numpy(anno[t-offs]) for t in frames])
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
    pnames = ["param209","param211"]
    vids = []
    cols = ["blue","red"]
    # cols = ["red","blue"]
    for pname,col in zip(pnames,cols):
        vids.append(collect_results(vid,anno,method,dname,vname,pname,frames,col,col))

    # -- bass --
    method = "bass"
    pname = "param0"
    col = "grey"
    # vids = [collect_results(vid,anno,method,dname,vname,pname,frames,col,col)] +  vids

    # -- add text --
    # titles = ["bass_ref","bist_relabel_5p2", "bist_relabel_1p2", "bist_relabel_1p3"]
    titles = ["bist_relabel_5p2", "bist_relabel_1p3"]
    _vids = []
    for t in range(len(vids)):
        _vids.append(add_text(vids[t],titles[t]))
    vids = _vids

    # -- ensure save root --
    # save_root = save_root / ("fmt%d"%fmt)
    save_root = save_root / "fmt2"
    if not save_root.exists(): save_root.mkdir(parents=True)

    # -- save grid --
    vids = th.stack(vids)
    # vids = vids[...,-450:,-450:]
    # print(vids.shape)
    seq = []
    def prepare_for_seq(img):
        img = np.clip(255.*img.cpu().numpy(),0.,255.).astype(np.uint8)
        img = rearrange(img,'c h w -> h w c')
        return img

    for ix,t in enumerate(frames):

        # -- save name --
        save_name = save_root/ ("%05d.png"%t)

        #
        # -- format 1 [1 + 2x2]--
        #

        # -- 2x2 grid --
        grid_ix = tv_utils.make_grid(vids[:,ix],nrow=2)
        _, grid_h, grid_w = grid_ix.shape

        # # -- add padding to bass --
        # bass = vids[-1,ix]
        # _, H, W = bass.shape
        # top_pad = (grid_h - H) // 2
        # bottom_pad = grid_h - (H + top_pad)
        # side_pad = 10  # Space between grid and img5
        # padded_bass = th.nn.functional.pad(bass, (0, side_pad, top_pad, bottom_pad), value=0)

        # # -- combine 2x2 and extra image --
        # final_image = th.cat([padded_bass, grid_ix], dim=2)

        # -- save --
        seq.append(prepare_for_seq(grid_ix))
        tv_utils.save_image(grid_ix,save_name)



def main():

    # # -- example --
    # dname = "davis"
    # vname = "judo"
    # frames = np.arange(0,30+1)
    # save_root = Path("output/viz_results/relabeling/")/dname/vname
    # collect_and_save(dname,vname,frames,save_root)
    # return

    # -- example --
    dname = "davis"
    vname = "kite-surf"
    # frames = np.arange(0,32)
    frames = np.arange(0,10)
    save_root = Path("output/viz_results/relabeling/")/dname/vname
    # collect_and_save(dname,vname,frames,save_root)
    # collect_and_save(dname,vname,frames,save_root)

    # -- example --
    dname = "davis"
    vname = "soapbox"
    frames = np.arange(35,55+1)
    save_root = Path("output/viz_results/relabeling/")/dname/vname
    create_poster_pair(dname,vname,frames,save_root)
    # collect_and_save(dname,vname,frames,save_root)
    # collect_and_save(dname,vname,frames,save_root)

    # -- example --
    dname = "davis"
    vname = "horsejump-high"
    frames = np.arange(0,15)
    save_root = Path("output/viz_results/relabeling/")/dname/vname
    # collect_and_save(dname,vname,frames,save_root)
    # collect_and_save(dname,vname,frames,save_root)

    # -- example --
    dname = "davis"
    vname = "parkour"
    frames = np.arange(10,20+1)
    save_root = Path("output/viz_results/relabeling/")/dname/vname
    # collect_and_save(dname,vname,frames,save_root)
    # collect_and_save(dname,vname,frames,save_root)

    # -- example --
    dname = "davis"
    vname = "dance-twirl"
    frames = np.arange(0,15+1)
    save_root = Path("output/viz_results/relabeling/")/dname/vname
    # collect_and_save(dname,vname,frames,save_root)
    # collect_and_save(dname,vname,frames,save_root)

    # -- example --
    dname = "davis"
    vname = "blackswan"
    frames = np.arange(0,15+1)
    save_root = Path("output/viz_results/relabeling/")/dname/vname
    # collect_and_save(dname,vname,frames,save_root)
    # collect_and_save(dname,vname,frames,save_root)

    # -- example --
    dname = "segtrackerv2"
    vname = "frog_2"
    frames = np.arange(1,40+1)
    save_root = Path("output/viz_results/relabeling/")/dname/vname
    # collect_and_save(dname,vname,frames,save_root)

if __name__ == "__main__":
    main()
