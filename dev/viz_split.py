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

def gaussian_blur(image, kernel_size=15, sigma=5):
    """
    Applies a Gaussian blur to an image using a 2D convolution.

    Parameters:
    - image (th.Tensor): Image tensor with shape (C, H, W).
    - kernel_size (int): Size of the Gaussian kernel.
    - sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
    - th.Tensor: Blurred image tensor.
    """
    # Create a 1D Gaussian kernel
    x = th.arange(kernel_size) - kernel_size // 2
    gauss = th.exp(-0.5 * (x / sigma) ** 2)
    gauss /= gauss.sum()

    # Create 2D kernel by outer product
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)  # Shape: (1,1,K,K)
    kernel = kernel.to(image.device)

    # Apply the kernel to the alpha channel using depthwise convolution
    F  = th.nn.functional
    padding = kernel_size // 2
    blurred = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
    return blurred.squeeze()

def read_spix(root,frames):
    spix = []
    for frame in frames:
        fn = root / ("%05d.csv"%frame)
        spix_f = np.asarray(pd.read_csv(fn,header=None))
        spix.append(th.from_numpy(spix_f))
    spix = th.stack(spix)
    return spix

def inset_crop(img,crop_points,color,loc):
    if img.shape[1] == 3:
        alpha = th.ones_like(img[:,:1])
        img = th.cat([img,alpha],1)
    hs,he,ws,we = crop_points
    crop = img[:,:,hs:he,ws:we].clone()
    T,C,_H,_W = crop.shape

    # -- resize --
    tgt_H = 250
    new_width = int(_W * (tgt_H / _H))
    crop = F.resize(crop, size=[tgt_H, new_width])
    T,C,_H,_W = crop.shape
    print(img.shape)

    # -- get slices for locs --
    T,C,H,W = img.shape
    alpha = 0.5
    off = 8
    if loc == "right":
        hslice0 = slice(H-_H-off,H-off)
        wslice0 = slice(W-_W-off,W-off)
        hslice1 = slice(H-_H-off,H)
        wslice1 = slice(W-_W-off,W)
    elif loc == "left":
        hslice0 = slice(H-_H-off,H-off)
        wslice0 = slice(off,_W+off)
        hslice1 = slice(H-_H-off,H)
        wslice1 = slice(off,_W+2*off)
    elif loc == "center":
        WS = W//2-_W//2
        hslice0 = slice(H-_H-off,-off)
        wslice0 = slice(WS,WS+_W)
        hslice1 = slice(H-_H-off,H)
        wslice1 = slice(WS,WS+_W+off)
    else:
        raise ValueError(".")

    # -- add shadow --
    dark = th.zeros((1,1,_H+off,_W+off)).cuda()
    # dark[:,:,-_H-off:-off,-_W-off:-off] = 1.0
    dark[:,:,hslice0,wslice0] = 1.0
    dark = gaussian_blur(dark[0,0],kernel_size=7,sigma=5.)
    # reg = img[:,:,-_H-off:,-_W-off:]
    reg = img[:,:,hslice1,wslice1]

    # dark = 1-dark
    _dark = th.where(dark<1e-1,reg,dark)
    pix = th.where(dark<1e-1,reg,(1-dark)*0.5)
    # img[:,:,-_H-off:,-_W-off:] = alpha * reg + (1-alpha)*dark

    # -- ... --
    # img[:,:,-_H-off:,-_W-off:] = _dark
    # img[:,:3,-_H-off:,-_W-off:] = pix[:,:3]
    img[:,:,hslice1,wslice1] = _dark
    img[:,:3,hslice1,wslice1] = pix[:,:3]
    img[:,3] = 1.0

    # -- fill crop --
    T,C,_H,_W = crop.shape
    CW = 2
    crop[:,:,:CW,:] = 0.
    crop[:,:,-CW:,:] = 0.
    crop[:,:,:,:CW] = 0.
    crop[:,:,:,-CW:] = 0.
    # img[:,:,-_H-off:-off,-_W-off:-off] = crop


    # -- show cropped region --
    box = th.tensor([ws,hs,we,he]).cuda()[None,:]
    fill_color = color_tensor_from_string(color).cuda()
    color = fill_color.cpu().numpy().tolist()
    color = tuple([int(255.*c) for c in color])
    print(color)
    alpha = 0.5
    fill_color = fill_color.view(3,1,1) * th.ones_like(img[0,:3,hs:he,ws:we])
    for t in range(len(img)):
        img[t,:3] = tv_utils.draw_bounding_boxes(img[t,:3],box,colors=color,width=3)
        img[t,:3,hs:he,ws:we] = alpha * img[t,:3,hs:he,ws:we] + (1-alpha) * fill_color

    # -- replace with crop --
    img[:,:,hslice0,wslice0] = crop

    return img

def add_text(img,name,color):
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

def color_tensor_from_string(color):
    if color == "grey":
        color = th.tensor([1.0,1.0,1.0])*0.25
    elif color == "blue":
        color = th.tensor([0.0,0.0,1.0])
    elif color == "red":
        color = th.tensor([1.0,0.0,0.0])*1.0
    elif color == "purple":
        color = th.tensor([1.0,0.0,1.0])*0.8
    return color

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
    # alpha = 0.5
    # marked = color_spix(marked,spix,anno,color,alpha,method)

    return marked


def collect_and_save(dname,vname,frames,crop,fmt,save_root):

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
    pnames = ["param402","param401","param400"]
    vids = []
    cols = ["red","purple","blue"]
    # cols = ["red","blue"]
    for pname,col in zip(pnames,cols):
        vids.append(collect_results(vid,anno,method,dname,vname,pname,frames,col,col))

    # -- bass --
    method = "bass"
    pname = "param0"
    col = "grey"
    vids = [collect_results(vid,anno,method,dname,vname,pname,frames,col,col)] +  vids

    # -- add text --
    titles = ["bass_ref","bist_split_0", "bist_split_4", "bist_split_8"]
    cols = ["grey","red","purple","blue"]
    _vids = []
    for t in range(len(vids)):
        vid_t = add_text(vids[t],titles[t],cols[t])
        vid_t = inset_crop(vid_t,crop,cols[t],"right")
        _vids.append(vid_t)
    vids = _vids

    # -- ensure save root --
    save_root = save_root
    # save_root = save_root / "fmt1"
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
        save_name = save_root/ ("%s_%05d_%d.png"%(vname,t,fmt))

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
    # print(f"Saved to {str(save_root)}")
    # if len(seq) != 0:
    #     fname = save_root / ("%s.mp4"%vname)
    #     print("Saving movie: ",fname)
    #     fps = 2.5
    #     imageio.mimsave(fname, seq, fps=fps, format='mp4', codec='libx264')


def create_poster_pair(dname,vname,frames,crops,fmt,save_root):

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
    pnames = ["param402","param401"]
    vids = []
    cols = ["red","blue"]
    for pname,col in zip(pnames,cols):
        vids.append(collect_results(vid,anno,method,dname,vname,pname,frames,col,col))

    # -- bass --
    method = "bass"
    pname = "param0"
    col = "grey"
    # vids = [collect_results(vid,anno,method,dname,vname,pname,frames,col,col)] +  vids

    # -- add text --
    # titles = ["bass_ref","bist_split_0", "bist_split_4", "bist_split_8"]
    # cols = ["grey","blue","purple","red"]
    titles = ["bist_split_0", "bist_split_4"]
    cols = ["red","blue"]
    _vids = []
    locs = ["left","center","right"]
    assert len(locs) >= len(crops)
    for t in range(len(vids)):
        vid_t = vids[t].clone()
        vid_t = add_text(vids[t],titles[t],cols[t])
        for ic,crop in enumerate(crops):
            vid_t = inset_crop(vid_t,crop,cols[t],locs[ic])
        _vids.append(vid_t)
    vids = _vids

    # -- ensure save root --
    save_root = save_root / "poster"
    # save_root = save_root / "fmt1"
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
        save_name = save_root/ ("%s_%05d_%d.png"%(vname,t,fmt))

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
    # print(f"Saved to {str(save_root)}")
    # if len(seq) != 0:
    #     fname = save_root / ("%s.mp4"%vname)
    #     print("Saving movie: ",fname)
    #     fps = 2.5
    #     imageio.mimsave(fname, seq, fps=fps, format='mp4', codec='libx264')


def main():

    # # -- example --
    # dname = "davis"
    # vname = "judo"
    # frames = np.arange(0,30+1)
    # save_root = Path("output/viz_results/split/")/dname/vname
    # collect_and_save(dname,vname,frames,fmt,save_root)
    # return

    # -- example --
    dname = "davis"
    # vname = "kite-surf"
    vname = "bmx-trees"
    # frames = np.arange(0,32)
    frames = np.arange(25,26)
    # frames = np.arange(0,10)
    save_root = Path("output/viz_results/split/")/dname/vname
    # crop = [200,200+100,230+50+50,230+100+50+50]
    # collect_and_save(dname,vname,frames,crop,0,save_root)
    # crop = [140,140+100,230,230+100]
    # collect_and_save(dname,vname,frames,crop,1,save_root)
    # crop = [410,410+100,50,50+100]
    # collect_and_save(dname,vname,frames,crop,2,save_root)
    # crop = [100,100+100,400+50+50+50+50+100,400+100+50+50+50+50+100]
    # collect_and_save(dname,vname,frames,crop,3,save_root)

    crops = [
        [120,120+100,230-50,230+100-50],
        [100,100+100,230+200,230+100+200],
        [120,120+100,230+320+80,230+100+320+80]
    ]
    frames = np.arange(35,36)
    create_poster_pair(dname,vname,frames,crops,0,save_root)


    # -- example --
    dname = "davis"
    vname = "soapbox"
    crop = [160,160+100,330,330+100]
    frames = np.arange(54,55)
    save_root = Path("output/viz_results/split/")/dname/vname
    # collect_and_save(dname,vname,frames,crop,save_root)

    # -- example --
    dname = "davis"
    vname = "horsejump-high"
    frames = np.arange(10,11)
    save_root = Path("output/viz_results/split/")/dname/vname
    # crop = [100,100+100,170+50+50,170+100+50+50]
    # collect_and_save(dname,vname,frames,crop,0,save_root)
    # crop = [140,140+100,230,230+100]
    # collect_and_save(dname,vname,frames,crop,1,save_root)
    # crop = [410,410+100,50,50+100]
    # collect_and_save(dname,vname,frames,crop,2,save_root)
    # crop = [100,100+100,400+50+50+50+50+100,400+100+50+50+50+50+100]
    # collect_and_save(dname,vname,frames,crop,3,save_root)

    # -- example --
    dname = "davis"
    vname = "breakdance"
    frames = np.arange(20,21)
    save_root = Path("output/viz_results/split/")/dname/vname
    # crop = [100,100+100,170+50+50,170+100+50+50]
    # crop = [390,390+100,250+50+100,250+50+100+100]
    # collect_and_save(dname,vname,frames,crop,0,save_root)
    # crop = [75,75+100,120+50+50,120+100+50+50]
    # collect_and_save(dname,vname,frames,crop,1,save_root)
    # crop = [410,410+100,190,190+100]
    # collect_and_save(dname,vname,frames,crop,2,save_root)


    # -- example --
    dname = "davis"
    vname = "dance-twirl"
    frames = np.arange(10,11)
    save_root = Path("output/viz_results/split/")/dname/vname
    # crop = [45,45+100,230+50+50+10,230+100+50+50+10]
    # collect_and_save(dname,vname,frames,crop,0,save_root)
    # crop = [140,140+100,230,230+100]
    # collect_and_save(dname,vname,frames,crop,1,save_root)
    # crop = [410,410+100,50,50+100]
    # collect_and_save(dname,vname,frames,crop,2,save_root)
    # crop = [100,100+100,400+50+50+50+50+100,400+100+50+50+50+50+100]
    # collect_and_save(dname,vname,frames,crop,3,save_root)

    # -- example --
    dname = "davis"
    vname = "parkour"
    frames = np.arange(20,21)
    save_root = Path("output/viz_results/split/")/dname/vname
    # crop = [60,60+100,250+100+120,250+100+100+120]
    # collect_and_save(dname,vname,frames,crop,0,save_root)
    # crop = [160,160+100,250+100+120,250+100+100+120]
    # collect_and_save(dname,vname,frames,crop,1,save_root)

    # -- example --
    dname = "davis"
    vname = "blackswan"
    frames = np.arange(10,11)
    save_root = Path("output/viz_results/split/")/dname/vname
    # collect_and_save(dname,vname,frames,crop,save_root)

    # -- example --
    dname = "segtrackerv2"
    vname = "frog_2"
    frames = np.arange(10,11)
    save_root = Path("output/viz_results/split/")/dname/vname
    # collect_and_save(dname,vname,frames,crop,save_root)

if __name__ == "__main__":
    main()
