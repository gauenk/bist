"""

   Compute a pixel's similarity to a superpixel...

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
import torch.nn.functional as F

import glob
from pathlib import Path
# from run_eval import read_video,read_seg,get_video_names
from st_spix.utils import rgb2lab
from st_spix.spix_utils.updated_io import read_video,read_seg
from st_spix.spix_utils.evaluate import computeSummary,scoreSpixPoolingQualityByFrame,count_spix,read_spix
from st_spix.spix_utils.evaluate import get_video_names

import bist
import st_spix
import prop_cuda
import bist_cuda
# import bist2_cuda
from st_spix.sp_pooling import sp_pooling

import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def draw_points(canvas, locs, square_size, colors, shape='circle'):

    # Generate colors from colormap
    # num_points = len(locs)
    # colors = cm.get_cmap(cmap_name, num_points)(np.linspace(0, 1, num_points))[:, :3]  # RGB colors
    C,H,W = canvas.shape

    # Convert to tensors
    # locs = th.tensor(locs)  # Shape: (S, 2)
    colors = th.tensor(colors).float().cuda()  # Shape: (S, 3)

    # Define square radius
    radius = square_size
    r = square_size // 2

    if shape == 'circle':


        # Create a grid for the entire canvas
        h_grid = th.arange(H).view(H, 1).expand(H, W).cuda()  # (H, W)
        w_grid = th.arange(W).view(1, W).expand(H, W).cuda()  # (H, W)

        # Apply colors to canvas
        for i in range(len(locs)):
            # Compute squared distance from center
            dist_sq = (h_grid - locs[i, 1]) ** 2 + (w_grid - locs[i, 0]) ** 2
            mask = dist_sq <= radius ** 2  # Boolean mask for the circle
            canvas[:3,mask] = colors[i].view(3,1)
            canvas[3,mask] = 1.0

    elif shape == 'square':

        # Create grid for the squares
        h_range = th.arange(-r, r + 1)
        w_range = th.arange(-r, r + 1)
        dh, dw = th.meshgrid(h_range, w_range, indexing='ij')  # Shape: (2, square_size, square_size)
        dh, dw = dh.cuda(), dw.cuda()

        # Compute pixel locations for each square
        square_w = locs[:, 0, None, None] + dw
        square_h = locs[:, 1, None, None] + dh  # Shape: (S, square_size, square_size)

        # Clamp values to stay within bounds
        square_h = square_h.clamp(0, H - 1).long()
        square_w = square_w.clamp(0, W - 1).long()

        for i in range(len(locs)):
            canvas[:3,square_h[i], square_w[i]] = colors[i].view(3,1,1)
            canvas[3,square_h[i], square_w[i]] = 1.0
    else:
        raise ValueError("")

    return canvas


def warp_image(image, flow):
    """
    Warps an image using the given optical flow offsets.

    Args:
        image: Tensor of shape (B, C, H, W), input image.
        flow: Tensor of shape (B, H, W, 2), containing flow offsets in (dx, dy) format.

    Returns:
        Warped image of shape (B, C, H, W).
    """
    B, C, H, W = image.shape

    # Generate a normalized grid for indexing
    # y, x = th.meshgrid(th.arange(H), th.arange(W), indexing='ij')
    # grid = th.stack((x, y), dim=-1).float().to(flow.device)
    # grid = grid.unsqueeze(0).expand(B, -1, -1, -1)*1.0
    grid = flow.clone()

    # Normalize grid to [-1, 1] range for grid_sample
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # Normalize x
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # Normalize y
    # grid = grid.flip(-1)  # PyTh expects (y, x) instead of (x, y)
    print(grid[...,0].min(),grid[...,0].max())
    print(grid[...,1].min(),grid[...,1].max())

    # Apply grid sampling
    warped_image = F.grid_sample(image, grid, mode='bilinear',
                                 padding_mode='border',
                                 align_corners=True)
    print(th.where(warped_image[0].sum(0)>0))
    return warped_image

def read_image(fname):
    return tvio.read_image(fname)/255.
def csv_to_th(fname):
    return th.from_numpy(pd.read_csv(str(fname),header=None).to_numpy())
def save_image(img,fname):
    tv_utils.save_image(img,fname)
def crop_image(img,crop):
    hs,he,ws,we = crop
    return img[...,hs:he,ws:we]
def read_spix(root,frames):
    spix = []
    for frame in frames:
        fn = root / ("%05d.csv"%frame)
        spix_f = np.asarray(pd.read_csv(fn,header=None))
        spix.append(th.from_numpy(spix_f))
    spix = th.stack(spix)
    return spix
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

def sp_pooling_th(image, mask):
    """
    Compute the average value of the image for each unique mask value and return both:
    - "down": shape (T, C, S), where S is the max mask value + 1
    - "pool": shape (T, C, H, W), where each pixel is replaced by its mask's average value
    """
    image = image.float()
    T, C, H, W = image.shape
    mask = mask.long()  # Ensure mask is long type for indexing

    S = mask.max().item() + 1  # The number of possible mask values
    mask_flat = mask.view(T, H * W)  # Flatten spatially
    image_flat = image.contiguous().view(T, C, H * W) #Flatten spatially

    # Compute sum per mask value
    down = th.zeros((T, C, S), device=image.device, dtype=th.float32)
    count = th.zeros((T, 1, S), device=image.device, dtype=th.float32)
    # print(down.device,mask_flat.device)

    down.scatter_reduce_(2, mask_flat.unsqueeze(1).expand(-1, C, -1), image_flat, reduce="sum")
    count.scatter_reduce_(2, mask_flat.unsqueeze(1), th.ones_like(image_flat), reduce="sum")

    # Avoid division by zero
    count[count == 0] = 1
    down /= count

    # Map back to full resolution
    pool = down.gather(2, mask_flat.unsqueeze(1).expand(-1, C, -1)).view(T, C, H, W)

    # -- final reshape --
    down = rearrange(down,'t c s -> t s c')
    print("out: ",down.shape)
    pool = rearrange(pool,'t c h w -> t h w c')
    return pool, down

def upsample_th(down,spix):
    T,H,W = spix.shape
    # T, C, H, W = image.shape
    C = down.shape[2]
    spix = spix.long()  # Ensure spix is long type for indexing
    S = spix.max().item() + 1  # The number of possible spix values
    spix_flat = spix.view(T, H * W)  # Flatten spatially
    spix_flat = spix_flat.unsqueeze(1).expand(-1, C, -1)
    down = rearrange(down,'t s c -> t c s')
    pool = down.gather(2, spix_flat).view(T, C, H, W)
    pool = rearrange(pool,'t c h w -> t h w c')
    return pool

def get_sims(vid,spix,scale=1.):
    T,F,H,W = vid.shape
    use_video_pooling = False

    # -- ... --
    # def relabel_mask_th(mask):
    #     unique_labels, inverse_indices = th.unique(mask, return_inverse=True)
    #     # relabeled_mask = inverse_indices + 1  # Shift to start from 1 instead of 0
    #     return inverse_indices.reshape(mask.shape)

    # -- shrink to reduce dimensions --
    # print(spix.shape)
    _unique,spix = th.unique(spix, return_inverse=True)
    ispix = spix
    # print(spix.shape)

    # -- downsample --
    # _vid = vid
    vid = rearrange(vid,'t f h w -> t h w f')
    means,down = sp_pooling(vid,spix)
    # _means,_down = sp_pooling_th(_vid,spix)
    # print(means.shape,_means.shape)
    # print(down.shape,_down.shape)
    # print(th.sum((means-_means)**2),th.sum((down-_down)**2))

    # -- only keep the "down" from this video subsequence --
    spids = th.arange(down.shape[1]).to(vid.device)
    # print("pre: ",spids.shape,down.shape)
    vmask = (spix.unsqueeze(-1) == spids.view(1, 1, 1, 1, -1)).any((0,1,2,3))
    spids = spids[vmask]
    down = down[:,vmask]
    # print(spids.shape,down.shape)

    # -- pwd --
    vid = rearrange(vid,'t h w f -> t (h w) f')
    pwd = th.cdist(vid,down)**2 # sum-of-squared differences
    pwd = rearrange(pwd,'t (h w) s -> t s h w',t=T,h=H)

    # -- mask invalid ["empty" spix in down have "0" value] --
    mask = ~(spix.unsqueeze(-1) == spids.view(1, 1, 1, 1, -1)).any((1, 2, 3))
    pwd[mask] = th.inf

    # -- normalize --
    # sims = th.softmax(-scale*pwd,1)
    sims = th.exp(-scale*pwd)
    sims = sims / sims.max(1,keepdim=True).values

    return sims,_unique,ispix,down

def get_alphas(sim,inv):
    # uniq,inv = th.unique(spix,return_inverse=True)
    mask = sim[inv]
    return mask

def mask2col(mask):

    # -- pastel red --
    tmp  = mask.clone()
    s = 1-tmp[0]
    col0 = mask.clone()
    # col0[0] = 1.0*s
    # col0[1] = (50/255.)*s
    # col0[2] = (50/255.)*s
    # col0[0] = 1.0*s
    # col0[1] = 0.
    # col0[2] = 0.

    # -- pastel blue --
    s = tmp[0]
    col1 = mask.clone()
    col1[0] = (50/255.)*s
    col1[1] = (100/255.)*s
    col1[2] = 1.0*s
    # col1[0] = 0.
    # col1[1] = 0.
    # col1[2] = (1.0)*s

    # -- mix --
    mask = (col0 + col1)/2.0
    mask = mask/mask.max()
    return mask

def animate_sims(img,dname,vname,pname,method,frame,crop,ix):



    # -- save config --
    save_root = Path("output/plots/viz_spix_attn/%s/anim/%d"%(vname,ix))
    if not save_root.exists(): save_root.mkdir(parents=True)


    #
    # -- convert to red/blue --
    #
    #     Pastel Red: RGB(255, 153, 153) → Hex: #FF9999
    #     Pastel Blue: RGB(153, 204, 255) → Hex: #99CCFF

    # GB(255, 102, 102) → Hex: #FF6666
    # Stronger Pastel Blue: RGB(102, 153, 255) →

    # ing Color): RGB(102, 204, 255) → Hex: #66CCFF
    # Soft Pastel Green (Ending Color): RGB(153, 255, 153) → Hex: #99FF99

    # -- pastel red --
    tmp  = img.clone()
    s = 1-tmp[0]
    col0 = img.clone()
    # col0[0] = 1.0*s
    # col0[1] = (50/255.)*s
    # col0[2] = (50/255.)*s
    # col0[0] = 1.0*s
    # col0[1] = 0.
    # col0[2] = 0.

    # -- pastel blue --
    s = tmp[0]
    col1 = img.clone()
    col1[0] = (50/255.)*s
    col1[1] = (100/255.)*s
    col1[2] = 1.0*s
    # col1[0] = 0.
    # col1[1] = 0.
    # col1[2] = (1.0)*s

    # -- mix --
    img = (col0 + col1)/2.0
    img = img/img.max()

    # -- read data --
    root = Path("result/%s/%s/%s/"%(dname,method,pname))/vname
    spix = read_spix(root,[frame]).int()
    spix = crop_image(spix,crop)
    in_spix = spix.clone()
    pix_pooled,pix_down = sp_pooling_th(img[None,:],spix)

    # # -- mark image --
    # img[:,:3] = get_marked_video(img_i[:,:3],spix,color)

    #
    # -- init canvas --
    #

    print("img.shape: ",img.shape)
    img = img.contiguous().float()
    C,_H,_W = img.shape
    padH = 100
    pad = 300
    canvas = th.zeros((4,_H+padH,_W+2*pad),dtype=th.float32,device="cuda")
    # canvas[:3,:_H,pad:_W+pad] = img
    # canvas[3,:_H,pad:_W+pad] = 1.
    # print(_H,_W,img.shape)
    canvas[:3,:_H,:_W] = img
    canvas[3,:_H,:_W] = 1.
    # canvas[:3,padH:_H+padH,:_W] = img
    # canvas[3,padH:_H+padH,:_W] = 1.

    # Create coordinate grids
    # _,H,W = canvas.shape
    # y = th.linspace(0, 1, H).view(H, 1).cuda()
    # x = th.linspace(0, 1, W).view(1, W).cuda()
    # gradient = (y + x) / 2
    # canvas[:3] = (y + x).view(1,H,W)/2.
    # canvas[3] = 1.0
    C,H,W = canvas.shape
    print("canvas.shape: ",canvas.shape)

    # -- expand spix for canvas --
    spix = spix + 1
    # spix = th.nn.functional.pad(spix,(pad,pad,0,padH),value=0).cuda()
    spix = th.nn.functional.pad(spix,(0,2*pad,0,padH),value=0).cuda()
    pix_down = th.cat([th.zeros_like(pix_down[:,:1]),pix_down],1)
    _unique,ispix = th.unique(spix, return_inverse=True)
    sizes = th.bincount(spix.ravel())
    SM = spix.max().item()+1
    S = len(_unique)

    print(canvas.shape,spix.shape)
    thresh = 0.1
    uniq = th.unique(spix[0][th.where(canvas[:3].sum(0)>thresh)])
    uniqz = th.unique(spix[0][th.where(canvas[:3].sum(0)<=thresh)])
    sizes[uniqz] = len(uniq)+2
    sizes[uniq] = th.arange(len(uniq)).cuda()+1
    S = len(uniq)
    # uniq = _unique[nz_spix]
    print("spix.shape: ",spix.shape)

    # -- read image --
    read_root = Path("output/plots/viz_spix_attn/%s"%vname)
    fname = read_root/("mask%d.png"%ix)
    # canvas = read_image(fname)[:3]

    # -- init target --
    y_tgt = 1.*th.zeros(SM).cuda()
    x_tgt = 1.*th.zeros(SM).cuda()
    y_tgt[...] = H-padH//2
    x_tgt[uniq] = (th.arange(S).cuda()+1.0)/(S*1.0)*(W-1-50)+25
    tgt = th.stack([x_tgt,y_tgt],-1).unsqueeze(0).cuda()

    # -- get locations as means --
    hi, wi = th.meshgrid(th.arange(H), th.arange(W), indexing='ij')
    coord_image = th.stack((wi, hi), dim=0).float()  # Shape: (2, H, W)
    coord_image = coord_image.unsqueeze(0).expand(1, -1, -1, -1).cuda()
    mean_loc,down_loc = sp_pooling_th(coord_image,spix)
    # coord_image = rearrange(coord_image,'1 f h w -> h w f')
    # upsample_th(down,spix)
    # print(down_loc.shape,S,spix.max()+1)
    # exit()

    # -- point image for arrows --
    point = th.zeros_like(canvas)

    # coord_image = down_loc[0][ispix[0]]
    # coord_image = mean_loc[0]
    # print(coord_image[:5,:5])
    # print(mean_loc.shape)
    # print(coord_image.shape)
    # print(mean_loc[0][:5,:5])
    # exit()

    # -- ... --
    # print("itgt.shape: ",itgt.shape)
    # print(th.unique(itgt[:,:,0].ravel()))
    # print(th.unique(itgt[:,:,1].ravel()))
    # print(th.unique(coord_image[:,:,0].ravel()))
    # print(th.unique(coord_image[:,:,1].ravel()))
    # print("c: ",coord_image.shape)
    # print("itgt: ",itgt.shape)
    # exit()

    # -- init endpoint --
    # sizes[th.unique(spix[th.where(canvas.sum(1)==0)])] = 0
    print("canvas.shape: ",canvas.shape)
    niters = 10
    niters_full = 1000
    niters_ratio = niters_full / niters
    save_index = 0
    # niters = 5
    for i in range(niters_full):

        # -- get flow --
        pix_labels = th.arange(H*W).view(1,H,W).cuda()
        # print(pix_labels.max())
        pix_labels = pix_labels.int()
        # print(pix_labels.max())
        spix = spix.cuda().int()
        nspix = spix.max().item()+1
        sizes = sizes.cuda().int()
        alpha = 0.
        alpha = (1.0*i)/(niters_full-1)
        flow = alpha * (tgt.cuda() - down_loc.cuda())
        flow = flow.round().contiguous()
        # print(flow[0,:10])
        # print(down_loc[0,:10])

        # -- shape --
        # print("shapes:")
        # print(pix_labels.shape)
        # print(spix.shape,spix.max()+1)
        # print(sizes.shape)
        # print(flow.shape)
        # flow.shape = (B,nspix,2)
        # print(len(th.unique(ishifted)))
        # print(len(th.unique(pix_labels)))
        # exit()
        # print("ishifted.shape: ",ishifted.shape)

        # -- arrow --
        point_size = 1
        locs = (down_loc + flow)[0][uniq]
        colors = cm.get_cmap('spring', S)(np.linspace(0, 0.5, S))[:, :3]  # RGB colors
        colors = th.tensor(colors).float().cuda()  # Shape: (S, 3)
        sq = draw_points(point, locs, point_size, colors)
        point = th.maximum(point,sq).clone()

        # if (i % niters_ratio == 0) or (i == (niters_full-1)):
        if (i % niters_ratio == 0) or (i == (niters_full-1)):

            # -- shift labels --
            shifted_spix = bist_cuda.shift_labels(spix,spix,flow,sizes,nspix)
            ishifted = bist_cuda.shift_labels(pix_labels,spix,flow,sizes,nspix)
            ishifted[th.isin(shifted_spix,uniqz)] = -1
            shifted_spix = shifted_spix.int()
            shifted_spix[th.isin(shifted_spix,uniqz)] = -1

            # -- index image --
            canvas_ix = canvas[None,:].clone().cuda()
            shifted = bist.viz.shift_tensor(canvas_ix,ishifted[0]).cuda()
            # print("shifted.shape: ",shifted.shape,img.shape)
            img = img.cuda()

            # -- mark shifted image --
            color = th.tensor([0.0,0,1.0])*0.7 # boundary color
            shifted[:,:3] = get_marked_video(shifted[:,:3],shifted_spix,color)

            # -- [1st time] fill original image in background --
            _shifted = shifted.clone().cuda()
            _img = img.clone().cuda()

            _spix = in_spix.clone().cuda()
            _spix[th.isin(_spix+1,uniqz)] = -1
            _img[:3] = get_marked_video(_img[None,:3],_spix,color)[0]

            _spix = in_spix.clone().cuda()
            _spix[th.isin(_spix+1,uniq)] = -1
            color = th.tensor([0.0,0,1.0])*0.25 # boundary color
            _img[:3] = get_marked_video(_img[None,:3],_spix,color)[0]
            # save_image(_img,save_root/("example.png"))
            # exit()

            # nzargs = th.where((_shifted[0,:3,:_H,pad:_W+pad].sum(0)==0))
            # _tofill = _shifted[0,:3,:_H,pad:_W+pad]
            nzargs = th.where((_shifted[0,:3,:_H,:_W].sum(0)))
            _tofill = _shifted[0,:3,:_H,:_W]
            _tofill[:, nzargs[0], nzargs[1]] = _img[:, nzargs[0], nzargs[1]]
            # _shifted[0,3,:_H,pad:_W+pad] = 1.
            _shifted[0,3,:_H,:_W] = 1.

            # -- fill points in front of shifted --
            nzargs = th.where((point.sum(0)>0))
            mix0 = point[:,nzargs[0],nzargs[1]]
            mix1 = _shifted[0,:,nzargs[0],nzargs[1]]
            # mix_alpha = 0.35
            mix_alpha = 0.4
            mix = mix_alpha*mix0 + (1-mix_alpha)*mix1
            _shifted[0,:,nzargs[0],nzargs[1]] = mix
            # shifted[0,4] = shifted[0].sum(0)>0

            # -- put shifted spix in front of dotted line --
            nzargs = th.where((shifted[0].sum(0)==0))
            shifted[0,:,nzargs[0],nzargs[1]] = _shifted[0,:,nzargs[0],nzargs[1]]

            # -- fill original image in background --
            shifted = shifted.cuda()
            # img = img.cuda()
            nzargs = th.where((shifted[0,:3,:_H,:_W].sum(0)==0))
            _tofill = shifted[0,:3,:_H,:_W]
            # nzargs = th.where((shifted[0,:3,:_H,pad:_W+pad].sum(0)==0))
            # _tofill = shifted[0,:3,:_H,pad:_W+pad]
            _tofill[:, nzargs[0], nzargs[1]] = _img[:, nzargs[0], nzargs[1]]
            # shifted[0,3,:_H,pad:_W+pad] = 1.
            shifted[0,3,:_H,:_W] = 1.
            # _tofill = shifted[0,:3,:_H,pad:_W+pad]
            # _tofill[:, nzargs[0], nzargs[1]] = img[:, nzargs[0], nzargs[1]]
            # shifted[0,:3,:_H,pad:_W+pad] = img
            # print(th.mean((shifted-img[None,:].cuda())**2))

            # -- save --
            save_image(shifted,save_root/("frame_%d.png"%save_index))
            save_index += 1
            # save_image(canvas,save_root/("frame_%d.png"%save_index))
            # exit()
            # save_image(point,save_root/("frame_%d.png"%save_index))

    #
    # -- just on the last image -- ! --
    #

    shifted = th.nn.functional.pad(shifted,(0,0,0,padH//2),value=0).cuda()
    spix = th.nn.functional.pad(spix,(0,0,0,padH//2),value=0).cuda()
    _,_,H,W = shifted.shape

    hi, wi = th.meshgrid(th.arange(H), th.arange(W), indexing='ij')
    coord_image = th.stack((wi, hi), dim=0).float()  # Shape: (2, H, W)
    coord_image = coord_image.unsqueeze(0).expand(1, -1, -1, -1).cuda()
    mean_loc,down_loc = sp_pooling_th(coord_image,spix)

    # -- target --
    y_tgt = 1.*th.zeros(SM).cuda()
    x_tgt = 1.*th.zeros(SM).cuda()
    y_tgt[...] = H-padH//4
    x_tgt[uniq] = (th.arange(S).cuda()+1)/(S*1.0)*(W-1-100)+50
    tgt = th.stack([x_tgt,y_tgt],-1).unsqueeze(0).cuda()

    pix_down = pix_down.cuda()
    uniq = uniq.cuda()
    init_point = th.zeros_like(shifted[0])
    point = init_point.clone()
    point_size = 10
    sq = draw_points(point, tgt[0][uniq], point_size, pix_down[0][uniq], 'square')
    point = sq
    # point = th.maximum(point,sq).clone()

    # -- fill empty --
    nzargs = th.where((point.sum(0)>0))
    shifted[0][:, nzargs[0], nzargs[1]] = point[:,nzargs[0], nzargs[1]]
    # shifted[0][3,:_H,pad:_W+pad] = 1.
    # shifted[0,:,H-padH:] = point[:,H-padH:]
    save_image(shifted,save_root/("final.png"))


def main():

    # -- config --
    dname = "davis"
    method = "bist"
    pname = "param0"
    vname = "kid-football"
    offset = 0 if dname == "davis" else 1
    root = Path("result/%s/%s/%s/"%(dname,method,pname))/vname
    save_root = Path("output/plots/viz_spix_attn/%s"%vname)
    if not save_root.exists(): save_root.mkdir(parents=True)

    # -- viz info --
    frame = 4
    crop = [105-45,105+200-45,220,220+200]


    # -- read data --
    vid = th.from_numpy(read_video(dname,vname)[[frame-offset]])
    vid = rearrange(vid,'t h w c -> t c h w')
    spix = read_spix(root,[frame]).to(vid.device).int()
    vid = crop_image(vid,crop)
    spix = crop_image(spix,crop)
    save_image(vid,save_root/"ref.png")
    H,W = vid.shape[-2:]


    # -- crop to region --
    sims,uniq,ispix,down = get_sims(vid,spix,scale=200.)

    # -- for each selected pixel --
    K = 10
    # locs = [[75,85],[125,125],[20,20]]
    locs = [[75,85],[125,125],[20,20]]
    # locs = [[25,25]]
    for ix,loc in enumerate(locs):
        if ix in [0,2]: continue
        sim = sims[0,:,loc[0],loc[1]]
        # print(down[:,spix[0,loc[0],loc[1]]])
        print("."*30)

        arg0 = th.where(uniq==775)
        arg1 = th.where(uniq==1016)
        arg2 = th.where(uniq==1023)
        arg3 = th.where(uniq==947)
        print(sim.sum())
        print(sim[arg0])
        print(sim[arg1])
        print(sim[arg2])
        print(sim[arg3])
        print(sim.min())

        # print(down[0,:,0])
        # print(sim)
        alphas = get_alphas(sim,ispix)
        # print(alphas.max(),alphas.min())
        # quant0 = th.quantile(alphas,0.95)
        # alphas = th.where(alphas>quant0,1.0,alphas/quant0)
        # quant1 = th.quantile(alphas,0.10)
        # alphas = th.where(alphas<quant1,0.0,alphas-quant1)
        alphas = (alphas - alphas.min())/(alphas.max()-alphas.min())
        alphas = th.pow(alphas,1/10.)
        # alphas = th.where(alphas>0.45,1.0,alphas)
        # print(alphas.max(),alphas.min(),alphas.shape,vid.shape)
        img_i = vid
        mask = alphas[None,:].expand(1,3,H,W).clone()
        mask_og = mask.clone()
        print(alphas.shape,vid.shape)
        img_i = th.cat([img_i,alphas[None,:]],1)
        mask = mask2col(mask[0])[None,:]
        mask = th.cat([mask,alphas[None,:]],1)

        animate_sims(mask[0,:3],dname,vname,pname,method,frame,crop,ix)
        # # animate_sims(dname,vname,pname,method,frame,crop,1)
        # exit()

        for hi in range(loc[0]-K,loc[0]+K):
            for wi in range(loc[1]-K,loc[1]+K):
                # print(spix[0,hi,wi].item(),down[:,th.where(uniq==spix[0,hi,wi])])
                img_i[:,:3,hi,wi] = 0.2*img_i[:,:3,hi,wi] + th.tensor([[0.,0.,1.]])*0.8
                img_i[:,3,hi,wi] = 1.0
                mask[:,:3,hi,wi] = 0.2*mask[:,:3,hi,wi] + th.tensor([[0.,0.,1.]])*0.8
                mask[:,3,hi,wi] = 1.0

        color = th.tensor([1.0,1.0,1.0])*0.7 # boundary color
        img_i[:,:3] = get_marked_video(img_i[:,:3],spix,color)
        mask[:,:3] = get_marked_video(mask[:,:3],spix,color)
        save_image(img_i,save_root/("alphas%d.png"%ix))
        mask[:,3] = 1.0
        save_image(mask,save_root/("mask%d.png"%ix))

if __name__ == "__main__":
    main()
