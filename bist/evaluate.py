

import os,glob
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

import torch as th
import h5py
from PIL import Image
from einops import rearrange

import bist

def run(vid,anno,spix):

    # -- setup --
    spix = spix - spix.min() # fix offset by one
    summ = edict()
    spix_ids = np.unique(spix)
    gtpos = np.arange(anno.shape[0])
    sizes = count_spix(spix)
    anno_sizes = count_spix(anno)[:,1:] # skip 0 for anno
    gt_ids = np.unique(anno) # keep 0 for anno

    # -- info --
    device = "cuda:0"
    vid = vid.to(device).float()
    spix = spix.to(device).int()
    anno = anno.to(device).int()
    # vid = th.from_numpy(vid).to(device).float()
    # spix = th.from_numpy(spix).to(device).int()
    # anno = th.from_numpy(anno).to(device).int()
    sizes = sizes.to(device)
    anno_sizes = anno_sizes.to(device)

    # -- summary --
    summ.tex,summ.szv = scoreMeanDurationAndSizeVariation(sizes)
    summ.ev = scoreExplainedVariance(vid,spix)
    summ.ev3d = scoreExplainedVariance3D(vid,spix)
    summ.pooling,pooled = scoreSpixPoolingQuality(vid,spix)
    outs = scoreUEandSA(spix, sizes, anno, anno_sizes, gt_ids, gtpos)
    summ.ue2d,summ.sa2d,summ.ue3d,summ.sa3d = outs

    # -- summary stats --
    summ.spnum = spix.max().item()+1
    summ.ave_nsp = average_unique_spix(sizes)

    # -- optional strred --
    summ.strred0 = -1
    summ.strred1 = -1
    strred = False
    if strred:
        # from skvideo import measure
        _pooled = rearrange(pooled,'t c h w -> t h w c')
        _pooled = rgb_to_luminance_torch(_pooled)
        _vid = rgb_to_luminance_torch(vid)
        _,score0,score1 = measure.strred(_vid.cpu().numpy(),_pooled.cpu().numpy())
        summ.strred0 = float(score0)
        summ.strred1 = float(score1)


    return summ

def count_spix(spix):

    # -- setup --
    if not th.is_tensor(spix):
        spix = th.from_numpy(spix)
    spix = spix.long()
    device = spix.device
    nframes,H,W = spix.shape
    spix = spix.reshape(nframes,-1)
    nspix = spix.max().item()+1

    # -- allocate --
    counts = th.zeros((nframes, nspix), dtype=th.int32, device=device)
    ones = th.ones_like(spix, dtype=th.int32, device=device)
    if spix.min() < 0:
        print("invalid spix!")
        print(th.where(spix == spix.min()))
        exit()
    counts.scatter_add_(1, spix, ones)

    return counts


def rgb_to_luminance_torch(rgb):
    # Rec. 709 coefficients (sRGB)
    weights = th.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return th.tensordot(rgb, weights, dims=([-1], [0]))[...,None]  # Sum along RGB channels

def average_unique_spix(sizes):
    return th.mean(1.*th.sum(sizes>0,-1)).item() # sum across spix; ave across time

def scoreSpixPoolingQualityByFrame(vid,spix,metric="psnr"):
    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- pooling --
    pooled,down = bist.get_pooled_video(vid,spix)
    vid = rearrange(vid,'t h w f -> t f h w')
    pooled = rearrange(pooled,'t h w f -> t f h w')
    if metric == "psnr":
        res = bist.metrics.compute_psnrs(vid,pooled,div=1.)
    elif metric == "ssim":
        res = bist.metrics.compute_ssims(vid,pooled,div=1.)
    else:
        raise ValueError(f"Uknown metric name [{metric}]")
    return res,pooled

def scoreSpixPoolingQuality(vid,spix):
    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- pooling --
    pooled,down = bist.get_pooled_video(vid,spix)
    vid = rearrange(vid,'t h w f -> t f h w')
    pooled = rearrange(pooled,'t h w f -> t f h w')
    psnr = bist.metrics.compute_psnrs(vid,pooled,div=1.).mean().item()
    return psnr,pooled

def scoreExplainedVariance(vid,spix):
    # roughly: var(sp_mean) / var(pix)
    # but sp_mean is computed per frame

    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- Global mean --
    mean_global = vid.mean(dim=(1,2),keepdim=True)  # Mean across pixels (dim=0)

    # -- pixels vs mean --
    pix2mean = ((vid - mean_global) ** 2).sum((-1,-2,-3))  # Sum over color channels

    # -- sp-aves v.s. mean --
    pooled,down = bist.get_pooled_video(vid,spix)
    vid = rearrange(vid,'t h w f -> t f h w')
    pool2mean = ((pooled - mean_global)**2).sum((-1,-2,-3))
    score = (pool2mean / (pix2mean+1e-10)).mean().item()
    return score

def scoreExplainedVariance3D(vid,spix):
    # roughly: var(sp_mean) / var(pix)
    # but sp_mean is computed across all frames


    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- get global mean per frame --
    vid = 255.*vid
    mean_global = vid.mean(dim=(1,2),keepdim=True)  # Mean across pixels

    # -- compare unnormalized variance of pixels --
    pix2mean = ((vid - mean_global) ** 2).sum((-1,-2,-3))  # Sum over color channels

    # -- unnormalized variance of superpixels --
    pooled,down = bist.get_pooled_video(vid,spix,use3d=True)
    # print("Likely an issue here. Check if actually using this.")
    pooled2mean = ((pooled - mean_global)**2).sum((-1,-2,-3))

    # -- divide and average across # of frames --
    score = (pooled2mean/(pix2mean+1e-10)).mean().item()

    return score


def scoreMeanDurationAndSizeVariation(counts):

    # -- temporal extent; how long is each pixel alive? --
    TEX = th.mean(1.*(counts>0),1).mean().item()

    # -- how much does each superpixel change shape? --
    T,S = counts.shape
    counts = counts.to("cuda")
    mask = counts > 0
    num_valid = th.sum(mask, dim=0)
    num_valid_s = th.clamp(num_valid, min=1)

    # -- compute unbiased variance --
    sum_counts = th.sum(counts * mask, dim=0)  # Sum along T (ignoring zeros)
    sum_counts_sq = th.sum((counts**2) * mask, dim=0)  # Sum of squares along T
    variance = (sum_counts_sq / num_valid_s) - (sum_counts / num_valid_s) ** 2
    correction = num_valid_s/th.clamp(num_valid_s-1,min=1)
    variance = correction * variance
    stds = th.sqrt(variance)

    # -- compute average variance of only valid points --
    args = th.where(num_valid>1)
    stds = stds[args]
    SZV = th.mean(stds)
    SZV = SZV.item()
    return TEX, SZV


def scoreUEandSA(spix, counts, gtSeg, gtSize, gtList, gtPos):
    """
    This function is used to score 3D Undersegmentation Error and 3D
    Segmentation Accuracy for supervoxels.
    """

    # In case ground-truth is sparsely annotated
    spix = spix[gtPos, :, :] # T H W
    if spix.shape != gtSeg.shape:
        print('Error: gtSeg and spix dimension mismatch!')
        return -1., -1., -1., -1.

    # -- init --
    device = spix.device
    T,H,W = spix.shape
    T,S = counts.shape
    numGT = len(gtList)
    gtUE = th.zeros((T,numGT),device=device)
    gtUE3D = th.zeros((numGT),device=device)
    gtSA = th.zeros((T,numGT),device=device)
    gtSA3D = th.zeros((numGT),device=device)

    # -- setup for alt --
    counts = counts.long()
    spix = spix.long()
    gtSeg = gtSeg.long()
    gtSize = gtSize.long()
    invalid = spix.max().item()+1

    for i in range(len(gtList)): # number of masks

        # -- get counts overlapping with mask --
        gt_id = gtList[i]
        invalid_mask = gtSeg != int(gt_id)
        spix_i = spix.clone()
        spix_i[th.where(invalid_mask)] = invalid
        in_counts = count_spix(spix_i)[:,:S].long() # remove invalid

        # -- corrected ue --
        out_counts = th.zeros((T,S),device=device,dtype=th.long)
        min_counts = th.zeros((T,S),device=device,dtype=th.long)
        args = th.where(in_counts>0)
        out_counts[args] = counts[args] - in_counts[args]
        min_counts[args] = th.minimum(out_counts[args],in_counts[args])
        gtUE[:,i] = min_counts.sum(-1)

        # -- compute ue 3D --
        min_counts_s = th.zeros((S),device=device,dtype=th.long)
        out_counts_s = out_counts.sum(0)
        in_counts_s = in_counts.sum(0)
        args = th.where(in_counts_s>0)
        min_counts_s[args] = th.minimum(out_counts_s[args],in_counts_s[args])
        gtUE3D[i] = min_counts_s.sum()

        # -- compute sa 2d --
        gtSA_i = th.zeros((T,S),device=device,dtype=th.long)
        args = th.where(in_counts >= (0.5 * counts))
        gtSA_i[args] = in_counts[args]
        gtSA[:,i] = gtSA_i.sum(-1)

        # -- compute sa 3d --
        in_counts = in_counts.sum(0)
        args = th.where(in_counts >= (0.5 * counts.sum(0)))
        gtSA3D[i] = in_counts[args].sum()


    # -- info --
    # UE = how many extra pixels for all spix in the GT.
    # SA = how many pixels are more than half in the GT
    # UE_2d = th.mean((gtUE - gtSize) / gtSize)
    # SA_2d = th.mean(gtSA / gtSize)

    # -- remove gtlabel "0" for SA --
    gtSA = gtSA[:,1:]
    gtSA3D = gtSA3D[1:]

    # -- 2d metrics ["masked" average dropping the "0" frames --
    gtSize_mask = gtSize + (gtSize == 0)
    UE_2d = th.mean(gtUE / (H*W))
    SA_2d = th.mean(th.sum(gtSA / gtSize_mask, axis=0) / th.sum(gtSize > 0, axis=0))

    # -- 3d metrics --
    gtUE = gtUE.sum(0)
    gtSA = gtSA.sum(0)
    gtSize = gtSize.sum(0)
    assert th.all(gtSize>0).item()
    UE_3d = th.mean(gtUE3D / (T*H*W))
    SA_3d = th.mean(gtSA3D / gtSize)

    return UE_2d.item(),SA_2d.item(),UE_3d.item(),SA_3d.item()


