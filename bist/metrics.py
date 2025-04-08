
# -- metrics --
import numpy as np
import torch as th
from einops import rearrange
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as compute_ssim_ski
from skvideo.measure import strred as comp_strred

def compute_asa(sp,gt):

    # -- prepare --
    if not th.is_tensor(sp):
        sp = th.from_numpy(sp*1.).long()
    if not th.is_tensor(gt):
        gt = th.from_numpy(gt*1.).long()

    # -- normalize --
    gt = (gt*1.).long()-int(gt.min().item())

    # -- unpack --
    device = sp.device
    H,W = sp.shape

    # -- allocate hist --
    Nsp = max(len(sp.unique()),int(sp.max()+1))
    Ngt = max(len(gt.unique()),int(gt.max()+1))
    hist = th.zeros(Nsp*Ngt,device=device)

    # -- fill hist --
    inds = sp.ravel()*Ngt+gt.ravel()
    ones = th.ones_like(inds).type(hist.dtype)
    hist = hist.scatter_add_(0,inds,ones)
    hist = hist.reshape(Nsp,Ngt)

    # -- max for each superpixel across gt segs --
    maxes = th.max(hist,1).values
    asa = th.sum(maxes)/(H*W)

    return asa.item()

def get_brbp_edges(sp,gt,r=1):

    # -- prepare --
    if not th.is_tensor(sp):
        sp = th.from_numpy(sp*1.).long()
    if not th.is_tensor(gt):
        gt = th.from_numpy(gt*1.).long()

    # -- normalize gt --
    gt = gt.long()-1

    # -- unpack --
    device = sp.device
    H,W = sp.shape

    # -- get edges --
    edges_sp = th.zeros_like(sp[:-1,:-1]).bool()
    edges_gt = th.zeros_like(gt[:-1,:-1]).bool()
    for ix in range(2):
        for jx in range(2):
            # print(ix,H-1+ix,jx,W-1+jx,sp.shape)
            edges_sp = th.logical_or(edges_sp,(sp[ix:H-1+ix,jx:W-1+jx] != sp[1:,1:]))
            edges_gt = th.logical_or(edges_gt,(gt[ix:H-1+ix,jx:W-1+jx] != gt[1:,1:]))
    Nsp_edges = th.sum(edges_sp)

    # -- fuzz the edges_sp --
    if r > 0:
        pool2d = th.nn.functional.max_pool2d
        ksize = 2*r+1
        edges_sp = edges_sp[None,None,:]*1.
        edges_sp = pool2d(edges_sp,ksize,stride=1,padding=ksize//2)
        edges_sp = edges_sp[0,0].bool()

    return edges_sp,edges_gt,Nsp_edges

def compute_bp(sp,gt,r=1):

    # -- get edges --
    edges_sp,edges_gt,Nsp_edges = get_brbp_edges(sp,gt,r)

    # -- compute number equal --
    br = th.sum(1.*edges_sp*edges_gt)/th.sum(edges_sp)
    return br.item()

def compute_br(sp,gt,r=1):

    # -- get edges --
    edges_sp,edges_gt,_ = get_brbp_edges(sp,gt,r)

    # -- compute number equal --
    br = th.sum(1.*edges_sp*edges_gt)/th.sum(edges_gt)
    return br.item()


def compute_batched(compute_fxn,clean,deno,div=255.):
    metric = []
    for b in range(len(clean)):
        metric_b = compute_fxn(clean[b],deno[b],div)
        metric.append(metric_b)
    metric = np.array(metric)
    return metric

def compute_ssims(clean,deno,div=1.):
    # -- optional batching --
    if clean.ndim == 5:
        return compute_batched(compute_ssims,clean,deno,div)

    # -- to numpy --
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()

    # -- standardize image --
    # if np.isclose(div,1.):
    #     deno = deno.clip(0,1)*255.
    #     clean = clean.clip(0,1)*255.
    #     deno = deno.astype(np.uint8)/255.
    #     clean = clean.astype(np.uint8)/255.
    # elif np.isclose(div,255.):
    #     deno = deno.astype(np.uint8)*1.
    #     clean = clean.astype(np.uint8)*1.

    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].transpose((1,2,0))/div
        deno_t = deno[t].transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1,
                                  data_range=1.)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    return ssims

def compute_psnrs(clean,deno,div=1.):
    # -- optional batching --
    if clean.ndim == 5:
        return compute_batched(compute_psnrs,clean,deno,div)
    t = clean.shape[0]

    # -- to numpy --
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()

    # -- standardize image --
    # if np.isclose(div,1.):
    #     deno = deno.clip(0,1)*255.
    #     clean = clean.clip(0,1)*255.
    #     deno = deno.astype(np.uint8)/255.
    #     clean = clean.astype(np.uint8)/255.
    # elif np.isclose(div,255.):
    #     deno = deno.astype(np.uint8)*1.
    #     clean = clean.astype(np.uint8)*1.

    psnrs = []
    t = clean.shape[0]
    for ti in range(t):
        psnr_ti = comp_psnr(clean[ti,:,:,:], deno[ti,:,:,:], data_range=div)
        psnrs.append(psnr_ti)
    return np.array(psnrs)

def seg_metrics(pred,anno):
    pred = th.sigmoid(pred)
    print(th.unique(anno.ravel()))
    assert len(th.unique(anno.ravel())) <= 2
    args_p = th.where(anno==1)
    args_n = th.where(anno==0)
    npos = 1.*anno[args_p].numel() + 1.*(anno[args_p].numel()==0)
    nneg = 1.*anno[args_n].numel() + 1.*(anno[args_n].numel()==0)
    tpos = (pred[args_p]>0.5).sum()/npos
    tneg = (pred[args_n]<0.5).sum()/nneg
    perc_pos = npos / (npos + nneg)
    return tpos,tneg,perc_pos


def compute_strred(clean,deno,div=1):

    # -- optional batching --
    if clean.ndim == 5:
        return compute_batched(compute_strred,clean,deno,div)

    # -- numpify --
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()

    # -- reshape --
    clean = rearrange(clean,'t c h w -> t h w c')/float(div)
    deno = rearrange(deno,'t c h w -> t h w c')/float(div)

    # -- bw --
    if clean.shape[-1] == 3:
        clean = rgb2gray(clean,channel_axis=-1)
        deno = rgb2gray(deno,channel_axis=-1)

    # -- compute --
    with np.errstate(invalid='ignore'):
        outs = comp_strred(clean,deno)
    strred = outs[1] # get float
    return strred

def _blocking_effect_factor(im):
    im = th.from_numpy(im)

    block_size = 8

    block_horizontal_positions = th.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = th.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(th.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(th.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef

