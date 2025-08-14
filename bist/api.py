# Python API

# -- pytorch api --
import math
import torch as th
from pathlib import Path
from einops import rearrange,repeat

# -- run c++ api --
import subprocess

# -- connect to c++/cuda api --
import bin.bist_cuda
import bin.bist_cuda as bist_cuda
from .utils import extract
from ._paths import BIST_HOME


def run(vid,flows,**kwargs):

    # -- unpack --
    defaults = default_params()
    kwargs = extract(kwargs,defaults)
    sp_size = kwargs['sp_size']
    niters = kwargs['niters'] if 'niters' in kwargs else sp_size
    potts = kwargs['potts']
    sigma_app = kwargs['sigma_app']
    alpha = kwargs['alpha']
    gamma = kwargs['gamma']
    epsilon_new = kwargs['epsilon_new']
    epsilon_reid = kwargs['epsilon_reid']
    split_alpha = kwargs['split_alpha']
    target_nspix = kwargs['target_nspix']
    video_mode = kwargs['video_mode']
    use_sm = kwargs['use_sm']
    rgb2lab = kwargs['rgb2lab']

    # -- prep --
    assert vid.shape[-1] == 3,"Last Dimension Must be 3 Color Channels or 3 Features"
    assert flows.shape[-1] == 2,"Last Dimension Must be 2 Channels"

    # -- run --
    # print("niters: ",niters)
    fxn = bin.bist_cuda.run_bist
    spix = fxn(vid,flows,niters,sp_size,potts,sigma_app,alpha,
               gamma,epsilon_new,epsilon_reid,split_alpha,target_nspix,
               video_mode,use_sm,rgb2lab)
    return spix

def run_bin(vid_root,flow_root,spix_root,img_ext,**kwargs):

    # -- unpack --
    defaults = default_params()
    kwargs = extract(kwargs,defaults)
    sp_size = kwargs['sp_size']
    niters = kwargs['niters'] if 'niters' in kwargs else sp_size
    potts = kwargs['potts']
    sigma_app = kwargs['sigma_app']
    alpha = kwargs['alpha']
    gamma = kwargs['gamma']
    epsilon_new = kwargs['epsilon_new']
    epsilon_reid = kwargs['epsilon_reid']
    split_alpha = kwargs['split_alpha']
    tgt_nspix = kwargs['target_nspix']
    video_mode = kwargs['video_mode']
    rgb2lab = kwargs['rgb2lab']
    prop_nc = kwargs['prop_nc']
    prop_icov = kwargs['prop_icov']
    overlap = kwargs['overlap']
    logging = kwargs['logging']
    nimgs = kwargs['nimgs']
    verbose = kwargs['verbose']
    save_only_spix = 1 if kwargs['save_only_spix'] else 0
    read_video = 1 if video_mode else 0
    bist_bin = str(Path(BIST_HOME)/"bin/bist")

    # -- ensure strings --
    vid_root,flow_root,spix_root = str(vid_root),str(flow_root),str(spix_root)

    # -- prepare command --
    cmd = "%s -n %d -d %s/ -f %s/ -o %s/ --read_video %d --img_ext %s --sigma_app %2.5f --potts %2.2f --alpha %2.3f --split_alpha %2.3f --tgt_nspix %d --gamma %2.2f --epsilon_reid %1.8f --epsilon_new %1.8f --prop_nc %d --prop_icov %d --overlap %d --logging %d --nimgs %d --save_only_spix %d" % (bist_bin,sp_size,vid_root,flow_root,spix_root,read_video,img_ext,sigma_app,potts,alpha,split_alpha,tgt_nspix,gamma,epsilon_reid,epsilon_new,prop_nc,prop_icov,overlap,logging,nimgs,save_only_spix)

    # -- run binary --
    print(cmd)
    if verbose:
        print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    return output

def smloop(img,init_spix,**kwargs):
    # -- unpack --
    defaults = default_params()
    kwargs = extract(kwargs,defaults)
    sp_size = kwargs['sp_size']
    niters = kwargs['niters'] if 'niters' in kwargs else sp_size
    sigma_app = kwargs['sigma_app']
    potts = kwargs['potts']
    alpha = kwargs['alpha']
    split_alpha = kwargs['split_alpha']
    rgb2lab = kwargs['rgb2lab']

    # -- run --
    fxn = bin.bist_cuda.smloop
    spix = fxn(img,init_spix,niters,sp_size,sigma_app,potts,alpha,split_alpha,rgb2lab)
    return spix


def default_params():
    defaults = {"sp_size":25,"potts":10.0,"sigma_app":0.009,
                "alpha":math.log(0.5),"gamma":4.0,
                "epsilon_new":5e-2,"epsilon_reid":1e-6,
                "prop_nc":1,"prop_icov":1,"overlap":1,
                "split_alpha":0.0,"logging":0,
                "target_nspix":0,"nimgs":0,
                "video_mode":True,"rgb2lab":True,
                "use_sm":True,"save_only_spix":True,"verbose":False}
    return defaults

def get_marked_video(vid,spix,color):
    islast = vid.shape[-1] == 3
    assert islast is True
    color = color.contiguous().cuda().float()
    vid = vid.contiguous().cuda().float()
    spix = spix.contiguous().cuda().int()
    marked = bin.bist_cuda.get_marked_video(vid,spix,color)
    return marked

def shift_labels(labels,spix,flow,sizes=None):
    if sizes is None:
        sizes = th.bincount(spix.ravel())
    nspix = spix.max().item()+1
    shifted = bin.bist_cuda.shift_labels(labels,spix,flow,sizes,nspix)
    return shifted

# def get_pooled_video(vid, mask, use3d=False):
#     from st_spix.sp_pooling import pooling
#     vid = vid.contiguous()
#     mask = mask.contiguous()
#     pool,down = pooling(vid,mask,mask.max().item()+1)
#     return pool,down

def get_pooled_video(vid, mask, use3d=False, return_pool=True, cdim=-1):

    """
    Compute the average value of the vid for each unique mask value and return both:
    - "down": shape (T, S, C), where S is the max mask value + 1
    - "pool": shape (T, H, W, C), where each pixel is replaced by its mask's average value
    """

    if cdim == 1:
        vid = rearrange(vid,'t c h w -> t h w c').contiguous()

    vid = vid.float()  # Ensure the video tensor is float
    T, H, W, C = vid.shape  # Unpack the shape with (T, H, W, C)
    mask = mask.long()  # Ensure mask is long type for indexing

    S = mask.max().item() + 1  # The number of possible mask values
    mask_flat = mask.view(T, H * W)  # Flatten spatially
    vid_flat = vid.contiguous().view(T, H * W, C)  # Flatten spatially for video

    # Compute sum per mask value
    down = th.zeros((T, S, C), device=vid.device, dtype=th.float32)
    count = th.zeros((T, S, 1), device=vid.device, dtype=th.float32)  # Fix count shape

    # Scatter reduce to compute the sum per mask value
    down.scatter_add_(1, mask_flat.unsqueeze(2).expand(-1, -1, C), vid_flat)
    count.scatter_add_(1, mask_flat.unsqueeze(2), th.ones_like(vid_flat))

    # Possibly average over time
    if use3d:
        down = down.sum(0,keepdim=True).repeat(T,1,1)
        count = count.sum(0,keepdim=True).repeat(T,1,1)

    # Avoid division by zero
    count = count + (count == 0).float()  # Prevent division by zero
    down /= count

    # Map back to full resolution (restore to (T, H, W, C) shape)
    if return_pool:
        pool = down.gather(1, mask_flat.unsqueeze(2).expand(-1, -1, C)).view(T, H, W, C)
        if cdim == 1:
            pool = rearrange(pool,'t h w c -> t c h w').contiguous()
        return pool, down
    else:
        return down
