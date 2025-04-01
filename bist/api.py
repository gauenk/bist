# Python API

# -- pytorch api --
import math
import torch as th
from einops import rearrange,repeat

# -- connect to c++/cuda api --
import bin.bist_cuda
from .utils import extract

def run(vid,flows,**kwargs):
    
    # -- unpack --
    defaults = {"sp_size":25,"potts":10.0,"sigma_app":0.009,"alpha":math.log(0.5),
                "iperc_coeff":4.0,"thresh_new":5e-2,"thresh_relabel":1e-6,
                "split_alpha":0.0,"target_nspix":0,"video_mode":True,"rgb2lab":True}
    kwargs = extract(kwargs,defaults)
    sp_size = kwargs['sp_size']
    niters = sp_size
    potts = kwargs['potts']
    sigma_app = kwargs['sigma_app']
    alpha = kwargs['alpha']
    iperc_coeff = kwargs['iperc_coeff']
    thresh_new = kwargs['thresh_new']
    thresh_relabel = kwargs['thresh_relabel']
    split_alpha = kwargs['split_alpha']
    target_nspix = kwargs['target_nspix']
    video_mode = kwargs['video_mode']
    rgb2lab = kwargs['rgb2lab']

    # -- prep --
    assert vid.shape[-1] == 3,"Last Dimension Must be 3 Color Channels or 3 Features"

    # -- run --
    fxn = bin.bist_cuda.run_bist
    spix = fxn(vid,flows,niters,sp_size,potts,sigma_app,alpha,
               iperc_coeff,thresh_new,thresh_relabel,
               split_alpha,target_nspix,video_mode,rgb2lab)
    return spix

def get_marked_video(vid,spix,color):
    color = color.cuda().contiguous().float()
    islast = vid.shape[-1] == 3
    marked = bin.bist_cuda.get_marked_video(vid,spix,color)
    return marked

def shift_labels(labels,spix,flow,sizes=None):
    if sizes is None:
        sizes = th.bincount(spix.ravel())
    nspix = spix.max().item()+1
    shifted = bin.bist_cuda.shift_labels(labels,spix,flow,sizes,nspix)
    return shifted

def get_pooled_video(vid, mask):
    """
    Compute the average value of the vid for each unique mask value and return both:
    - "down": shape (T, S, C), where S is the max mask value + 1
    - "pool": shape (T, H, W, C), where each pixel is replaced by its mask's average value
    """
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
    down.scatter_reduce_(1, mask_flat.unsqueeze(2).expand(-1, -1, C), vid_flat, reduce="sum")
    count.scatter_reduce_(1, mask_flat.unsqueeze(2), th.ones_like(vid_flat), reduce="sum")

    # Avoid division by zero
    count[count == 0] = 1
    down /= count

    # Map back to full resolution (restore to (T, H, W, C) shape)
    pool = down.gather(1, mask_flat.unsqueeze(2).expand(-1, -1, C)).view(T, H, W, C)

    # Final reshape for "down" to (T, S, C)
    down = rearrange(down, 't s c -> t s c')
    pool = rearrange(pool, 't h w c -> t h w c')

    return pool, down