import torch as th
from einops import rearrange,repeat

def ensure_color_channel(tensor):
    islast = tensor.shape[-1] <= 4
    if tensor.shape[1] <= 4:
        tensor = rearrange(tensor,'b c h w -> b h w c')
    return tensor,islast

def restore_color_channel(tensor,islast):
    if islast:
        return tensor
    return rearrange(tensor,'b h w c -> b c h w')

def shift_tensor(tensor,index):

    # -- reshape --
    tensor,islast = ensure_color_channel(tensor)
    B,H,W,C = tensor.shape
    tensor = rearrange(tensor,'b h w c -> h w (b c)')
    shifted = th.zeros_like(tensor).to(tensor.device)
    # th.cuda.synchronize()
    # print(tensor.shape)
    # print(index.shape)

    # Mask out the valid indices
    valid_mask = index >= 0
    valid_indices = index[valid_mask]  # Extract valid indices

    # Compute the row/col positions of valid indices in the original image
    rows, cols = th.meshgrid(th.arange(H), th.arange(W), indexing='ij')
    rows = rows.to(tensor.device).long()
    cols = cols.to(tensor.device).long()
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]

    # Map the valid indices to their new positions
    shifted[valid_rows, valid_cols] = tensor.reshape(-1, B*C)[valid_indices]
    shifted = rearrange(shifted,'h w (b c) -> b h w c',b=B)
    shifted = restore_color_channel(shifted,islast)

    return shifted
