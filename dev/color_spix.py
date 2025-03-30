"""

    Color superpixels to inspect their lifetime

"""

import torch as th
import numpy as np
from einops import rearrange,repeat

import pandas as pd
import torchvision.io as tvio
import torchvision.utils as tv_utils

from skimage.segmentation import mark_boundaries

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt




def compute_hists(spix):
    nspix = spix.max()+1
    T,H,W = spix.shape
    hists = []
    for t in range(T):
        hists.append(np.bincount(spix[t].ravel(),minlength=nspix))
    hists = np.stack(hists)
    return hists

def plot_hist(root,name,hist):
    # Create the histogram
    sns.histplot(hist)
    plt.savefig(Path(root)/name,dpi=300)
    plt.close("all")

def plot_hists(root,hists):
    T,S = hists.shape
    for t in range(T):
        plot_hist(root,"%05d"%t,hists[t])


def get_nimgs(path):
    nimgs = 0
    for fn in path.iterdir():
        if fn.suffix in [".png",".jpg"]: nimgs+=1
    return nimgs

def read_video(path,nimgs):
    istart = 1
    vid = []
    for index in range(istart,nimgs+istart):
        fn = path / ("%05d.png" % index)
        img = tvio.read_image(fn)/255.
        vid.append(img)
    vid = np.stack(vid)
    return vid

def read_flo(filename):
    """
    Reads a .flo optical flow file and returns it as a numpy array.

    Args:
        filename (str): Path to the .flo file.

    Returns:
        numpy.ndarray: The optical flow array of shape (height, width, 2).
    """
    with open(filename, 'rb') as f:
        # Read the magic number and check its validity
        magic = f.read(4)
        if magic != b'PIEH':
            raise ValueError(f"Invalid .flo file: {filename}")

        # Read the width and height
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # Read the optical flow data
        flow_data = np.frombuffer(f.read(), dtype=np.float32)

        # Reshape the data to (height, width, 2)
        flow = flow_data.reshape((height, width, 2))

    return flow

def read_flow(path,nimgs):
    istart = 1
    flow = []
    for index in range(istart,nimgs+istart):
        fn = path / ("%05d.flo" % index)
        if not fn.exists(): continue
        flow_t = read_flo(fn)
        flow_t = rearrange(flow_t,'h w c -> c h w')
        flow.append(flow_t)
    flow = np.stack(flow)
    return flow

    stem = "%05d" % index
    flow_fn = flow_dir / ("%s.flo"%stem)
    flow = read_flo(flow_fn)


def read_spix(path,nimgs):
    istart = 1
    spix = []
    for index in range(istart,nimgs+istart):
        fn = path / ("%05d.csv" % index)
        spix_t = pd.read_csv(str(fn),header=None).to_numpy()
        spix.append(spix_t)
    spix = np.stack(spix)
    return spix

def color_spix(vid,spix,spix_id,cidx=0):
    for t in range(vid.shape[0]):
        for ci in range(3):
            vid[t,ci][th.where(spix[t]==spix_id)] = 1.*(ci==cidx)
    return vid

def color_spix_by_mask(vid,mask,cidx=0):
    for ci in range(3):
        vid[:,ci][th.where(mask)] = 1.*(ci==cidx)
    return vid

def get_spix_mask(spix,t,i_range,j_range):
    mask = th.zeros_like(spix)
    for i in range(i_range[0],i_range[1]):
        for j in range(j_range[0],j_range[1]):
            c = spix[t,i,j].item()
            mask[th.where(spix==c)] = 1
    print("unique mask: ",th.unique(spix[th.where(mask)]))
    # print("[0] unique mask: ",th.unique(spix[0][th.where(mask[0])]))
    return mask


def mark_spix_vid(vid,spix,mode=None):
    if mode is None: mode = "subpixel"
    is_tensor = th.is_tensor(vid)
    if th.is_tensor(vid): vid = vid.detach().cpu().numpy()
    if th.is_tensor(spix): spix = spix.detach().cpu().numpy()
    marked = []
    for ix,spix_t in enumerate(spix):
        img = rearrange(vid[ix],'f h w -> h w f')
        if mode is None:
            marked_t = mark_boundaries(img,spix_t)
        else:
            marked_t = mark_boundaries(img,spix_t,mode=mode)
        marked_t = rearrange(marked_t,'h w f -> f h w')
        marked.append(marked_t)
    marked = np.stack(marked)
    if is_tensor: marked = th.from_numpy(marked)
    return marked


def check_flow(vid,flow,spix):

    fn = "flow.csv"
    flow_t = pd.read_csv(fn,header=None).to_numpy()
    flow_t = np.stack([flow_t[:,::2],flow_t[:,1::2]],0)
    flow_t = th.from_numpy(flow_t)

    fn = "flow_ds.csv"
    flow_ds = pd.read_csv(fn,header=None).to_numpy()
    flow_ds = np.stack([flow_ds[:,::2],flow_ds[:,1::2]],0)
    flow_ds = th.from_numpy(flow_ds)[:,0]
    print(flow_ds.shape)


    num = th.sum(spix[0]==332)
    print(flow[0,0][th.where(spix[0]==332)].mean())
    print(flow[0,1][th.where(spix[0]==332)].mean())
    print(flow_ds[:,332],flow_ds[:,332]/num,num)

    # print(flow.shape,flow_t.shape)
    # print(flow.numpy()[0,:5,:5])
    # print(flow_t[:5,:5])
    # print(flow_t)
    # print(th.any(th.isnan(flow_t)))
    # print(th.where(th.isnan(flow_t)))
    # print(flow_t[1,133])
    # print(flow[0,1,133])
    # print(flow_t[1,132])
    # print(flow[0,1,132])
    delta = th.mean((flow[0] - flow_t)**2)
    print("delta: ",delta)
    exit()


def main():

    # -- base path --
    base = Path("result/color_spix/")
    if not base.exists(): base.mkdir()

    # -- paths --
    vid_path = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/PNGImages/frog_2/")
    flow_path = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/PNGImages/frog_2/BIST_flows/")
    spix_path = Path("result/bist/sp15/frog_2/")

    # -- read vid, flow, and spix --
    nimgs = get_nimgs(vid_path)
    nimgs = 100
    # nimgs = 10
    vid = read_video(vid_path,nimgs)
    flow = read_flow(flow_path,nimgs)
    spix = read_spix(spix_path,nimgs)
    print(vid.shape,flow.shape,spix.shape)
    # exit()

    # -- subset --
    vid = vid[::10]
    spix = spix[::10]
    # print(vid.shape,spix.shape)

    # -- plot hists --
    hist_root = base / "hists"
    if not hist_root.exists(): root.mkdir()
    hists = compute_hists(spix)
    plot_hists(hist_root,hists)

    # -- mark image with boundary --
    # marked = mark_spix_vid(vid,spix,mode="subpixel")
    # marked = th.from_numpy(marked)
    # tv_utils.save_image(marked,base / "marked.png")

    # -- to torch now --
    vid = th.from_numpy(vid)
    flow = th.from_numpy(flow)
    spix = th.from_numpy(spix)

    # -- debug flow --
    # check_flow(vid,flow,spix)

    # -- color superpixel --
    marked_c = vid.clone()
    # print(marked_c.shape)
    # mask0 = get_spix_mask(spix,0,[100,150],[280,340])
    # mask1 = get_spix_mask(spix,0,[100,150],[200,280])
    # mask2 = get_spix_mask(spix,0,[50,100],[200,280])

    mask0 = get_spix_mask(spix,0,[100,150],[280,340])
    mask1 = get_spix_mask(spix,0,[100,150],[200,280])
    mask2 = get_spix_mask(spix,0,[50,100],[200,280])

    marked_c = color_spix_by_mask(marked_c,mask0,cidx=0)
    marked_c = color_spix_by_mask(marked_c,mask1,cidx=1)
    marked_c = color_spix_by_mask(marked_c,mask2,cidx=2)
    # marked_c = color_spix(marked_c,spix,331,cidx=0)
    marked_c = mark_spix_vid(marked_c,spix,mode="subpixel")

    tv_utils.save_image(marked_c,base / "marked_c.png")


if __name__ == "__main__":
    main()
