import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange


import glob
from pathlib import Path
from run_eval import read_video,read_seg,get_video_names
from st_spix.utils import rgb2lab
from st_spix.spix_utils.evaluate import computeSummary,scoreSpixPoolingQualityByFrame,count_spix,read_spix


def check_spix_from_splits():
    seg = pd.read_csv("debug_seg.csv",header=None).to_numpy()
    seg1 = pd.read_csv("debug_seg1.csv",header=None).to_numpy()
    seg2 = pd.read_csv("debug_seg2.csv",header=None).to_numpy()
    # nspix = 502+1
    nspix = 600
    print(seg.shape,seg1.shape,seg2.shape)

    x,y = np.where(seg==0)
    mx = round(np.mean(1*x))
    my = round(np.mean(1*y))
    print(x,y)
    print(mx,my)
    print(seg[mx-2:mx+2,my-2:my+2])
    # exit()


    print("."*10)
    print(seg[:10,:10])
    print(seg1[:10,:10])
    print(seg2[:10,:10])

    print("."*10)
    print(len(np.unique(seg)))
    print(np.sum(seg>500))


    print("."*10)
    print(np.sum(seg==0))
    print(np.sum(seg1==0))
    print(np.sum(seg2==0))
    print(np.sum(seg1==0+nspix))
    print(np.sum(seg2==0+nspix))

    print("."*10)
    print("ninvalid")
    print(np.sum(seg1==-1))
    print(np.sum(seg2==-1))

    # print(np.sum(seg1==1))
    # print(np.sum(seg2==1))

def check_nspix_across_time():
    # vname = "monkey"
    # vname = "frog_2"
    # root = Path("result/dev/%s/"%vname)
    # spix = read_spix(root,vname,1)
    # vname = "car-roundabout"
    vname = "cows"
    root = Path("result_davis/bass/param0/%s/"%vname)
    # root = Path("result_davis/bist/param0/%s/"%vname)
    spix = read_spix(root,vname,0)
    for t,spix_t in enumerate(spix):
        print(t,len(np.unique(spix_t)))

# def read_spix(root,vname,offset):
#     nframes = len(glob.glob(str(root/"*_params.csv")))
#     spix = []
#     for frame_index in range(offset,nframes+offset):
#         fn = root / ("%05d.csv" % frame_index)
#         spix_t = pd.read_csv(str(fn),header=None).to_numpy()
#         spix.append(spix_t)
#     spix = np.stack(spix)
#     return spix

# def get_dataset_image_info(dname,vname):
#     if "seg" in dname:
#         root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
#         root = root /"SegTrackv2/PNGImages" /vname
#         ext = "png"
#         start = 1
#     elif "davis" in dname:
#         root = Path("/home/gauenk/Documents/data/davis/DAVIS/JPEGImages/480p/")/vname
#         ext = "jpg"
#         start = 0
#     else:
#         raise ValueError(".")
#     return root,ext,start

# def read_video(dname,vname):
#     root,ext,start = get_dataset_image_info(dname,vname)
#     nframes = len([f for f in root.iterdir() if str(f).endswith(ext)])
#     vid = []
#     for frame_ix in range(nframes):
#         fname = root/("%05d.%s" % (frame_ix+start,ext))
#         img = np.array(Image.open(fname).convert("RGB"))/255.
#         vid.append(img)
#     vid = np.stack(vid)
#     return vid

# def get_dataset_seg_info(dname,vname):
#     if "seg" in dname:
#         root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
#         root = root /"SegTrackv2/GroundTruth" /vname
#         ext = "png"
#         start = 1
#     elif "davis" in dname:
#         root = Path("/home/gauenk/Documents/data/davis/DAVIS/Annotations/480p/")/vname
#         ext = "png"
#         start = 0
#     else:
#         raise ValueError(".")
#     return root,ext,start

# def read_seg_loop(dname,root):
#     root,ext,start = get_dataset_seg_info(dname,vname)
#     nframes = len([f for f in root.iterdir() if str(f).endswith(".png")])
#     vid = []
#     for frame_ix in range(nframes):
#         fname = root/("%05d.png" % (frame_ix+start))
#         img = 1.*(np.array(Image.open(fname).convert("L")) >= 128)
#         vid.append(img)
#     vid = np.stack(vid)
#     return vid

# def read_seg(vname):
#     root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
#     root = root /"SegTrackv2/GroundTruth/" /vname
#     has_subdirs = np.all([f.is_dir() for f in root.iterdir()])
#     if has_subdirs:
#         seg = None
#         for ix,subdir in enumerate(root.iterdir()):
#             if seg is None:
#                 seg = read_seg_loop(subdir)
#             else:
#                 tmp = read_seg_loop(subdir)
#                 seg[np.where(tmp>0)] = ix+1
#                 # tmp[np.where(tmp)>0] = ix
#                 # print(ix,np.unique(tmp))
#                 # seg = seg + (ix+1)*read_seg_loop(subdir)
#     else:
#         seg = read_seg_loop(root)
#     # print(np.unique(seg))
#     # exit()
#     return seg


def short_eval(method,dname,pname,vname,tgt_nspix=0):

    # from st_spix.spix_utils.evaluate import computeSummary
    # dname = "davis"
    # vname = "car-roundabout"
    # vname = "car-shadow"
    # vname = "cows"
    # vname = "camel"
    # vname = "blackswan"
    # dname = "segtrackerv2"
    # vname = "frog_2"

    # root = Path("result_davis/bist/param0/%s/"%vname)
    if dname == "davis":
        root = Path("result_davis/%s/%s/%s/"%(method,pname,vname))
        offset = 0
    else:
        root = Path("result/%s/%s/%s/"%(method,pname,vname))
        offset = 1

    # root = Path("result_davis/bass/param0/%s/"%vname)
    # root = Path("result_davis/bist_ftsp/param0/%s/"%vname)
    # root = Path("result/dev/%s/"%vname)
    spix = read_spix(root,vname,offset)

    # -- count nspix --
    counts = count_spix(spix)
    tmp = 1.0*counts[-1]
    tmp = tmp[th.where(tmp>0)[0]]
    print(th.histogram(tmp,5,range=[0.,50.]))
    print(th.histogram(tmp,5,range=[0.,200.]))
    print(th.histogram(tmp,5,range=[0.,2000.]))

    # -- counts --
    counts = 1.*(counts>0)
    nsp_by_frame = th.sum(counts,1)
    print(nsp_by_frame)
    ave_nsp = th.sum(counts,1).mean(0).item()
    print(f"[{method},{vname}] Average Nsp: ",ave_nsp)

    # print(tgt_nspix)
    valid_nsp = th.logical_and(tgt_nspix*0.9 < nsp_by_frame,nsp_by_frame < tgt_nspix*1.1)
    # print(valid_nsp)
    valid_nsp = th.mean(1.0*valid_nsp)>0.90
    # valid_nsp = (tgt_nspix*0.9 < ave_nsp) and (ave_nsp < tgt_nspix*1.1)
    return valid_nsp


    # -- read data --
    vid = read_video(dname,vname)
    T,H,W,C = vid.shape
    npix = H*W
    # print(vid.shape,spix.shape)
    # seg = read_seg(dname,vname)

    # _summ = computeSummary(vid,seg,spix)
    # _summ.nspix = len(np.unique(spix))
    # print(_summ)

    # psnrs = scoreSpixPoolingQualityByFrame(vid,spix)
    # print(psnrs)
    # print(np.mean(psnrs))

    # ssims = scoreSpixPoolingQualityByFrame(vid,spix,'ssim')
    # print(ssims)
    # print(np.mean(ssims))

def inspect_app_var():
    dname = "davis"
    # vname = "bmx-trees"
    findex = 0
    vnames = ["bike-packing","blackswan","bmx-trees",
              "breakdance","camel","car-roundabout"]
    vname = vnames[5]


    param_fn = "/home/gauenk/Documents/packages/st_spix_refactor/result_davis/bist/param0/%s/%05d_params.csv"%(vname,findex)
    fn = "/home/gauenk/Documents/packages/st_spix_refactor/result_davis/bist/param0/%s/%05d.csv"%(vname,findex)

    # -- read data --
    vid = read_video(dname,vname)
    df = pd.read_csv(param_fn)
    seg = pd.read_csv(fn,header=None).to_numpy()
    spix = np.argmax(np.bincount(seg[40:80,-10:].ravel())).item()

    # -- compute variance --
    vid = read_video(dname,vname)
    vid = rearrange(vid,'b h w c -> b c h w')
    vid = th.from_numpy(vid)
    # print(type(vid))
    img = vid[[findex]]
    # print(img.shape,img.max())
    img_lab = rgb2lab(img)
    B,C,H,W = img.shape
    # print(img_lab.shape,img_lab.max())
    outs = np.bincount(seg.ravel())
    # rand_spix = np.random.permutation(np.unique(seg.ravel()))[:10]
    rand_spix = np.unique(seg.ravel())
    app_vars = []
    for _spix in rand_spix:
        mask = th.from_numpy(seg == _spix)
        app_var = [img_lab[0,i][mask].var().item() for i in range(3)]
        if np.any(np.isnan(app_var)):
            print("NAN @ %d"%_spix)
            print(df.iloc[_spix])
            continue
        app_vars.append(np.array(app_var))
        # print("var[%d]: "%_spix,)
    app_vars = np.stack(app_vars).sum(-1)
    print(app_vars.min(),app_vars.max())
    print("quants: ",np.quantile(app_vars,[0.1,0.5,0.8,0.9,0.95]))
    # print(app_vars[spix])
    # exit()

    mask = th.from_numpy(seg == spix)
    locs = 1.*th.stack(th.where(mask))
    app_var = [img_lab[0,i][mask].var().item() for i in range(3)]
    print("var: ",app_var)
    print("means: ",[img_lab[0,i][mask].mean().item() for i in range(3)])
    print("locs: ",[locs[i].mean().item() for i in range(2)])

    print(np.max(outs),np.argmax(outs))
    df['sigma_s'] = df['sigma_s.x'] + df['sigma_s.y'] + df['sigma_s.z']
    sigma = df['sigma_s'].to_numpy()
    counts = df['count'].to_numpy()
    pcounts = df['prior_count'].to_numpy()

    print(np.argmax(sigma),np.max(sigma))
    print(np.argmax(counts),np.max(counts))
    print(np.argmax(pcounts),np.max(pcounts))
    # print(df.iloc[127])
    # print(df.iloc[1616])

    # spix = 1687
    print("-- %s --"%spix)
    print(df.iloc[spix])

    spix = 869
    print("-- %s --"%spix)
    print(df.iloc[spix])

    exit()


def get_app_variances(dname,vname):
    findex = 0
    param_fn = "/home/gauenk/Documents/packages/st_spix_refactor/result_davis/bass/param0/%s/%05d_params.csv"%(vname,findex)
    fn = "/home/gauenk/Documents/packages/st_spix_refactor/result_davis/bass/param0/%s/%05d.csv"%(vname,findex)

    # -- read data --
    vid = read_video(dname,vname)
    df = pd.read_csv(param_fn)
    seg = pd.read_csv(fn,header=None).to_numpy()
    spix = np.argmax(np.bincount(seg[40:80,-10:].ravel())).item()

    # -- compute variance --
    vid = read_video(dname,vname)
    vid = rearrange(vid,'b h w c -> b c h w')
    vid = th.from_numpy(vid)
    # print(type(vid))
    img = vid[[findex]]
    # print(img.shape,img.max())
    img_lab = rgb2lab(img)
    B,C,H,W = img.shape
    # print(img_lab.shape,img_lab.max())
    outs = np.bincount(seg.ravel())
    # rand_spix = np.random.permutation(np.unique(seg.ravel()))[:10]
    rand_spix = np.unique(seg.ravel())
    app_vars = []
    for _spix in rand_spix:
        mask = th.from_numpy(seg == _spix)
        app_var = [img_lab[0,i][mask].var().item() for i in range(3)]
        if np.any(np.isnan(app_var)):
            print("NAN @ %d"%_spix)
            print(df.iloc[_spix])
            continue
        app_vars.append(np.array(app_var))
        # print("var[%d]: "%_spix,)
    app_vars = np.stack(app_vars).sum(-1)
    quants = np.quantile(app_vars,[0.1,0.5,0.75,0.9,0.95])
    return quants

def gather_app_var_bass():

    dname = "davis"
    names = get_video_names(dname)
    quants = []
    for vname in names[:10]:
        quants.append(get_app_variances(dname,vname))
    print([len(q) for q in quants])
    quants = np.stack(quants)
    print(quants.shape)
    print(np.mean(quants,0))
    print(np.std(quants,0))


def sample_gamma():
    samples = np.random.gamma(shape=1.01, scale=0.006, size=100000)
    print(np.mean(samples))
    quants = np.quantile(samples,[0.1,0.5,0.75,0.9,0.95])
    print(quants)

def main():
    # sample_gamma()
    # gather_app_var_bass()
    # inspect_app_var()
    # return

    # spix = check_nspix_across_time()

    dname = "davis"
    # pname = "param0"
    # vnames = ["bike-packing","blackswan","bmx-trees",
    #           "breakdance","camel","car-roundabout"]
    # vnames = ["bike-packing","bmx-trees"]
    # vnames = ["bmx-trees","car-roundabout"]
    # vnames = ["car-roundabout"]
    # vnames = ["bmx-trees"]
    # vnames = ["bike-packing"]
    sp_grid = [300,500,800,1000,1200]
    # sp_grid = [0]

    # vnames = vnames[2:]
    # vnames = ["bmx-trees","breakdance"]
    # vnames = ["bmx-trees"]
    # vnames = ["bmx-trees","car-roundabout"]
    # vnames = ["car-roundabout"]
    # vnames = ["bmx-trees"]
    # vnames = ["breakdance"]
    # vnames = ["bike-packing"]
    # dname = "davis"

    # dname = "segtrackerv2"
    # vnames = get_segtrackerv2_videos()
    # vnames = ["frog_2","girl"]
    # vnames = ["girl"]
    # vnames = ["frog_2"]

    # vnames = ["breakdance"]
    vnames = get_video_names(dname)
    # vnames = vnames[:10]
    # vnames = vnames[10:]
    # vnames = vnames[20:]
    # vnames = vnames[-5:]
    # vnames = [vnames[-1]]
    # vnames = ["worm_1"]

    for sp in sp_grid:
        nvalid = 0
        for vname in vnames:
            pname = "sp%d"%sp
            print("\n\n")
            print("-"*20 + " %s "%vname + "-"*20)
            print("\n\n")
            nvalid += short_eval("bass",dname,pname,vname,tgt_nspix=sp)
            # short_eval("bist",dname,pname,vname,tgt_nspix=sp)
        print("percent valid: ",nvalid/(1.*len(vnames)))


if __name__ == "__main__":
    main()

