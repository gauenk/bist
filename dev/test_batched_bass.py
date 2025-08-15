import numpy as np
import torch as th
import subprocess
import bist
from bist.api import default_params,extract,BIST_HOME
from pathlib import Path

def run(vid_root,flow_root,spix_root,img_ext,**kwargs):
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
    niters = kwargs['niters'] if 'niters' in kwargs else sp_size
    use_sm = kwargs['use_sm']
    overlap = kwargs['overlap']
    logging = kwargs['logging']
    nimgs = kwargs['nimgs']
    verbose = kwargs['verbose']
    save_only_spix = 1 if kwargs['save_only_spix'] else 0
    read_video = 1 if video_mode else 0
    batch_mode = kwargs['batch_mode']
    bist_bin = str(Path(BIST_HOME)/"bin/bist")


    # -- ensure output exists --
    if not Path(spix_root).exists():
        Path(spix_root).mkdir(parents=True, exist_ok=True)

    # -- ensure strings --
    vid_root,flow_root,spix_root = str(vid_root),str(flow_root),str(spix_root)

    # -- prepare command --
    cmd = "%s -n %d -d %s/ -f %s/ -o %s/ --read_video %d --img_ext %s --sigma_app %2.5f --potts %2.2f --alpha %2.3f --split_alpha %2.3f --tgt_nspix %d --gamma %2.2f --epsilon_reid %1.8f --epsilon_new %1.8f --prop_nc %d --prop_icov %d --niters %d --use_sm %d --overlap %d --logging %d --nimgs %d --save_only_spix %d --batch_mode %d" % (bist_bin,sp_size,vid_root,flow_root,spix_root,read_video,img_ext,sigma_app,potts,alpha,split_alpha,tgt_nspix,gamma,epsilon_reid,epsilon_new,prop_nc,prop_icov,niters,use_sm,overlap,logging,nimgs,save_only_spix,batch_mode)

    # -- run binary --
    print(cmd)
    if verbose:
        print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    return output

def relabel_spix(spix_b, spix_s):
    """
    Relabel the superpixels in spix_b to match the order of spix_s.
    This is a simple relabeling based on the first appearance of each label.
    """
    spix_b = spix_b.clone()
    for ti in range(spix_b.shape[0]):
        src = spix_b[ti].ravel()
        ref = spix_s[ti].ravel()

        vals, inv = th.unique(src, return_inverse=True, sorted=False)
        _vals, _inv = th.unique(ref, return_inverse=True, sorted=False)

        pos = th.arange(src.numel(), device=src.device)
        first_idx = th.full((vals.numel(),), src.numel(), device=src.device)
        first_idx = first_idx.scatter_reduce(0, inv, pos, reduce='amin', include_self=True)

        xfer = ref[first_idx]
        order = th.argsort(vals)
        xfer = xfer[order]
        # print(xfer.shape,xfer.min(),xfer.max(),spix_b.min(),spix_b.max())
        # exit()

        spix_b[ti] = xfer[spix_b[ti].long()].reshape(spix_b[ti].shape)

    return spix_b

def main_cpp():

    # -- setup --
    vname = "kid-football"
    vpath = "data/examples/%s/imgs"%vname
    fpath = "data/examples/%s/flows"%vname
    vid = bist.utils.read_video("data/examples/%s/imgs"%vname).cuda()/255.
    cfg = {"niters":20,"video_mode":False,"sp_size":10,"potts":10.0,"save_only_spix":False,"use_sm":False}
    sroot = Path("output/test_batched_bass/")
    if not sroot.exists():
        sroot.mkdir(parents=True, exist_ok=True)

    # -- run both modes --
    cfg['batch_mode'] = True
    run(vpath,fpath,str(sroot/"batched"),"jpg",**cfg)
    cfg['batch_mode'] = False
    run(vpath,fpath,str(sroot/"standard"),"jpg",**cfg)

    # -- read and compare --
    spix_b = bist.utils.read_spix(sroot/"batched")*1.0
    spix_s = bist.utils.read_spix(sroot/"standard")*1.0
    print(spix_b.shape, spix_s.shape)

    # -- dumb relabeling --
    spix_r = relabel_spix(spix_b, spix_s)

    # -- direct compare --
    delta = th.mean(th.abs(spix_r - spix_s))
    print("Direct compare: ", delta)


    print(spix_r[0,:5,:5])
    print(spix_s[0,:5,:5])

    print(spix_r[0,-5:,-5:])
    print(spix_s[0,-5:,-5:])
    spix_b = bist.utils.read_spix(sroot/"batched")*1.0


    # -- compare --
    for ti in range(spix_b.shape[0]):
        b_t = spix_b[ti]
        s_t = spix_s[ti]
        s_u = th.unique(s_t)
        b_u = th.unique(b_t)
        print(len(b_u), len(s_u), th.max(b_u), th.max(s_u))
        if not(len(s_u) == th.max(s_u) + 1):
            print("Standard spix not compactified")
        if not(len(b_u) == th.max(b_u) + 1):
            print("Batched spix not compactified")


def main_py():


    # -- setup --
    vname = "kid-football"
    vpath = "data/examples/%s/imgs"%vname
    fpath = "data/examples/%s/flows"%vname
    vid = bist.utils.read_video("data/examples/%s/imgs"%vname)[:,:,:].cuda()/255.
    fflows = th.zeros_like(vid[...,:2])
    cfg = {"niters":20,"video_mode":False,"sp_size":10,"potts":10.0,"use_sm":False}
    sroot = Path("output/test_batched_bass/")
    if not sroot.exists():
        sroot.mkdir(parents=True, exist_ok=True)


    # -- run both modes --
    spix_s = bist.run(vid,fflows,**cfg)*1.
    spix_b = bist.run_batched(vid,**cfg)*1.
    spix_b_og = spix_b.clone()
    print(spix_b.shape)

    # -- dumb relabeling --
    spix_r = relabel_spix(spix_b, spix_s)

    # -- direct compare --
    for i in range(spix_r.shape[0]):
        delta = th.mean(th.abs(spix_r[i] - spix_s[i]))
        print("Direct compare: ", delta)
    return


    print(spix_r[1,:5,:5])
    print(spix_s[1,:5,:5])
    print(spix_b[0])
    print("-"*20)
    print("-"*20)
    print(spix_s[1])
    print(spix_b[1])
    print("-"*20)
    print("-"*20)
    print(spix_s[2])
    print(spix_b[2])
    print("-"*20)
    print("-"*20)

    print(spix_r[1,-5:,-5:])
    print(spix_s[1,-5:,-5:])


    # -- compare --
    spix_b = spix_b_og.clone()*1.
    for ti in range(spix_b.shape[0]):
        b_t = spix_b[ti]
        s_t = spix_s[ti]
        s_u = th.unique(s_t)
        b_u = th.unique(b_t)
        print(len(b_u), len(s_u), th.max(b_u), th.max(s_u))
        if not(len(s_u) == th.max(s_u) + 1):
            print("Standard spix not compactified")
        if not(len(b_u) == th.max(b_u) + 1):
            print("Batched spix not compactified")

if __name__ == "__main__":
    main_cpp()
    # main_py()
