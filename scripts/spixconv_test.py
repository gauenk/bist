""""

     Test the superpixel convolution models

"""



import bist

import os
import math
import time

import torch as th
import numpy as np
import pandas as pd
from pathlib import Path

def evaluate_exp(exp,save_root,eval_root):

    # -- make save directory --
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- logging --
    print("\n"*5)
    print("Testing Experiment Config: ")
    print(exp)

    # -- config --
    sigma = exp['sigma']
    num_workers = 0
    seed = 123
    bist.utils.seed_everything(seed)
    if num_workers > 0: th.set_num_threads(num_workers)

    # -- load network --
    model = bist.spixconv.model.SpixConvDenoiser(**exp).cuda()
    model = model.train()
    nparams = sum(p.numel() for p in model.parameters())
    _ = th.randn(1).cuda()
    print(model)
    print("Number of Parameters: ",nparams)

    # -- load checkpoint --
    fname = save_root / "checkpoint.ckpt"
    state = th.load(fname,weights_only=False)['model_state_dict']
    model.load_state_dict(state)

    # -- load testing stuff --
    data = bist.spixconv.davis.get_davis_dataset()
    flow_fxn = bist.spixconv.spynet.get_flow_fxn()
    def add_noise(video):
        return video + (sigma/255.)*th.randn_like(video)
    def np_pad(ndarray):
        return np.pad(ndarray,(0,130),
                      mode='constant',constant_values=-1)
    # -- testing loop --
    timer_start = time.time()
    info = {"psnrs":[],"ssims":[]}

    for dindex in range(len(data.te)):

        # -- read video/annotation --
        vname = data.te.groups[dindex]
        video,anno,_ = data.te[dindex]
        video,anno = video.cuda(),anno.cuda()
        noisy = add_noise(video)

        # -- compute flows --
        fflow,_ = flow_fxn(noisy)

        # -- forward --
        spix = model.get_spix(noisy,fflow)
        spix = spix.detach()
        with th.no_grad():
            deno = model.crop_forward(noisy,spix,10,128,0.1)

        # -- eval --
        psnrs = bist.metrics.compute_psnrs(video,deno,div=1.)
        ssims = bist.metrics.compute_ssims(video,deno,div=1.)
        psnr_ave,ssim_ave = psnrs.mean().item(),ssims.mean().item()

        # -- log --
        print("[%s]: %2.3f %0.3f" %(vname,psnr_ave,ssim_ave))
        info['psnrs'].append(np_pad(psnrs))
        info['ssims'].append(np_pad(ssims))
        return

    # -- save --
    fname = eval_root / "results.ckpt"
    print("Saving results to ",str(fname))
    pd.DataFrame(fname).to_csv(fname)

def read_exps(save_root):
    exps = []
    for fname in save_root.iterdir():
        if not str(fname).endswith(".csv"): continue
        exp = pd.read_csv(fname).to_dict(orient='records')[0]
        exps.append(exp)
    return exps

def main():

    print("PID: ",os.getpid())
    # -- save root --
    save_root = Path("results/spixconv/")
    if not save_root.exists():
        save_root.mkdir(parents=True)
    exps = read_exps(save_root / "info")

    # -- evaluate over a grid --
    eval_root = save_root / "eval"
    nreps = 10
    results = []
    for exp in exps:
        for rep in range(nreps):
            spix_root = save_root/exp['group']/exp['id']/("rep%d"%rep)
            if not spix_root.exists(): continue
            res = evaluate_exp(exp,spix_root,eval_root)
            # res['rep'] = rep
            # results.append(res)
    # results = pd.concat(results)

    # -- view results --
    # view_sigma_grid(results)



if __name__ == "__main__":
    main()


