""""

     Test the superpixel convolution models

"""



import bist

import os
import math
import time

import torch as th
from einops import rearrange
import numpy as np
import pandas as pd
from pathlib import Path

def evaluate_exp(exp,save_root,eval_root,refresh=False):

    # -- make save directory --
    assert save_root.exists()
    if not eval_root.exists():
        eval_root.mkdir(parents=True)
    cache_root = eval_root / "cache"

    # -- logging --
    print("\n"*5)
    print("Testing Experiment Config: ")
    print(exp)

    # -- config --
    task = "deno"
    exp['task'] = task
    sigma = exp['sigma']
    num_workers = 0
    seed = 123
    bist.utils.seed_everything(seed)
    if num_workers > 0: th.set_num_threads(num_workers)

    # -- load network --
    model = bist.spixconv.model.SpixConvNetwork(**exp).cuda()
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
    results = []
    # for dindex in range(len(data.te)):
    for dindex in range(2):

        # -- get video name --
        vname = data.te.groups[dindex]

        # -- read cache --
        res = bist.utils.read_cache(cache_root,"davis",
                                    vname,exp['group'],exp['id'])
        if not(res is None) and not(refresh):
            results.append(res)
            continue

        # -- read video/annotation --
        video,anno,_ = data.te[dindex]
        # video,anno = video[:10],anno[:10]
        video,anno = video[:2],anno[:2]
        video,anno = video.cuda(),anno.cuda()

        # -- get noisy --
        if task == "deno":
            noisy = add_noise(video)
        else:
            noisy = video

        # -- compute flows --
        fflow,_ = flow_fxn(noisy)

        # -- forward --
        with th.no_grad():
            spix = model.get_spix(noisy,fflow)
            pred = model.crop_forward(10,128,0.1,noisy,spix)

        # -- eval --
        tpos,tneg,ppos = -1,-1,-1
        psnrs,ssims,ave_psnr,ave_ssim = [],[],-1.0,-1.0
        if task == "deno":
            psnrs = bist.metrics.compute_psnrs(video,pred,div=1.)
            ssims = bist.metrics.compute_ssims(video,pred,div=1.)
            ave_psnr,ave_ssim = psnrs.mean().item(),ssims.mean().item()
        else:
            pred = th.sigmoid(pred)
            tpos,tneg,ppos = bist.metrics.seg_metrics(pred,anno)

        # -- log sample --
        print("[%s]: %2.3f %0.3f" %(vname,ave_psnr,ave_ssim))
        res = {}
        res['vname'] = vname
        res['psnrs'] = np_pad(psnrs)
        res['ssims'] = np_pad(ssims)
        res['ave_psnr'] = ave_psnr
        res['ave_ssim'] = ave_ssim
        res['tpos'] = tpos
        res['tneg'] = tneg
        res['ppos'] = ppos

        # -- save cache --
        for k,v in exp.items(): res[k] = v
        results.append(pd.DataFrame([res]))
        bist.utils.save_cache([res],cache_root,"davis",
                              vname,exp['group'],exp['id'])

    # -- save --
    results = pd.concat(results)
    return results

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
    refresh = True

    # -- evaluate over a grid --
    eval_root = save_root / "eval"
    nreps = 10
    results = []
    for exp in exps:
        if exp['sigma'] != 30: continue
        # if exp['sigma'] != 50: continue
        if exp['sconv_reweight_source'] == "sims+ftrs": continue
        # if exp['sconv_reweight_source'] != "ftrs": continue
        if exp['conv_type'] != "sconv": continue
        # if exp['sconv_reweight_source'] != "sims": continue
        for rep in range(nreps):
            spix_root = save_root/exp['group']/exp['id']/("rep%d"%rep)
            if not spix_root.exists(): continue
            # exp['sconv_reweight_source'] = "ftrs"
            # exp['sconv_norm_scale'] = 10.0
            # exp['sconv_norm_type'] = "exp_max"
            res = evaluate_exp(exp,spix_root,eval_root,refresh)
            res['rep'] = rep
            results.append(res)
    results = pd.concat(results)
    results = results.rename(columns={"sconv_reweight_source":"scs"})
    print(results[['scs','vname','ave_psnr','ave_ssim']])

    results = results.drop(columns=['vname',"psnrs","ssims"])
    results = results.groupby("scs").mean(["ave_psnrs","ave_ssims"]).reset_index()
    print(results[['scs','ave_psnr','ave_ssim']])

    # -- view results --
    # view_sigma_grid(results)



if __name__ == "__main__":
    main()


