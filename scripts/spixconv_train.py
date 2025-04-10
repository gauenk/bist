""""

     Train the superpixel convolution models

"""



import bist

import os
import math
import time

import numpy as np
import torch as th
import pandas as pd
from pathlib import Path

def sigma_grid_exps(default):
    config = {"conv_type":["sconv"],
              # "sigma":[10,20,30],
              "sigma":[50,30,70,10],
              "sconv_reweight_source":["sims"],
              # "spix_method":["bist","bass","exh"],
              "sconv_norm_type":["max"],
              "sconv_norm_scale":[0.0],
              "spix_method":["bist"],
              "group":["sigma_grid_v1"]}
    exps = bist.utils.get_exps(config,default)[1:2]
    # exps = bist.utils.get_exps(config,default)[2:3]
    # exps = bist.utils.get_exps(config,default)[3:]
    # exps = bist.utils.get_exps(config,default)[:1]
    # exps = bist.utils.get_exps(config,default)[3:4]

    config = {"conv_type":["sconv"],
              "sigma":[50,30,70,10],
              # "sigma":[10,20,30],
              "sconv_reweight_source":["pftrs"],
              "sconv_norm_type":["exp_max"],
              "sconv_norm_scale":[10.0],
              "group":["sigma_grid_v4"]}
    # exps = bist.utils.get_exps(config,default)[1:2]

    config = {"conv_type":["sconv"],
              "sigma":[50,30,70,10],
              # "sigma":[10,20,30],
              "sconv_reweight_source":["ftrs"],
              "sconv_norm_type":["exp_max"],
              "sconv_norm_scale":[10.0],
              "group":["sigma_grid"]}
    # exps = bist.utils.get_exps(config,default)[:1]
    # exps = bist.utils.get_exps(config,default)[1:2]
    # exps = bist.utils.get_exps(config,default)[2:3]
    # exps = bist.utils.get_exps(config,default)[3:]
    # exps = bist.utils.get_exps(config,default)[3:4]
    # exps += bist.utils.get_exps(config,default)

    config = {"conv_type":["sconv"],
              # "sigma":[10,20,30],
              "sigma":[50,30,70,10],
              "sconv_reweight_source":["sims+ftrs"],
              # "spix_method":["bist","bass","exh"],
              "spix_method":["bist"],
              "sconv_norm_scale":[10.0],
              "group":["sigma_grid_v2"]}
    # exps = bist.utils.get_exps(config,default)[:1]
    # exps = bist.utils.get_exps(config,default)[1:2]
    # exps = bist.utils.get_exps(config,default)[3:4]

    config = {"conv_type":["conv"],
              "sigma":[10,20,30],
              # "sigma":[10],
              "group":["sigma_grid_v3"]}
    # exps = bist.utils.get_exps(config,default)[:1]
    # exps = bist.utils.get_exps(config,default)[1:2]
    # exps = bist.utils.get_exps(config,default)[2:3]

    return exps

def sconv_norm_exps(default):
    config = {"sconv_reweight_source":["ftrs"],
              "sconv_norm_type":["exp_max","sm"],
              "sconv_norm_scale":[1.0,10.0,20.0],
              "group":["sconv_norm"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def kernel_norm_exps(default):
    config = {"kernel_norm_type":["none","sum"],
              "kernel_norm_scale":[0.0],
              "group":["kernel_norm"]}
    exps = bist.utils.get_exps(config,default)
    config = {"kernel_norm_type":["sm"],
              "kernel_norm_scale":[1.0,5.0,10.0],
              "group":["kernel_norm"]}
    exps += bist.utils.get_exps(config,default)
    config = {"kernel_norm_type":["exp_max"],
              "kernel_norm_scale":[1.0,5.0,10.0],
              "group":["kernel_norm"]}
    exps += bist.utils.get_exps(config,default)
    return exps

def architecture_exps(default):
    config = {"conv_kernel_size":[3,7],
              "sconv_kernel_size":[3,7,9],
              "network_depth":[3,5],
              "group":["architecture"]}
    exps = bist.utils.get_exps(config,default)

    # config = {"spixconv_ftrs":[True,False],
    #           "conv_type":["sconv","conv"],
    #           "group":["architecture"]}
    # exps += bist.utils.get_exps(config,default)

    # -- input features for superpixel config --
    config = {"use_spixftrs_net":[True],
              "spixftrs_dim":[3,6,9],"group":["architecture"]}
    exps += bist.utils.get_exps(config,default)
    config = {"use_spixftrs_net":[False],
              "spixftrs_dim":[0],"group":["architecture"]}
    exps += bist.utils.get_exps(config,default)
    return exps

def spix_params_exps(default):
    config = {"sims_norm_scale":[5.0,10.,20.],
              "sigma_app":[0.045,0.09,0.018],
              "potts":[1.0,10.,20.],"group":["spix_params"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def save_exp_info(exp,save_root):
    if not save_root.exists():
        save_root.mkdir(parents=True)
    fname = "%s_%s.csv" % (exp['group'],exp['id'])
    exp = {k:[v] for k,v in exp.items()}
    pd.DataFrame(exp).to_csv(save_root / fname,index=False)

def run_exp(exp,save_root):

    # -- make save directory --
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- logging --
    print("\n"*5)
    print("Training Experiment Config: ")
    print(exp)

    # -- config --
    # task = exp['task']
    task = "deno"
    exp['task'] = task
    sigma = exp['sigma']
    patch_size = 128
    tr_nframes = 3
    num_workers = 1
    nepochs = 30
    # lr = 1.0e-3
    lr = 1.0e-4
    seed = 123
    nbatches_per_epoch = 1000
    log_every = 100
    bist.utils.seed_everything(seed)
    th.set_num_threads(num_workers)
    shrink = task == "seg"

    # -- load network --
    model = bist.spixconv.model.SpixConvNetwork(**exp).cuda()
    model = model.train()
    nparams = sum(p.numel() for p in model.parameters())
    _ = th.randn(1).cuda()
    print(model)
    print("Number of Parameters: ",nparams)

    # -- load train stuff --
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    data = bist.spixconv.davis.get_davis_dataset(tr_nframes,shrink)
    get_loaders = bist.spixconv.davis.get_loaders
    flow_fxn = bist.spixconv.spynet.get_flow_fxn()
    def add_noise(video):
        return video + (sigma/255.)*th.randn_like(video)

    # -- training loop --
    timer_start = time.time()
    for epoch in range(nepochs):
        loss,nspix = 0.0,0.0
        psnr = 0.0 # deno metrics
        tpos,tneg,perc_pos = 0.0,0.0,0.0 # seg metrics
        means,stds = np.zeros(3),np.zeros(3)
        model = model.train()

        # -- get data loader --
        loader = get_loaders(data,num_workers,seed+epoch).tr
        _ = next(iter(loader)) # init with repro
        data_iter = iter(loader)

        for idx in range(nbatches_per_epoch):

            # -- reset --
            optimizer.zero_grad()

            # -- read video/annotation --
            video,anno,dindex = next(data_iter)
            if task == "seg":
                while anno.mean() < 0.10:
                    video,anno,dindex = next(data_iter)

            # -- random crop --
            get_random_crop = bist.spixconv.davis.get_random_crop
            video,anno = get_random_crop(video,anno,patch_size,task)
            video,anno = video[0].cuda(),anno[0].cuda()

            # -- relabel --
            if task == "seg": noisy = video-0.5
            elif task == "deno": noisy = add_noise(video)
            else: raise ValueError(f"Unknown task [{task}]")

            # -- compute flows --
            fflow,_ = flow_fxn(noisy)

            # -- forward --
            spix = model.get_spix(noisy,fflow,rgb2lab=True)
            output = model(noisy,spix.detach())

            # -- loss --
            if task == "deno":
                batch_loss = th.sqrt(th.mean((output-video)**2)+1e-6)
                batch_loss.backward()
            elif task == "seg":
                criterion = th.nn.BCEWithLogitsLoss()
                batch_loss = criterion(pred,anno)
                batch_loss.backward()

            # -- update --
            optimizer.step()

            # -- tracking --
            loss += float(batch_loss)
            nspix += 1.*len(th.unique(spix))
            if task == "deno":
                psnr += bist.metrics.compute_psnrs(output,video).mean()
            elif task == "seg":
                pred = th.sigmoid(pred)
                _tpos,_tneg,_perc_pos = bist.metrics.seg_metrics(pred,anno)
                tpos += _tpos
                tneg += _tneg
                prec_pos += _perc_pos

            # -- view stats --
            _means,_stds = model.get_rweight_stats()
            means += _means
            stds += _stds

            # -- loss logging --
            if (idx + 1) % log_every == 0:

                total_steps = nbatches_per_epoch
                fill_width = math.ceil(math.log10(nbatches_per_epoch))
                cur_steps = str(idx + 1).zfill(fill_width)
                epoch_width = math.ceil(math.log10(nepochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = loss / (idx + 1)
                avg_nspix = int(nspix // (idx + 1))
                avg_psnr = psnr / (idx + 1)
                avg_tpos = tpos / (idx + 1)
                avg_tneg = tneg / (idx + 1)
                avg_ppos = perc_pos / (idx + 1)
                ave_means = means/(idx+1)
                ave_stds = stds/(idx+1)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                if task == "seg":
                    print('Epoch:{}, {}/{}, loss: {:.4f}, nsp: {:d}, tp: {:.3f}, tn: {:.3f}, pp: {:.3f} time: {:.2f}'.format(cur_epoch, cur_steps, nbatches_per_epoch, avg_loss, avg_nspix, avg_tpos, avg_tneg, avg_ppos, duration))
                elif task == "deno":
                    print('Epoch:{}, {}/{}, loss: {:.4f}, nsp: {:d}, psnr: {:.2f}, time: {:.3f}'.format(cur_epoch, cur_steps, nbatches_per_epoch, avg_loss, avg_nspix, avg_psnr, duration))
                print("[Reweights] ",ave_means,ave_stds)

                # -- save --
                if (epoch+1) % 12 == 0 and ((idx+1) == log_every):
                    fname = save_root / ("checkpoint_%d.ckpt"%(epoch+1))
                    print("Saving checkpoint: ",fname)
                    state = {'model_state_dict': model.state_dict()}
                    th.save(state, fname)

    # -- save --
    fname = save_root / "checkpoint.ckpt"
    print("Writing model to ",str(fname))
    th.save({'model_state_dict': model.state_dict()}, fname)

def main():
    print("PID: ",os.getpid())

    # -- global config --
    default = {
        "sigma":10,"dim":6,
        "conv_type":"sconv","spix_method":"bist",
        "sims_norm_scale":30.0,"sconv_reweight_source":"sims",
        "use_spixftrs_net":False,"spixftrs_dim":0,
        "sconv_norm_type":"max","sconv_norm_scale":0.0,
        "kernel_norm_type":"none","kernel_norm_scale":0.0,
        "conv_kernel_size":3,"sconv_kernel_size":7,"net_depth":3,
        # -- spix params --
        # "sp_size":15,"sigma2_app":0.09,"potts":1.,
        # "alpha":2.0,"split_alpha":2.0,"sm_start":0,"rgb2lab":False}
        # -- spix params --
        # "sp_size":15,"potts":1.0,"sigma_app":0.009,
        # "alpha":math.log(0.5),"iperc_coeff":4.0,
        # "thresh_new":5e-2,"thresh_relabel":1e-6,"rgb2lab":True}
        # -- spix params --
        "sp_size":6,"potts":10.0,"sigma_app":0.008,
        "alpha":math.log(2),"split_alpha":0.0,"iperc_coeff":4.0,
        "thresh_new":5e-2,"thresh_relabel":1e-6,"rgb2lab":True}

    # -- save root --
    save_root = Path("results/spixconv/")
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- collect experiment grids --
    exps = sigma_grid_exps(default)[:1]
    # exps = sigma_grid_exps(default)[1:]
    # exps += sconv_norm_exps(default)
    # exps = sconv_norm_exps(default)
    # exps += kernel_norm_exps(default)
    # exps += architecture_exps(default)
    # exps += spix_params_exps(default)

    # -- run each experiment a few times --
    nreps = 1
    for exp in exps:
        for rep in range(nreps):
            exp_root = save_root/exp['group']/exp['id']/("rep%d"%rep)
            # print(exp_root)
            save_exp_info(exp,save_root / "info")
            run_exp(exp,exp_root)


if __name__ == "__main__":
    main()
