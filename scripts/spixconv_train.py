""""

     Train the superpixel convolution models

"""



import bist

import math
import time

import torch as th
import pandas as pd
from pathlib import Path

def sigma_grid_exps(default):
    config = {"conv_type":["rw_conv"],
              "sigma":[10,20,30],
              "sconv_reweight_source":["sims"],
              "spix_method":["bist","bass","exh"],
              "group":["sigma_grid"]}
    exps = bist.utils.get_exps(config,default)
    config = {"conv_type":["conv"],
              "sigma":[10,20,30],
              "group":["sigma_grid"]}
    exps += bist.utils.get_exps(config,default)
    config = {"conv_type":["sconv"],
              "sigma":[10,20,30],
              "sconv_reweight_source":["ftrs"],
              "group":["sigma_grid"]}
    exps += bist.utils.get_exps(config,default)

    return exps

def sconv_norm_exps(default):
    config = {"sconv_norm_type":["sm","exp_max"],
              "sconv_norm_scale":[1.0,5.0,10.0,20.],
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
    sigma = exp['sigma']
    patch_size = 96
    tr_nframes = 3
    num_workers = 8
    nepochs = 30
    lr = 1.0e-3
    seed = 123
    nbatches_per_epoch = 1000
    log_every = 100

    # -- load network --
    model = bist.spixconv.model.SpixConvDenoiser(**exp).cuda()
    print(model)

    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    data,loader = bist.spixconv.davis.get_davis_dataset(tr_nframes,num_workers)
    flow_fxn = bist.spixconv.spynet.get_flow_fxn()
    print(loader.tr)
    def add_noise(video):
        return video + (sigma/255.)*th.randn_like(video)

    # -- training loop --
    timer_start = time.time()
    for epoch in range(nepochs):
        loss,psnr = 0.0,0.0
        model = model.train()
        data_iter = iter(loader.tr)
        th.manual_seed(int(seed)+epoch)
        for idx in range(nbatches_per_epoch):

            # -- reset --
            optimizer.zero_grad()

            # -- read video/annotation --
            video,anno = next(data_iter)
            video,anno = video[0].cuda(),anno[0].cuda()

            # -- random crop --
            video,anno = bist.spixconv.davis.random_crop(video,anno,patch_size)
            noisy = add_noise(video)

            # -- compute flows --
            fflow,_ = flow_fxn(noisy)

            # -- forward --
            spix,spix_ftrs = model.get_spix(video,fflow)
            spix = spix.detach()
            deno = []
            for t in range(spix.shape[0]):
                deno.append(model(noisy[[t]],spix[[t]],spix_ftrs[[t]]))
            deno = th.cat(deno)

            # -- loss --
            deno_loss = th.sqrt(th.mean((deno-video)**2)+1e-6)
            deno_loss.backward()

            # -- track training --
            loss += float(deno_loss)
            psnr += bist.metrics.compute_psnrs(video,deno,div=1.).mean().item()

            # -- update --
            optimizer.step()

            # -- loss logging --
            if (idx + 1) % log_every == 0:
                total_steps = nbatches_per_epoch
                fill_width = math.ceil(math.log10(nbatches_per_epoch))
                cur_steps = str(idx + 1).zfill(fill_width)
                epoch_width = math.ceil(math.log10(nepochs))
                cur_epoch = str(epoch).zfill(epoch_width)
                avg_loss = loss / (idx + 1)
                avg_psnr = psnr / (idx + 1)
                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print('Epoch:{}, {}/{}, loss: {:.4f}, psnr: {:.2f}, time: {:.3f}'.\
                      format(cur_epoch, cur_steps, nbatches_per_epoch, avg_loss, avg_psnr, duration))




def main():

    # -- global config --
    default = {
        "sigma":20,"dim":6,
        "conv_type":"sconv","spix_method":"bass",
        "sims_norm_scale":10.0,
        "use_spixftrs_net":False,"spixftrs_dim":3,
        "sconv_norm_type":"exp_max","sconv_norm_scale":1.0,
        "kernel_norm_type":"none","kernel_norm_scale":0.0,
        "conv_kernel_size":3,"sconv_kernel_size":7,"net_depth":3,
        "sp_size":15,"potts":10.0,"sigma_app":0.09,
        "alpha":math.log(0.5),"iperc_coeff":4.0,
        "thresh_new":5e-2,"thresh_relabel":1e-6}

    # -- save root --
    save_root = Path("results/spixconv/")
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- collect experiment grids --
    exps = sigma_grid_exps(default)
    exps += sconv_norm_exps(default)
    exps += kernel_norm_exps(default)
    exps += architecture_exps(default)
    exps += spix_params_exps(default)

    # -- run each experiment a few times --
    nreps = 1
    for exp in exps:
        for rep in range(nreps):
            exp_root = save_root/exp['group']/exp['id']/("rep%d"%rep)
            # print(exp_root)
            # save_exp_info(exp,save_root / "info")
            run_exp(exp,save_root)


if __name__ == "__main__":
    main()
