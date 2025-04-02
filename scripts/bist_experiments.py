"""

  BIST Experiments

"""


import bist
import math
import pandas as pd
from pathlib import Path

def bist_exps(default):
    config = {"video_mode":[True],"group":["bist"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def bass_exps(default):
    config = {"video_mode":[False],"group":["bass"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def split_step_exps(default):
    config = {"iperc_coeff":[0.0,4.0,8.0],
              "alpha":[math.log(0.1),math.log(0.5),0.0,math.log(2.0)],
              "group":["split"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def relabeling_exps(default):
    config = {"thresh_new":[1.0e-1,5.0e-2,1e-2,1e-3],
              "thresh_relabel":[1e-6,1e-5,1e-4],"group":["relabel"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def boundary_shape_exps(default):
    config = {"sigma_app":[0.0045,0.009,0.018],"potts":[1.0,10.,20.],"group":["bshape"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def conditioned_boundary_updates_exps(default):
    config = {"prop_nc":[0,1],"prop_icov":[0,1],"group":["cboundary"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def optical_flow_exps(default):
    config = {"flow":["default","raft","spynet"],"group":["flow"]}
    exps = bist.utils.get_exps(config,default)
    return exps

def save_exp_info(exp,save_root):
    if not save_root.exists():
        save_root.mkdir(parents=True)
    fname = "%s_%s.csv" % (exp['group'],exp['id'])
    exp = {k:[v] for k,v in exp.items()}
    pd.DataFrame(exp).to_csv(save_root / fname,index=False)

def run_exp(dname,exp,spix_root):

    if not spix_root.exists():
        spix_root.mkdir(parents=True)

    # -- unpack directories --
    vnames = bist.utils.get_video_names(dname)
    image_root = bist.utils.get_image_root(dname)
    img_ext = bist.utils.get_dataset_ext(dname)

    # -- select optical flow --
    if exp['flow'] == "default":
        flow_subdir = "BIST_flows"
    elif exp['flow'] == "raft":
        flow_subdir = "RAFT_flows"
    elif exp['flow'] == "spynet":
        flow_subdir = "SPYNET_flows"
    else:
        raise ValueError("Uknown flow method [%s]"%exp['flow'])

    # -- run bist --
    for vname in vnames:
        vid_root = image_root/vname
        flow_root = vid_root/flow_subdir
        spix_root_v = spix_root / vname
        print("Running experiment: ",vname)
        bist.run_bin(vid_root,flow_root,spix_root_v,img_ext,**exp)

def main():

    # -- global config --
    dname = "davis"
    default = {"sp_size":25,"potts":10.0,"sigma_app":0.009,
               "alpha":math.log(0.5),"iperc_coeff":4.0,
               "thresh_new":5e-2,"thresh_relabel":1e-6,
               "video_mode":True,"flow":"default","group":"bist","nimgs":0}

    # -- save root --
    save_root = Path("results/")/dname
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- collect experiment grids --
    exps = bass_exps(default)
    exps += bist_exps(default)
    exps += split_step_exps(default)
    exps += relabeling_exps(default)
    exps += boundary_shape_exps(default)

    # -- run each experiment a few times --
    nreps = 1
    for exp in exps:
        for rep in range(nreps):
            spix_root = save_root / exp['group'] / exp['id'] / ("rep%d"%rep)
            save_exp_info(exp,save_root / "info")
            run_exp(dname,exp,spix_root)

    # -- run [conditional boundary] several times --
    exps = conditioned_boundary_updates_exps(default)
    nreps = 10
    for exp in exps:
        for rep in range(nreps):
            spix_root = save_root / exp['group'] / exp['id'] / ("rep%d"%rep)
            save_exp_info(exp,save_root / "info")
            run_exp(dname,exp,spix_root)

    # -- run [optical flow] on segtrackv2 --
    dname = "segtrackv22"
    save_root = Path("results/")/dname
    if not save_root.exists():
        save_root.mkdir(parents=True)
    exps = optical_flow_exps(default) # only on segtrack
    nreps = 1
    for exp in exps:
        for rep in range(nreps):
            spix_root = save_root / exp['group'] / exp['id'] / ("rep%d"%rep)
            save_exp_info(exp,save_root / "info")
            run_exp(dname,exp,spix_root)


if __name__ == "__main__":
    main()
