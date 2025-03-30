
import os
import tqdm
import re,math
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import copy
dcopy = copy.deepcopy
from collections import OrderedDict as odict
from einops import rearrange
from mesh import mesh

def get_video_names(dname):
    if "segtrack" in dname.lower():
        return get_segtrackerv2_videos()
    elif "davis" in dname.lower():
        return get_davis_videos()
    else:
        raise KeyError(f"Uknown dataset name [{dname}]")

def get_segtrackerv2_videos():
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/GroundTruth/")
    vnames = list([v.name for v in root.iterdir()])
    # vnames = ["frog_2","girl"]
    # vnames = ["worm_1"]
    # vnames = ["penguin","frog_2","cheetah"]
    # vnames = ["penguin","frog_2","cheetah"]
    # vnames = ["penguin"]
    # return vnames[6:]
    # return vnames[:5]
    # vnames = ["hummingbird"]
    # return ["dogs-jump"]
    return vnames
    # return vnames[:10]
    # return vnames[:8]
    # return vnames[:3]


def get_davis_videos():
    try:
        # vnames = np.loadtxt("/app/in/DAVIS/ImageSets/2017/train-val.txt",dtype=str)
        vnames = np.loadtxt("/app/in/DAVIS/ImageSets/2017/val.txt",dtype=str)
    except:
        # fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
        fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/val.txt"
        vnames = np.loadtxt(fn,dtype=str)
    # vnames = ["kid-football"]
    # vnames = vnames[:10]
    # vnames = vnames[:2]
    # vnames = vnames[10:]
    # vnames = vnames[20:]
    # vnames = vnames[-4:]
    # vnames = vnames[-2:]
    # vnames = ["bmx-trees","breakdance"]
    # vnames = ["bmx-trees"]
    # vnames = vnames[:4]
    # vnames = ["bike-packing","blackswan","bmx-trees",
    #          "breakdance","camel","car-roundabout"]
    # vnames = ["bmx-trees","car-roundabout"]
    # vnames = ["bike-packing","bmx-trees"]
    # vnames = ["car-roundabout"]
    # vnames = ["soapbox"]
    # vnames = ["bmx-trees"]
    # vnames = ["breakdance"]
    # print(vnames)
    # exit()
    # vnames = vnames[2:]
    # return vnames[:3]
    # vnames = ["kite-surf","horsejump-high"]
    return vnames
    # return vnames[:8]
    # return vnames[:5]
    # return ["dogs-jump","blackswan"]
    # return ["dogs-jump"]
    # return ["bmx-trees"]


def get_flow_strings(flow_name):
    if flow_name == "raft":
        return "RAFT_flows",0
    elif flow_name == "spynet":
        return "SPYNET_flows",1
    elif flow_name == "ftsp":
        return "BIST_flows",2
    elif flow_name == "default":
        return "BIST_flows",2
    else:
        raise KeyError(f"Uknown flow name [{flow_name}]")

def mangle_method(method,flow_name):
    if flow_name == "default":
        return method
    else:
        return method+"_"+flow_name
def mangle_result_dir(dname):
    # print("RUNNING IN ABLATION MODE!")
    # ablate_names = ["only_app","only_shape","no_relabel","all"]
    # return "result_ablate_only_app"
    # return "result_ablate_only_shape"
    # return "result_ablate_no_relabel"
    # return "result_ablate_all"

    # -- normal mode --
    return Path("result") / dname
    # if dname == "davis":
    #     return "result_davis"
    # else:
    #     return "result"

# def check_flow_dir(flow_dir,img_dir):
#     counts = 0
#     for fn in flow_dir.iterdir():
#         fn.suffix == ".mat"
#         print(fn)
#     return flow_files

def get_command(dname,method,vid_name,params):

    # -- unpack --
    def unpack(params,key,default):
        return params[key] if key in params else default
    sp_size = params['sp_size']
    potts = params['potts']
    alpha = params['alpha']
    param_offset = unpack(params,"param_offset",0)
    tgt_nspix = unpack(params,"tgt_nspix",0)
    iperc_coeff = unpack(params,"iperc_coeff",4.0)
    split_alpha = unpack(params,"split_alpha",0.0)
    prop_nc = unpack(params,"prop_nc",1)
    prop_icov = unpack(params,"prop_icov",1)
    flow_name  = unpack(params,"flow_name","default")
    thresh_relabel = unpack(params,"thresh_relabel",0.00001)
    thresh_new = unpack(params,"thresh_new",0.01)
    sigma_app = unpack(params,"sigma_app",0.009)
    logging = unpack(params,"logging",0)
    param_id = params['param_id'] + param_offset
    flow_subdir,flow_ix = get_flow_strings(flow_name)
    nimgs = 3 if logging == 1 else 0

    # split_alpha = 2.0#alpha
    # iperc_coeff = 2.
    # thresh_relabel = 0.00001
    # thresh_new = 0.01
    # param_id = 11
    # param_id = 10
    # param_id = 0

    # -- check if done --
    if dname == "segtrackerv2":
        # root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
        root = Path("/srv/disk8tb/home/gauenk/Documents/packages/LIBSVXv4.0/Data/")
        img_dir = root /"SegTrackv2/PNGImages/" /vid_name
    elif dname == "davis":
        root = Path("/home/gauenk/Documents/data/davis/DAVIS/")
        img_dir = root /"JPEGImages/480p/" /vid_name
    else:
        raise ValueError(f"Uknown dataset [{dname}].")

    # potts = 1.
    # alpha = 0.
    # sp_str = "sp%d" % sp_size
    suffix = "jpg" if dname == "davis" else "png"
    from_tsp_to_bist_flows(img_dir)
    read_video = 1 if method == "bist" else 0
    # method_str = mangle_method(method,flow_name)
    if logging == 1:
        method_str = str(Path(method) / "logged")
    else:
        method_str = method
    result_dir = mangle_result_dir(dname)
    # output_subdir = str(Path(method) / sp_str / vid_name)
    if tgt_nspix > 0:
        output_subdir = str(Path(method_str) / ("sp%d"%tgt_nspix) / vid_name)
    else:
        output_subdir = str(Path(method_str) / ("param%d"%param_id) / vid_name)
    output_dir = Path(result_dir) / output_subdir
    flow_dir = img_dir / flow_subdir
    if not output_dir.exists(): output_dir.mkdir(parents=True)
    # print(output_dir)
    # exit()
    # if not check_flow_dir(flow_dir,img_dir):
    #     print(f"Skipping {vname} since incomplete flows.")
    #     return

    #  --alpha 0.
    # cmd = "./bin/bist -n %d -d %s/ -f %s/BIST_flows/ -o %s/ --read_video 1 --img_ext png --sigma_app 0.01 --potts 10." % (sp_size,img_dir,img_dir,output_dir)
    # cmd = "./bin/bist -n %d -d %s/ -f %s/BIST_flows/ -o %s/ --read_video 1 --img_ext png --sigma_app 0.01 --potts 10. --alpha 1.0" % (sp_size,img_dir,img_dir,output_dir)
    # cmd = "./bin/bist -n %d -d %s/ -f %s/BIST_flows/ -o %s/ --read_video 1 --img_ext png --sigma_app 0.01 --potts 10. --alpha 1.0" % (sp_size,img_dir,img_dir,output_dir)
    cmd = "./bin/bist -n %d -d %s/ -f %s/ -o %s/ --read_video %d --img_ext %s --sigma_app %2.5f --potts %2.2f --alpha %2.3f --split_alpha %2.3f --tgt_nspix %d --iperc_coeff %2.2f --thresh_relabel %1.8f --thresh_new %1.8f --prop_nc %d --prop_icov %d --logging %d --nimgs %d" % (sp_size,img_dir,flow_dir,output_dir,read_video,suffix,sigma_app,potts,alpha,split_alpha,tgt_nspix,iperc_coeff,thresh_relabel,thresh_new,prop_nc,prop_icov,logging,nimgs)
    print(cmd)

    # -- execute --
    done = True
    if output_dir.exists():
        if len(list(output_dir.iterdir())) == 0:
            done = False
        for in_fn in img_dir.iterdir():
            out_fn = output_dir / (in_fn.stem+".csv")
            if "TSP_flows" in str(out_fn): continue
            if not out_fn.exists():
                done = False
                break
    else:
        done = False

    # -- save params --
    # output_subdir = str(Path(method) / ("param%d"%param_id) / vid_name)
    # info_fn = str(Path(method) / ("param%d"%param_id) / vid_name)
    # info_fn = Path("./result") / method_str / "info" / ("param%d.csv"%param_id)
    info_fn = Path(result_dir) / method_str / "info" / ("param%d.csv"%param_id)
    if not info_fn.parents[0].exists(): info_fn.parents[0].mkdir()
    params = {k:[v] for k,v in params.items()}
    pd.DataFrame(params).to_csv(str(info_fn),index=False)

    return cmd,done


def from_tsp_to_bist_flows(root):
    # convert ".mat" to ".flo" file format
    tsp = root / "TSP_flows"
    bist = root / "BIST_flows"
    if not bist.exists():
        bist.mkdir(parents=True)
    tsp_files = list(tsp.iterdir())
    bist_files = list(bist.iterdir())
    if (len(tsp_files) == len(bist_files)):
        return

    for flow_fn in tsp.iterdir():
        flow = read_mat_flow(flow_fn)
        flow_flo = bist/(flow_fn.name.split("_")[0]+".flo")
        # print(flow.shape)
        # print(flow_fn,flow_flo)
        write_flo(flow,flow_flo)
        # exit()

def read_mat_flow(flow_fn):
    from scipy.io import loadmat
    flow = loadmat(str(flow_fn))['flow'][0][0]
    flow = np.stack([-flow[0],-flow[1]],axis=0)
    return flow

def write_flo(flow, filename):
    """
    Writes a .flo file (optical flow) from a numpy array.

    Args:
        flow (numpy.ndarray): The optical flow array of shape (height, width, 2).
        filename (str): The path to save the .flo file.
    """

    # if th.is_tensor(flow):
    #     flow = flow.detach().cpu().numpy()
    flow = rearrange(flow,"two h w -> h w two")
    with open(filename, 'wb') as f:
        # Write the header
        f.write(b'PIEH')  # Magic number
        f.write(np.array(flow.shape[1], dtype=np.int32).tobytes())  # Width
        f.write(np.array(flow.shape[0], dtype=np.int32).tobytes())  # Height
        # Write the flow data
        f.write(flow.astype(np.float32).tobytes())

def run_command(cmd):
    import subprocess
    # print(cmd)
    # output = ""
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    olines = output.split("\n")
    print(olines[-3:])
    match = re.search(r'Mean Time:\s([0-9]*\.?[0-9]+)', olines[-2])
    runtime = float(match.group(1))
    return output,runtime

def get_bist_experiments():

    def of(params,reps=1):
        _params = []
        for rep in range(reps):
            for pidx,p in enumerate(params):
                p['param_id'] = pidx+rep*len(params)
                _params.append(dcopy(p))
        return _params

    default_v0 = of(mesh({"sp_size":[25],
                          "potts":[10.0],
                          "alpha":[math.log(0.5)],
                          "split_alpha":[0.0],
                          "iperc_coeff":[4.0],
                          "param_offset":[0],
                          "thresh_relabel":[1e-6],
                          "thresh_new":[5.0e-2],
                          "logging":[0]}))
    ablate_relabel = of(mesh({"sp_size":[25],
                              "potts":[10.0],
                              "alpha":[math.log(0.5)],
                              "tgt_nspix":[0],
                              "split_alpha":[0.0],
                              "iperc_coeff":[4.0],
                              "param_offset":[200],
                              "thresh_relabel":[1e-6],
                              "thresh_new":[1.0e-1,5.0e-2,1e-2,1e-3]}),3)
    # ablate_relabel = of(mesh({"sp_size":[25],
    #                           "potts":[10.0],
    #                           "alpha":[math.log(0.5)],
    #                           "tgt_nspix":[0],
    #                           "split_alpha":[0.0],
    #                           "iperc_coeff":[4.0],
    #                           "param_offset":[200],
    #                           "thresh_relabel":[1e-4,1e-5,1e-6,1e-7],
    #                           "thresh_new":[1.0e-1,5.0e-2,1e-2,1e-3]}),3)
    # ablate_relabel += of(mesh({"sp_size":[25],
    #                           "potts":[10.0],
    #                           "alpha":[math.log(0.5)],
    #                           "tgt_nspix":[0],
    #                           "split_alpha":[0.0],
    #                           "iperc_coeff":[4.0],
    #                           "param_offset":[216],
    #                           "thresh_relabel":[1e-4,1e-5,1e-6,1e-7],
    #                           "thresh_new":[1.0e-1]}))
    # ablate_split = of(mesh({"sp_size":[25],
    #                         "potts":[10.0],
    #                         "alpha":[2.0,0.0,math.log(0.5)],
    #                         "tgt_nspix":[0],
    #                         "iperc_coeff":[4.0,2.0,0.0],
    #                         "split_alpha":[0.0,2.0],
    #                         "param_offset":[40],
    #                         "thresh_relabel":[1e-6],
    #                         "thresh_new":[5.0e-2]}))
    ablate_split = of(mesh({"sp_size":[25],
                            "potts":[10.0],
                            "alpha":[math.log(0.5)],
                            "tgt_nspix":[0],
                            "iperc_coeff":[8.0,4.0,0.0],
                            "split_alpha":[0.0],
                            "param_offset":[400],
                            "thresh_relabel":[1e-6],
                            "thresh_new":[5.0e-2]}),3)
    ablate_bndy = of(mesh({"sp_size":[25],
                           "potts":[10.0],
                           "alpha":[math.log(0.5)],
                           "tgt_nspix":[0],
                           "iperc_coeff":[4.0],
                           "split_alpha":[0.0],
                           "prop_nc":[0,1],
                           "prop_icov":[0,1],
                           "param_offset":[640]}),10)
    ablate_sp = of(mesh({"sp_size":[25],
                         "potts":[10.0,1.0],
                         "alpha":[math.log(0.5)],
                         "tgt_nspix":[0],
                         "iperc_coeff":[4.0],
                         "split_alpha":[0.0],
                         "param_offset":[300],
                         "sigma_app":[0.001,0.009,0.05,0.100],
                         "thresh_relabel":[1e-6],
                         "thresh_new":[5e-2]}))
    # target = mesh({"sp_size":[25],
    #                "alpha":[0.0], # alpha won't matter
    #                "split_alpha":[0.0], # alpha won't matter
    #                "potts":[10.0],
    #                "tgt_nspix":[0],
    #                "iperc_coeff":[math.log(0.5)], # won't matter
    #                "tgt_nspix":[300,500,800,1000,1200]})
    ablate_flows = of(mesh({"sp_size":[25],
                            "potts":[10.0],
                            "alpha":[math.log(0.5)],
                            "iperc_coeff":[4.0],
                            "split_alpha":[0.0],
                            "param_offset":[80],
                            "flow_name":["raft","spynet","default"]}))
    exps = default_v0 + ablate_relabel + ablate_split + ablate_bndy + ablate_sp + ablate_flows
    # exps = default_v0 + ablate_split + ablate_bndy
    # exps = default_v0
    exps = default_v0
    # exps = ablate_bndy
    # exps = ablate_relabel
    # exps = ablate_split
    # exps = [ablate_bndy[0],ablate_bndy[-1]]
    # exps = [ablate_bndy[1],ablate_bndy[2]]
    # exps = [ablate_bndy[0]]
    return exps

def get_method_params(method):
    params = mesh({"sp_size":[25],
                   # "alpha":[0.,0.5,1.0,1.5,2.0,2.5,3.0,3.5],
                   # "alpha":[0.,0.5,1.0,2.0,3.0,3.5,4.0],
                   # "alpha":[0.,0.5,1.0,2.0,3.0,4.0],
                   # "alpha":[math.log(0.5)],
                   # "alpha":[-1.0],
                   # "alpha":[1.0],
                   #
                   # ----------------------------------
                   #     Default BIST
                   # ----------------------------------
                   #
                   # "potts":[10.0],
                   # "alpha":[1.0],
                   # "split_alpha":[1.0],
                   # "iperc_coeff":[2.0,5.0],
                   # # "param_offset":[101],
                   # # "param_offset":[120],
                   # "param_offset":[140],
                   # "thresh_relabel":[1e-5,1e-6],
                   # # "thresh_new":[1e-2,5e-2,1e-1,5e-1]
                   # # "thresh_new":[2e-2,1e-2,5e-3]
                   # # "thresh_new":[1.5e-2]
                   # "thresh_new":[5.0e-2]
                   #
                   # ----------------------------------
                   #   Default BIST for DAVIS [aggresive]
                   # ----------------------------------
                   #
                   # "potts":[10.0],
                   # "alpha":[math.log(0.5)],
                   # "split_alpha":[math.log(0.5)],
                   # "iperc_coeff":[math.log(0.5)],
                   # "param_offset":[0],
                   # "thresh_relabel":[1e-6],
                   # "thresh_new":[5.0e-2],
                   # "logging":[0]
                   #
                   # ----------------------------------
                   #   Default BIST for DAVIS [conservative]
                   # ----------------------------------
                   #
                   "potts":[10.0],
                   "alpha":[math.log(0.5)],
                   "split_alpha":[0.0],
                   "iperc_coeff":[math.log(0.5)],
                   "param_offset":[1],
                   "thresh_relabel":[1e-6],
                   "thresh_new":[1.0e-2],
                   "logging":[0]
                   # # "thresh_new":[1.0,5.0e-2,1.0e-2]
                   #
                   # ----------------------------------
                   #   Default BIST for SegTrackerv2 [conservative]
                   # ----------------------------------
                   #
                   # "potts":[10.0],
                   # "alpha":[math.log(0.5)],
                   # "split_alpha":[0.0],
                   # "iperc_coeff":[math.log(0.5)],
                   # "param_offset":[0],
                   # "thresh_relabel":[1e-6],
                   # "thresh_new":[5.0e-2]
                   #
                   # ----------------------------------
                   #  Ablation BIST [Relabel]
                   # ----------------------------------
                   #
                   # "alpha":[0.0],
                   # "potts":[10.0],
                   # "tgt_nspix":[0],
                   # "iperc_coeff":[math.log(0.5)],
                   # "split_alpha":[math.log(0.5)],
                   # "param_offset":[200],
                   # "thresh_relabel":[1e-4,1e-5,1e-6,1e-7],
                   # "thresh_new":[1.0,5.0e-2,1e-2,1e-3]
                   #
                   #
                   # ------------------------------------------
                   #  Ablation BIST [Split] Grid for 40 - 51
                   # ------------------------------------------
                   #
                   # "alpha":[2.0,0.0,math.log(0.5)],
                   # "potts":[10.0],
                   # "tgt_nspix":[0],
                   # "iperc_coeff":[2.,0.,4.],
                   # "split_alpha":[2.0,0.0],
                   # "param_offset":[40],
                   # "thresh_relabel":[1e-6],
                   # "thresh_new":[5.0e-2]
                   #
                   # -------------------------------------------------------------------
                   #  Ablation BIST [boundary updates] Grid for 10 - 12 (or 60 - 72)
                   # -------------------------------------------------------------------
                   #
                   # "alpha":[math.log(0.5)],
                   # "potts":[10.0],
                   # "tgt_nspix":[0],
                   # "iperc_coeff":[math.log(0.5)],
                   # "split_alpha":[math.log(0.5)],
                   # "prop_nc":[0,1],
                   # "prop_icov":[0,1],
                   # "param_offset":[60],
                   #
                   # ----------------------------------
                   #  Ablation BIST [Sigma2,Potts]
                   # ----------------------------------
                   #
                   # "alpha":[math.log(0.5)],
                   # "potts":[10.0,1.0],
                   # "tgt_nspix":[0],
                   # "iperc_coeff":[math.log(0.5)],
                   # "split_alpha":[math.log(0.5)],
                   # "param_offset":[300],
                   # "sigma_app":[0.001,0.009,0.05,0.100],
                   # "thresh_relabel":[1e-6],
                   # "thresh_new":[5e-2]
                   #
                   # ----------------------------------
                   #        Targed Spix [the lines]
                   # ----------------------------------
                   #
                   # "alpha":[0.0], # alpha won't matter
                   # "split_alpha":[0.0], # alpha won't matter
                   # "potts":[10.0],
                   # "tgt_nspix":[0],
                   # "iperc_coeff":[math.log(0.5)],
                   # "tgt_nspix":[300,500,800,1000,1200],
                   #
                   # ----------------------------------
                   #   Default BIST on Various Flows
                   # ----------------------------------
                   #
                   # "potts":[10.0],
                   # "alpha":[math.log(0.5)],
                   # "split_alpha":[math.log(0.5)],
                   # "iperc_coeff":[math.log(0.5)],
                   # "param_offset":[80],
                   # "flow_name":["raft","spynet","default"],
    })
                   # "potts":[10.]})

    # -- old naming system --
    for pidx,p in enumerate(params):
        p['param_id'] = pidx

    if method == "bass":
        # params = mesh({"sp_size":[25],
        #                "potts":[10.0],
        #                "alpha":[math.log(0.5),2.0],
        #                "split_alpha":[0.0],
        #                "param_offset":[0]})
        params = mesh({"sp_size":[25],
                       "potts":[10.0],
                       "alpha":[math.log(0.5)],
                       "split_alpha":[0.0],
                       "param_offset":[0]})
        for pidx,p in enumerate(params):
            p['param_id'] = pidx

        # params = mesh({"sp_size":[25],
        #                "potts":[10.0],
        #                "alpha":[2.0,0.,math.log(0.5)],
        #                "split_alpha":[2.0,0.0],
        #                "param_offset":[100]})
        # params = mesh({"sp_size":[25],
        #                "potts":[10.0],
        #                "alpha":[2.0,0.,math.log(0.5)],
        #                "split_alpha":[2.0,0.0],
        #                "param_offset":[100]})
        params_v = params
    elif method == "bist":
        # params_v = odict({"n":[5,10,15,20,25,30,35,40,50,80,100],})
        params_v = get_bist_experiments()
        # params_v = params
    # elif method == "bist":
    #     # params_v = odict({"n":[15,20,25,30],})
    #     # params_v = odict({"n":[15],})
    else:
        raise ValueError(f"Uknown method params [{method}]")

    # param_grid = []
    # N = len(params_v["n"])
    # for n in range(N):
    #     params_n = {}
    #     for key,val in params_v.items():
    #         val_n = val[n] if len(val) > 1 else val[0]
    #         params_n[key] = val_n
    #     param_grid.append(params_n)
    # # names = ["%02dsp"%(p['superpixels']/100) for p in param_grid]
    return params_v

def link_output_files(method,pname,source_dir,link_dir,vid_name):

    # -- are we in the docker? --
    # parents = list(Path(".").resolve().parents)
    # if len(parents) == 1:
    #     in_docker = True
    # else:
    #     PC_ROOT = list(Path(".").resolve().parents)[-2]
    #     in_docker = PC_ROOT != "/home"
    # # links_dir = "links_d" if in_docker else "links"


    # -- link each output file --
    for src_fn in source_dir.iterdir():

        # -- get filenames --
        csv_fn = src_fn.name
        dest_fn = (vid_name+"_"+csv_fn)
        link_fn = link_dir/dest_fn
        relative_src_fn = Path("../")/vid_name/csv_fn
        if "runtime" in csv_fn: continue
        if "results" in csv_fn: continue
        if "correlation" in csv_fn: continue
        if "summary" in csv_fn: continue
        if not(".csv" in str(csv_fn)): continue

        # -- run link --
        # print(out_fn,link_fn)
        # cmd = "ln -s %s %s" % (str(src_fn.resolve()),str(link_fn.resolve()))
        cmd = "ln -s %s %s" % (str(relative_src_fn),str(link_fn))
        # print(cmd)
        # exit()
        if not link_fn.exists():
            run_command(cmd)
        # exit()

def save_run_info(output,dname,method,param_id,vid_name):
    base = Path("./result/%s/%s/run_log"%(dname,method))
    if not base.exists(): base.mkdir(parents=True)
    fn = str(base/("param%d_%s.txt"%(param_id,vid_name)))
    f = open(fn, "w+")
    f.write(output)
    f.close()

def main():


    # -- ensure correct directory --
    print("PID: ",os.getpid())

    # -- config --
    # dname = "davis"
    dname = "segtrackerv2"
    vid_names = get_video_names(dname)
    vid_names = vid_names

    # -- run script --
    # methods = ["ers","seeds","etps","ccs"]
    # methods = ["ccs"]
    # methods = ["etps"]
    # methods = ["bass"]
    # methods = ["bist"]
    # methods = ["bass","bist"]
    # methods = ["bass"]
    methods = ["bist"]
    for method in methods:
        param_grid = get_method_params(method)
        print("Number of grids: ",len(param_grid))
        # exit()
        # for params in tqdm.tqdm(param_grid):
        for params in param_grid:
            times = []
            for vid_name in vid_names:
                cmd,done = get_command(dname,method,vid_name,params)
                if not done:
                # if True:
                    output,time = run_command(cmd)
                    save_run_info(output,dname,method,params['param_id'],vid_name)
                    times.append(time)
            print(times)
            print(np.mean(times),np.std(times))
            # exit()
                # exit()
                # run_command(cmd)

    # -- collect links for summary --
    # methods = ["etps","seeds","ers","ccs"]
    # for method in methods:
    #     param_grid,pnames = get_method_params(method)
    #     for params,pname in zip(param_grid,pnames):
    #         link_dir = Path("out/%s/%s/%s/links" % (dname,method,pname))
    #         if link_dir.exists(): shutil.rmtree(link_dir)
    #         link_dir.mkdir()
    #         for vid_name in vid_names:
    #             output_dir = Path("out/%s/%s/%s/%s/" % (dname,method,pname,vid_name))
    #             if not output_dir.exists(): continue
    #             link_output_files(method,pname,output_dir,link_dir,vid_name)
    #             # exit()

    # -- collect counts to ensure correctness --
    # methods = ["etps","seeds","ers"]
    # methods = ["ccs"]
    # for method in methods:
    #     counts = {}
    #     param_grid,pnames = get_method_params(method)
    #     for params,pname in zip(param_grid,pnames):
    #         counts[pname] = 0
    #         output_dir = Path("out/%s/%s/%s/links" % (dname,method,pname))
    #         # print(output_dir)
    #         counts[pname] = len(list(output_dir.iterdir()))
    #     print(counts)



if __name__ == "__main__":
    main()
