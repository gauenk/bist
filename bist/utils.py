
"""

   Utils for BIST

"""


# -- imports --
import os
import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from einops import rearrange,repeat
import torchvision.io as tvio
import torchvision.utils as tv_utils
import torch.nn.functional as F
from easydict import EasyDict as edict

# -- local paths --
from ._paths import *

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#           Get Video Names
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_data_root(dname):
    if "segtrack" in dname.lower():
        return Path(SEGTRACKv2_ROOT)
    elif "davis" in dname.lower():
        return Path(DAVIS_ROOT)
    else:
        raise KeyError(f"Uknown dataset name [{dname}]")

def get_image_root(dname):
    if "segtrack" in dname.lower():
        return Path(SEGTRACKv2_ROOT)/"PNGImages/"
    elif "davis" in dname.lower():
        return Path(DAVIS_ROOT)/"JPEGImages/480p/"
    else:
        raise KeyError(f"Uknown dataset name [{dname}]")

def get_dataset_start_index(dname):
    if "segtrack" in dname.lower():
        return 1
    elif "davis" in dname.lower():
        return 0
    else:
        raise KeyError(f"Uknown dataset name [{dname}]")

def get_video_names(dname):
    if "segtrack" in dname.lower():
        return get_segtrackv2_videos()
    elif "davis" in dname.lower():
        return get_davis_videos()
    else:
        raise KeyError(f"Uknown dataset name [{dname}]")

def get_segtrackv2_videos():
    root = Path(SEGTRACKv2_ROOT) / "GroundTruth/"
    vnames = list([v.name for v in root.iterdir()])
    return vnames

def get_davis_videos():
    fname = Path(DAVIS_ROOT) / "ImageSets/2017/val.txt"
    vnames = np.loadtxt(fname,dtype=str)
    return vnames

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#               Data IO
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def save_spix(spix, root, fmt):

    root = Path(root)
    if not root.exists():
        root.mkdir(parents=True)

    print("Saving superpixels to %s"%str(root))
    for t in range(spix.shape[0]):
        spix_t = spix[t].cpu().numpy()
        fname = str(root / (fmt%t))
        pd.DataFrame(spix_t).to_csv(fname,header=False,index=None)

def save_video(video, root, fmt):

    root = Path(root)
    if not root.exists():
        root.mkdir(parents=True)

    print("Saving video to %s"%str(root))
    for t in range(video.shape[0]):
        fname = str(root / (fmt%t))
        save_image(video[t],fname)

def read_image(fname):
    return tvio.read_image(fname)/255.
def csv_to_th(fname):
    return th.from_numpy(pd.read_csv(str(fname),header=None,index_col=False).to_numpy())
def save_image(img,fname):
    if img.shape[-1] <= 4:
        img = rearrange(img,'h w c -> c h w')
    tv_utils.save_image(img,fname)
def crop_image(img,crop):
    hs,he,ws,we = crop
    return img[...,hs:he,ws:we]

def read_video(root):
    image_files = sorted([f for f in os.listdir(str(root)) if f.endswith(('.png', '.jpg'))])
    vid = []
    for image_file in image_files:
        # Read each image using thvision
        image_path = os.path.join(root, image_file)
        image = tvio.read_image(image_path).permute(1,2,0)/255.  # shape: (H, W, C)
        vid.append(image.float())
    vid = th.stack(vid)
    return vid

def read_spix(root,frames):
    spix = []
    for frame in frames:
        fn = root / ("%05d.csv"%frame)
        spix_f = np.asarray(pd.read_csv(fn,header=None))
        spix.append(th.from_numpy(spix_f))
    spix = th.stack(spix)
    return spix



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#            Optical Flow
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def read_flows(root,precision="32"):
    files = sorted([f for f in os.listdir(str(root)) if f.endswith(('.flo'))])
    flows = []
    for fname in files:
        path = os.path.join(root, fname)
        flow = th.from_numpy(read_flo(path,precision).copy())
        flows.append(flow)
    flows = th.stack(flows)
    return flows

def write_flows(flows,root,start_index=0,precision="32"):
    if isinstance(root,str): root = Path(root)
    if not root.exists(): root.mkdir(parents=True)
    if th.is_tensor(flows):
        flows = flows.cpu().numpy()

    T = flows.shape[0]
    for t in range(T):
        path = root / ("%05d.flo"%(t+start_index))
        write_flo(flows[t],path,precision)


def write_flo(flow, filename, precision="32"):
    """
    Writes a .flo file (optical flow) from a numpy array.

    Args:
        flow (numpy.ndarray): The optical flow array of shape (height, width, 2).
        filename (str): The path to save the .flo file.
    """

    # if th.is_tensor(flow):
    #     flow = flow.detach().cpu().numpy()
    precision_map = {"32": np.float32, "16": np.float16}
    if precision not in precision_map:
        raise ValueError("Invalid precision. Choose from '32' or '16'")

    flow = rearrange(flow,"two h w -> h w two")
    with open(filename, 'wb') as f:
        # Write the header
        f.write(b'PIEH')  # Magic number
        f.write(np.array(flow.shape[1], dtype=np.int32).tobytes())  # Width
        f.write(np.array(flow.shape[0], dtype=np.int32).tobytes())  # Height
        # Write the flow data
        f.write(flow.astype(precision_map[precision]).tobytes())

def read_flo(filename, precision="32"):
    """
    Reads a .flo optical flow file and returns it as a numpy array.

    Args:
        filename (str): Path to the .flo file.

    Returns:
        numpy.ndarray: The optical flow array of shape (height, width, 2).
    """

    precision_map = {"32": np.float32, "16": np.float16}
    if precision not in precision_map:
        raise ValueError("Invalid precision. Choose from '32' or '16'")

    with open(filename, 'rb') as f:
        # Read the magic number and check its validity
        magic = f.read(4)
        if magic != b'PIEH':
            raise ValueError(f"Invalid .flo file: {filename}")

        # Read the width and height
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # Read the optical flow data
        flow_data = np.frombuffer(f.read(), dtype=precision_map[precision])

        # Reshape the data to (height, width, 2)
        flow = flow_data.reshape((height, width, 2))

    return flow.astype(np.float32) # always go back to float32



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#           Unpacking Utils
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def extract_self(self,kwargs,defs):
    for k in defs:
        setattr(self,k,optional(kwargs,k,defs[k]))

def extract(_cfg,defs):
    return extract_defaults(_cfg,defs)

# def extract_defaults(_cfg,defs):
#     cfg = edict(dcopy(_cfg))
#     for k in defs: cfg[k] = optional(cfg,k,defs[k])
#     return cfg

def extract_defaults(cfg,defs):
    _cfg = edict()
    for k in defs: _cfg[k] = optional(cfg,k,defs[k])
    return _cfg

