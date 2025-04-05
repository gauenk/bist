"""

   DAVIS reader for spixconv train/test

"""

# -- imports --
import os,random
import numpy as np
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict
import torchvision.io as tvio
from .._paths import DAVIS_ROOT

# -- pytorch --
import torch as th
from torch.utils.data import DataLoader

# -- for training --
import torchvision.transforms as T
import torchvision.transforms.functional as xformF

def get_davis_dataset(tr_nframes=-1):

    # -- dataset --
    data = edict()
    data.tr = DAVIS(DAVIS_ROOT,"train-val",tr_nframes)
    data.val = DAVIS(DAVIS_ROOT,"val",-1)
    data.te = DAVIS(DAVIS_ROOT,"test-dev",-1)

    return data

def get_loaders(data,num_workers,seed):

    # -- random generator --
    g = th.Generator()
    g.manual_seed(seed)
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # -- loaders --
    loader = edict()
    loader.tr = DataLoader(data.tr,shuffle=True,
                           batch_size=1,num_workers=num_workers,
                           worker_init_fn=seed_worker,generator=g)
    loader.val = DataLoader(data.val,shuffle=False,
                            batch_size=1,num_workers=num_workers,
                            worker_init_fn=seed_worker,generator=g)
    loader.te = DataLoader(data.te,shuffle=False,
                           batch_size=1,num_workers=num_workers,
                           worker_init_fn=seed_worker,generator=g)
    return loader

class DAVIS():

    def __init__(self,root,split,nframes):

        # -- load paths --
        root = Path(root)
        self.vnames = _get_vid_names(root/("ImageSets/2017/%s.txt"%split))
        self.paths = read_files(root/"JPEGImages/480p/",self.vnames,nframes)
        self.groups = sorted(list(self.paths.keys()))

        # -- limit num of samples --
        nsamples = -1
        self.indices = enumerate_indices(len(self.paths),nsamples)
        self.nsamples = len(self.indices)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # -- indices --
        image_index = self.indices[index]
        group = self.groups[image_index]
        vid_files = self.paths[group]
        video = _read_video(vid_files)
        anno = _read_annos(vid_files,video.shape)
        return video,anno,index


def _get_vid_names(vid_fn):
    with open(vid_fn,"r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    return names

def _read_annos(paths,ishape):
    vid = []
    for path_t in paths:
        path_t = str(path_t).replace("JPEGImages","Annotations")
        path_t = path_t.replace("jpg","png")
        path_t = Path(path_t)
        if not path_t.exists(): break
        vid_t = tvio.read_image(path_t).float()[0]
        vid.append(vid_t)
    if len(vid) == 0:
        vid = th.zeros(ishape)
    else:
        vid = th.stack(vid).float()
    return vid

def _read_video(paths):
    vid = []
    for path_t in paths:
        if not Path(path_t).exists(): break
        vid_t = tvio.read_image(path_t).float()/255.
        vid.append(vid_t)
    vid = th.stack(vid)
    return vid


def enumerate_indices(total_samples,selected_samples,rand_sel=True,skip=1):
    if selected_samples > 0:
        if rand_sel == True:
            indices = th.randperm(total_samples)
            indices = indices[:selected_samples]
        else:
            indices = th.arange(total_samples)[::skip]
            indices = indices[:selected_samples]
    else:
        indices = th.arange(total_samples)
    return indices

def _get_video_paths(root):
    image_files = sorted([f for f in os.listdir(str(root)) if f.endswith(('.png', '.jpg'))])
    image_files = [Path(root)/f for f in image_files]
    return image_files

def read_files(iroot,vnames,nframes):
    # -- get files --
    files = {}
    dil,stride = 1,1
    for vname in vnames:
        vid_dir = iroot/vname
        vid_paths = _get_video_paths(vid_dir)
        total_nframes = len(vid_paths)
        assert total_nframes > 0

        # -- pick number of sub frames --
        nframes_vid = nframes
        if nframes_vid <= 0:
              nframes_vid = total_nframes

        # -- compute num subframes --
        n_subvids = (total_nframes - (nframes_vid-1)*dil - 1)//stride + 1

        # -- reflect bound --
        def bnd(num,lim):
            if num >= lim: return 2*(lim-1)-num
            else: return num

        for group in range(n_subvids):
            start_t = group * stride
            if n_subvids == 1: vid_id = vname
            else: vid_id = "%s:%02d" % (vname,start_t)
            end_t = start_t + nframes_vid
            paths_t = [vid_paths[bnd(t,total_nframes)] for t in range(start_t,end_t)]
            files[vid_id] = paths_t

    return files


def random_crop(video,anno,size):
    F = xformF
    i, j, h, w = T.RandomCrop.get_params(video[0], output_size=[size,size])
    video = th.stack([F.crop(frame, i, j, h, w) for frame in video])
    anno = th.stack([F.crop(anno_f, i, j, h, w) for anno_f in anno])
    return video,anno

