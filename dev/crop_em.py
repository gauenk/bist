import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange


import glob
from pathlib import Path
from run_eval import read_video,read_seg,save_video


def main():
    dname = "davis"
    vname = "breakdance"
    vid = read_video(dname,vname)
    vid = vid[:,:,:480]
    print("vid.shape: ",vid.shape)

    root = Path("data/square/")
    vid = rearrange(th.from_numpy(vid),'t h w c -> t c h w')
    save_video(root,vid,0)


if __name__ == "__main__":
    main()
