
"""

    Create a "BIST in Action" video using on your own sequence.

    - Step 1: Run BIST (or BASS) with "logging=1"; with spix saved to "path/to/video_name/log"
    - Step 2: Run bist.animate.run("path/to/video_name/log/","path/to/save/video_name/")

"""

import tqdm
import os,shutil,random
import subprocess
import colorsys

import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange,repeat
import torchvision.io as tvio
import torchvision.utils as tv_utils

from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# -- package --
import bist

# -- save the mp4 --
import imageio
import imageio.v3 as iio


def run(vid,spix,log_root,anim_root,frames,method="bist"):


    # -- prepare --
    vid = vid.contiguous()
    spix = spix.contiguous()
    log_root = Path(log_root)
    anim_root = Path(anim_root)


    #
    # -- animate each sequence --
    #

    for frame in frames:
        if frame == 0 or (method == "bass"):
            animate_bass(vid[[frame]],spix[[frame]],anim_root,log_root,frame)
        else:
            animate_bist(vid[[frame]],vid[[frame-1]],spix[[frame-1]],anim_root,log_root,frame)

    #
    # -- concat all frames into a full movie --
    #

    # -- write filenames --
    write_mp4_root = anim_root/"mp4"
    write_fname = (anim_root/"mp4"/"file_names.txt").resolve()
    np.savetxt(write_fname,["file "+str((write_mp4_root/("%05d.mp4"%i)).resolve()) for i in frames],fmt="%s")

    # -- concat with ffmpeg --
    write_mp4 = write_mp4_root / "animation.mp4"
    cmd = "ffmpeg -y -f concat -safe 0 -i %s -c copy %s" % (write_fname,write_mp4.resolve())
    print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#               Primary Animation Functions
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def animate_bist(img1,img0,spix0,anim_root,log_root,frame):

    #
    #
    #   1.) Create the Frames for Animation
    #
    #

    # -- patchs with frame --
    frame_str    = "%05d" % frame
    bndy_root    = anim_root/"bndy"/frame_str
    saf_root     = anim_root/"saf"/frame_str
    merge_root   = anim_root/"merge"/frame_str
    split_root   = anim_root/"split"/frame_str
    relabel_root = anim_root/"relabel"/frame_str

    # -- read directorties --
    read_bndy_root    = log_root/"bndy"/frame_str
    read_split_root   = log_root/"split"/ frame_str
    read_merge_root   = log_root/"merge"/frame_str
    read_relabel_root = log_root/"relabel"/frame_str

    # -- viz show and fill --
    show_shift_and_fill(img1,img0,spix0,saf_root,log_root,frame)

    # -- viz boundary updates --
    show_boundary_updates(img1,bndy_root,read_bndy_root,frame)

    # -- viz splits, merges, & relabeling --
    max_spix_prev = spix0.max().item()
    for ix in range(2):
        show_splitting(img1,read_split_root,split_root,ix,max_spix_prev,frame)
        show_merging(img1,read_merge_root,merge_root,ix,max_spix_prev,frame)
        show_relabel(img1,read_relabel_root,relabel_root,ix,max_spix_prev,frame)

    #
    #
    #    2.) Put the frames in order
    #
    #

    # -- writing --
    write_root = anim_root/"mp4"/frame_str
    if write_root.exists(): shutil.rmtree(str(write_root))
    write_root.mkdir(parents=True)

    # -- bist algorithm ordering --
    order = ["bndy_16",
             "split_3","bndy_16",
             "bndy_16",
             "relabel_1","merge_3","bndy_16",
             "bndy_16",
             "split_3","bndy_16",
             "bndy_16",
             "relabel_1","merge_3","bndy_17"]
    starts = {"split":0,"merge":0,"bndy":0,"relabel":0}
    inames = []

    print("Saving bist sequence at ",write_root)
    anim_index = 0
    anim_index = link_seq(saf_root,write_root,anim_index,inames,"saf")
    for elem in order:
        name,num = elem.split("_")[0],int(elem.split("_")[1])
        read_root = anim_root/name/frame_str
        anim_index = link_seq(read_root,write_root,anim_index,inames,name,starts[name],starts[name]+num)
        starts[name]+=num
    print(anim_index)
    num_anim_frames = anim_index

    #
    #
    #    3.) Write the MP4
    #
    #

    # Define video writer
    write_root = anim_root/"mp4"
    if not write_root.exists(): write_root.mkdir()
    fname = write_root / (frame_str+".mp4")
    fps = 30  # Frames per second
    print("Writing mp4 file ",fname)
    writer = imageio.get_writer(fname, fps=fps, codec="libx264")
    frame_durations = {"saf":0.5,"split": 1.0, "merge":1.0, "bndy": 0.15, "relabel":1.0}  # in seconds

    print(num_anim_frames)
    for anim_index in range(num_anim_frames):
        category = inames[anim_index]
        read_fname = anim_root/"mp4" /frame_str/ ("%05d.png"%anim_index)
        img = iio.imread(read_fname)
        dur = frame_durations[category]
        if anim_index == (num_anim_frames-1): dur = 1.0
        num_frames = int(dur * fps)  # Convert duration to frame count
        for _ in range(num_frames):
            writer.append_data(img)
    writer.close()


def animate_bass(img,spix,anim_root,log_root,frame):


    #
    #
    #   1.) Create the Frames for Animation
    #
    #


    # -- write directories --
    frame_str  = "%05d" % frame
    bndy_root  = anim_root/"bndy"/frame_str
    merge_root = anim_root/"merge"/frame_str
    split_root = anim_root/"split"/frame_str

    # -- read directories --
    read_bndy_root  = log_root/"bndy"/frame_str
    read_split_root = log_root/"split"/frame_str
    read_merge_root = log_root/"merge"/frame_str

    # -- create update frames --
    niters_div_4 = (25-1)//4+1 # niters == 25 here...
    max_spix_prev = -1
    for ix in range(niters_div_4):
        show_splitting(img,read_split_root,split_root,ix,max_spix_prev,frame,"bass")
        if ix == (niters_div_4-1): continue
        show_merging(img,read_merge_root,merge_root,ix,max_spix_prev,frame,"bass")
    show_boundary_updates(img,bndy_root,read_bndy_root,frame,"bass")

    #
    #
    #    2.) Put the frames in order
    #
    #

    # -- writing --
    write_root = anim_root/"mp4"/frame_str
    if write_root.exists(): shutil.rmtree(str(write_root))
    write_root.mkdir(parents=True)

    # -- bass algorithm ordering --
    order = ["split_3","bndy_16",
             "bndy_16",
             "merge_3","bndy_16",
             "bndy_16"]*6 + ["split_3","bndy_17"]
    starts = {"split":0,"merge":0,"bndy":0}
    inames = []

    print("Saving bass sequence at ",write_root)
    anim_index = 0
    for elem in order:
        name,num = elem.split("_")[0],int(elem.split("_")[1])
        read_root = anim_root/name/frame_str
        anim_index = link_seq(read_root,write_root,anim_index,inames,name,starts[name],starts[name]+num)
        starts[name]+=num
    num_anim_frames = anim_index


    #
    #
    #    3.) Write the MP4
    #
    #

    # Define video writer
    write_root = anim_root/"mp4" # even tho bass
    if not write_root.exists(): write_root.mkdir()
    fname = write_root / (frame_str+".mp4")
    fps = 30  # Frames per second
    print("Writing mp4 file ",fname)
    writer = imageio.get_writer(fname, fps=fps, codec="libx264")

    # Write frames with correct timing
    frame_durations = {"saf":0.5,"split": 1.0, "merge":1.0, "bndy": 0.15}  # in seconds

    # -- get initial frame --
    img = rearrange(img,'1 h w c -> c h w')
    init_img = th.nn.functional.pad(img,(0,0,45,0),value=1.0)
    init_img = add_text_to_tensor(init_img,"Frame %d"%(frame+1),"right")
    init_img = add_text_to_tensor(init_img,"BASS","left")
    init_img = add_text_to_tensor(init_img,"Input Image","center")
    init_img = rearrange(init_img,'c h w -> h w c').cpu().numpy()
    init_img = np.clip(init_img*255,0.,255.).astype(np.uint8)
    dur = 1.5
    num_frames = int(dur * fps)  # Convert duration to frame count
    for _ in range(num_frames):
        writer.append_data(init_img)

    # -- bass seq --
    for anim_index in range(num_anim_frames):
        category = inames[anim_index]
        read_fname = anim_root/"mp4"/frame_str/("%05d.png"%anim_index)
        img = iio.imread(read_fname)

        dur = frame_durations[category]
        if anim_index == (num_anim_frames-1): dur = 1.5
        num_frames = int(dur * fps)  # Convert duration to frame count
        for _ in range(num_frames):
            writer.append_data(img)
    writer.close()













# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#
#
#
#              Create Frames for Each Step
#
#
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def show_shift_and_fill(img1,img0,spix0,save_root,log_root,frame):

    # -- create dir --
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- mark --
    img0,img1 = img0.clone(),img1.clone()
    color = th.tensor([1.0,1.0,1.0])*0.7
    anim_index = 0
    frame_str = "%05d"%frame

    # -- nice background --
    bkg = tvio.read_image("assets/anim/transparent_png.jpg").cuda()/255.

    # -- shift --
    shifted = csv_to_th_dir(log_root/"shifted"/frame_str).cuda()
    for shift in shifted:
        img_shift = bist.utils.index_tensor(img0,shift)
        img_shift = fill_invalid(img_shift,bkg,shift)
        spix_shift = bist.utils.index_tensor(spix0[:,:,:,None],shift)[:,:,:,0]
        img_shift_m = bist.get_marked_video(img_shift,spix_shift,color)
        save_image(img_shift_m[0],save_root,anim_index,"shift",frame)
        anim_index+=1

    # -- swap to next frame --
    fill_init = fill_invalid(img1,bkg,shifted[-1])
    fill_init_m = bist.get_marked_video(fill_init,spix_shift,color)
    save_image(fill_init_m[0],save_root,anim_index,"fill",frame)
    anim_index+=1

    # -- fill step --
    spix_filled = csv_to_th_dir(log_root/"filled"/frame_str).cuda()
    for spix in spix_filled:
        fill = fill_invalid(img1,bkg,spix)
        fill_m = bist.get_marked_video(fill,spix,color)
        # img_shift = fill_invalid(img_shift,bkg,shift)
        save_image(fill_m[0],save_root,anim_index,"fill",frame)
        anim_index+=1
    return anim_index

def show_boundary_updates(img1,save_root,read_root,iframe,method="bist"):
    img1 = img1.clone()
    if not save_root.exists():
        save_root.mkdir(parents=True)
    color = th.tensor([1.0,1.0,1.0])*0.7
    nsegs = len(list(read_root.iterdir()))
    for ix in range(nsegs):
        fname = read_root / ("%05d.csv"%ix)
        spix = csv_to_th(fname)
        marked = bist.get_marked_video(img1,spix,color)
        save_image(marked[0],save_root,ix,"boundary_updates",iframe,method)
    save_image(marked[0],save_root,nsegs,"complete",iframe,method)
    return nsegs

def show_splitting(img1,split_root,save_root,split_ix,max_spix_prev,iframe,method="bist"):

    # -- config output --
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- read --
    proposed_spix = csv_to_th(split_root / ("%05d.csv" % (4*split_ix)))[None,].cuda()
    init_spix = csv_to_th(split_root / ("%05d.csv" % (4*split_ix+1)))[None,].cuda()
    accepted_spix = csv_to_th(split_root / ("%05d.csv" % (4*split_ix+3)))[None,].cuda()

    # -- ensure the proposed side is the accepted side [it can swap in the code] --
    cond0 = accepted_spix != proposed_spix
    cond1 = accepted_spix != init_spix
    args = th.where(th.logical_and(cond0,cond1))
    old_spix = th.minimum(accepted_spix[args],proposed_spix[args])
    new_spix = th.maximum(accepted_spix[args],proposed_spix[args])
    pairs = th.stack([old_spix,new_spix],-1)
    pairs = th.unique(pairs,dim=0)
    for old,new in pairs:
        args_old = th.where(accepted_spix == old)
        args_new = th.where(accepted_spix == new)
        proposed_spix[args_old] = old
        proposed_spix[args_new] = new
    max_spix = init_spix.max().item()

    #
    # -- Get the proposed splits --
    #

    # -- split filled regions on image --
    img1 = img1.clone()
    in_img1 = img1.clone()
    alpha = 0.5
    color = th.tensor([0.0,0.0,1.0]).cuda()*0.7
    img1 = th.where(max_spix>proposed_spix.unsqueeze(-1),img1,(1-alpha)*img1+alpha*color)

    # -- get marked img --
    color = th.tensor([1.0,1.0,1.0]).cuda()*0.7
    img1 = bist.get_marked_video(img1,init_spix,color)

    # -- save --
    save_image(img1[0],save_root,3*split_ix,"split_proposed",iframe,method)


    #
    # -- Get the accepted splits --
    #


    # -- split filled regions on image --
    img1 = in_img1.clone().cuda()
    alpha = 0.5
    color = th.tensor([0.0,0.0,1.0]).cuda()*0.7 # fill color
    img1 = th.where(max_spix>accepted_spix.unsqueeze(-1),img1,(1-alpha)*img1+alpha*color)
    color = th.tensor([1.0,1.0,1.0]).cuda()*0.7 # boundary color
    img1_m = bist.get_marked_video(img1,init_spix,color)
    save_image(img1_m[0],save_root,3*split_ix+1,"split_accepted",iframe,method)
    img1_m = bist.get_marked_video(img1,accepted_spix,color)
    save_image(img1_m[0],save_root,3*split_ix+2,"split_accepted",iframe,method)
    split_ix+=1

    return split_ix

def show_merging(img1,merge_root,save_root,merge_ix,max_spix_prev,iframe,method="bist"):

    # -- config output --
    img1 = img1.clone()
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- read --
    spix = csv_to_th(merge_root / ("%05d.csv" % (4*merge_ix)))[None,].cuda()
    proposed = csv_to_th(merge_root / ("%05d.csv" % (4*merge_ix+1)))[:,0].cuda()
    accepted_spix = csv_to_th(merge_root / ("%05d.csv" % (4*merge_ix+2)))[None,].cuda()

    # -- load details --
    merge_details_root = merge_root.parents[1] / "merge_details" / merge_root.name
    details = None
    if merge_details_root.exists():
        details = csv_to_th(merge_details_root/("%05d.csv" % (merge_ix))).cuda()

    # -- list of proposed ones into accepted merges --
    accepted_ref = th.unique(spix[th.where(accepted_spix != spix)])
    accepted_tgt = th.unique(accepted_spix[th.where(accepted_spix != spix)])

    # -- correct the accepted to match the proposed --
    _accepted_ref = th.unique(spix[th.where(accepted_spix != spix)])
    accepted_ref,accepted_tgt = [],[]
    for ref in _accepted_ref:
        args = th.logical_and(accepted_spix != spix,spix == ref)
        tgt = th.unique(accepted_spix[th.where(args)])
        assert len(tgt) == 1
        tgt = tgt.item()
        if proposed[ref] == tgt:
            accepted_ref.append(ref)
            accepted_tgt.append(tgt)
        elif proposed[tgt] == ref:
            accepted_ref.append(tgt)
            accepted_tgt.append(ref)
        else:
            print("what?")
            exit()
    accepted_ref  = th.tensor(accepted_ref).cuda()
    accepted_tgt  = th.tensor(accepted_tgt).cuda()

    # -- get topk (in terms of size) merged positions --
    counts = th.bincount(spix.ravel(),minlength=proposed.shape[0])
    vals,spix_to_merge = th.topk(counts * (proposed >= 0),k=5000)
    spix_to_merge = spix_to_merge[vals>0]
    spix_merged_into = th.unique(proposed[spix_to_merge])

    # -- colors --
    colors = color_wheel_tensor(len(spix_merged_into))
    dark = th.clamp(colors * 0.75,0.0,1.0)
    light = th.clamp(colors * 1.25,0.0,1.0)

    # -- show merged --
    iprop = img1.clone()
    iacc = img1.clone()
    alpha = 0.5
    for ix,spix_id in enumerate(spix_merged_into): # "target" the consuming superpixel

        # -- mark spix which is the target --
        color = dark[ix,:].cuda()[None,None,None,:] * th.ones_like(img1)
        if th.any(accepted_tgt == spix_id):
            iacc = th.where((spix_id == spix).unsqueeze(-1),(1-alpha)*iacc+(alpha)*color,iacc)
        if th.all(accepted_ref != spix_id): # if the target is not accepted as a reference, then draw it
            iprop = th.where((spix_id == spix).unsqueeze(-1),(1-alpha)*iprop+(alpha)*color,iprop)

        # -- show proposed references [the source of the arrow] --
        spix_to_merge = th.where(proposed == spix_id)[0]
        color = light[ix,:].cuda()[None,None,None,:] * th.ones_like(img1)
        for spix_id2 in spix_to_merge: # "references" the spix that dissappears
            if th.all(accepted_tgt != spix_id2): # [spix_id2] is never the target of an accepted pair
                iprop = th.where((spix_id2 == spix).unsqueeze(-1),(1-alpha)*iprop+(alpha)*color,iprop)
            if th.any(accepted_ref == spix_id2):
                iacc = th.where((spix_id2 == spix).unsqueeze(-1),(1-alpha)*iacc+(alpha)*color,iacc)

    # -- save --
    color = th.tensor([1.0,1.0,1.0])*0.7
    iprop = bist.get_marked_video(iprop,spix,color)
    save_image(iprop[0],save_root,3*merge_ix,"merge_proposed",iframe,method)
    iacc_m = bist.get_marked_video(iacc,spix,color)
    save_image(iacc_m[0],save_root,3*merge_ix+1,"merge_accepted",iframe,method)
    iacc_m = bist.get_marked_video(iacc,accepted_spix,color)
    save_image(iacc_m[0],save_root,3*merge_ix+2,"merge_accepted",iframe,method)

def show_relabel(img1,read_root,save_root,ix,max_spix_prev,iframe,method="bist"):


    # -- config output --
    if not save_root.exists():
        save_root.mkdir(parents=True)
    img1 = img1.clone()

    # -- read --
    init_spix = csv_to_th(read_root / ("%05d.csv" % (2*ix+0)))[None,].cuda()
    final_spix = csv_to_th(read_root / ("%05d.csv" % (2*ix+1)))[None,].cuda()
    max_spix = init_spix.max().item()

    delta_spix = th.unique(final_spix[th.where(init_spix != final_spix)])
    relabeled_spix = delta_spix[th.where(delta_spix <= init_spix.max())]
    new_spix = delta_spix[th.where(delta_spix > init_spix.max())]
    # print("[relabel]: len(relabeled_spix),len(new_spix) ",len(relabeled_spix),len(new_spix))

    # -- split filled regions on image --
    in_img1 = img1.clone()
    alpha = 0.5
    color = th.tensor([0.0,0.0,1.0]).cuda()*0.7
    args = th.isin(final_spix,relabeled_spix).unsqueeze(-1)
    img1 = th.where(args,(1-alpha)*img1+alpha*color,img1)
    color = th.tensor([1.0,0.0,0.0]).cuda()*0.7
    args = th.isin(final_spix,new_spix).unsqueeze(-1)
    img1 = th.where(args,(1-alpha)*img1+alpha*color,img1)

    # -- marked 0 --
    color = th.tensor([1.0,1.0,1.0]).cuda()*0.7
    img1_m = bist.get_marked_video(img1,init_spix,color)
    save_image(img1_m[0],save_root,ix,"relabel",iframe,method)














# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#
#
#
#                   Animation Utils
#
#
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def csv_to_th_dir(root):
    spix = []
    for index in range(len(list(root.iterdir()))):
        spix.append(csv_to_th(root / ("%05d.csv" % index)))
    return th.stack(spix)

def csv_to_th(fname):
    return th.from_numpy(pd.read_csv(str(fname),header=None).to_numpy())

def get_merge_colors(k=10):
    def adjust_brightness(color, frac):
        """Adjusts the brightness of an RGB color in a given range (0-1 for Matplotlib)."""
        # return tuple(np.clip(c * random.uniform(brightness_range[0], brightness_range[1]), 0, 1) for c in color)
        return tuple(np.clip(c * frac, 0, 1) for c in color)
    lighter_colors = []
    darker_colors = []
    tab10_colors = plt.get_cmap("tab10").colors  # Returns 10 distinct colors
    for base in tab10_colors:  # Loop through Matplotlib 'tab10' colors
        lighter_colors.append(adjust_brightness(base, 1.25))  # Brighter variation
        darker_colors.append(adjust_brightness(base, 0.75))  # Darker variation
    return th.tensor(lighter_colors),th.tensor(darker_colors)

def color_wheel_tensor(n: int) -> th.Tensor:
    """
    Generate a PyTh tensor of RGB values spaced equally around the color wheel.

    Args:
        n (int): Number of colors to generate.

    Returns:
        th.Tensor: Tensor of shape (n, 3) with RGB values in the range [0, 1].
    """
    hues = th.linspace(0, 1, n, dtype=th.float32)
    colors = [colorsys.hsv_to_rgb(h.item(), 1, 1) for h in hues]
    return th.tensor(colors, dtype=th.float32)

def link_seq(read_root,write_root,anim_index,inames,name,iread_start=0,iread_end=0):
    if iread_end == 0:
        iread_end = len(list(Path(read_root).iterdir()))
    for index in range(iread_start,iread_end):

        # -- paths --
        read_path = read_root / ("%05d.png" % index)
        assert read_path.exists(),str(read_path)
        write_path = write_root / ("%05d.png" % anim_index)

        # -- remove/unlink --
        try:
            os.unlink(str(write_path))
        except:
            pass
        # if anim_path.exists(): os.unlink(str(anim_path))

        # -- copy --
        cmd = "ln -s %s %s" % (read_path.resolve(),write_path)
        output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
        olines = output.split("\n")

        # -- append --
        inames.append(name)

        anim_index+=1
    return anim_index

def fill_invalid(tensor,bkg,spix):
    B,H,W,C = tensor.shape
    bkg = repeat(bkg,'1 h w -> 1 (hr h) (wr w) c',hr=2,wr=2,c=3)[:,:H,:W]
    assert bkg.shape[1] == H and bkg.shape[2] == W
    tensor = th.where(spix.unsqueeze(-1)==-1,bkg,tensor)
    return tensor

def get_text_img_from_name(name,ishape):
    fname = Path("./assets/anim/text_images")/("%s.png"%name)
    text = tvio.read_image(fname)/255.
    _,H,W = ishape
    topad = W - text.shape[2]
    left = topad//2
    right = topad - left
    text = th.nn.functional.pad(text,(left,right,0,0),value=1)
    text = text[:3]
    return text

def save_image(img,save_root,anim_index,name,iframe,method="bist"):
    fname = save_root / ("%05d.png"%anim_index)
    img = rearrange(img,'h w c -> c h w')

    # -- write frame index --
    img = th.nn.functional.pad(img,(0,0,45,0),value=1.0)
    text_img = get_text_img_from_name(name,img.shape)
    img[:,3:3+41] = text_img

    # -- write text --
    img = add_text_to_tensor(img,"Frame %d"%(iframe+1),"right")
    img = add_text_to_tensor(img,method.upper(),"left")

    # -- save --
    print("Saving: ",fname)
    tv_utils.save_image(img,fname)

def add_text_to_tensor(image_tensor, text, align="left", font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize=30):
    """
    Adds an integer text to the top-right corner of a PyTorch image tensor.

    Args:
        image_tensor (torch.Tensor): Image of shape (C, H, W) with values in [0, 1].
        text (str): Text to overlay on the image.
        font_path (str): Path to the TrueType font file.
        fontsize (int): Size of the text.

    Returns:
        torch.Tensor: Modified image tensor with text overlay.
    """

    # Convert tensor to PIL Image (C, H, W -> H, W, C) and scale to [0, 255]
    to_pil = transforms.ToPILImage()
    img = to_pil(image_tensor)

    # Draw text on image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, fontsize)

    # Get image dimensions
    img_width, img_height = img.size

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)  # Returns (left, top, right, bottom)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Position (top-right corner with 10px padding)
    if align == "right":
        x = img_width - text_width - 10
        y = 10
    elif align == "center":
        x = img_width//2 - text_width//2
        y = 10
    else:
        x = 10
        y = 10

    # Add text to image
    draw.text((x, y), text, font=font, fill="grey")

    # Convert back to PyTorch tensor (H, W, C -> C, H, W) and normalize to [0,1]
    to_tensor = transforms.ToTensor()
    return to_tensor(img)

