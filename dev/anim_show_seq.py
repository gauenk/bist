
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

import glob
from pathlib import Path
# from run_eval import read_video,read_seg,get_video_names
from st_spix.utils import rgb2lab
from st_spix.spix_utils.updated_io import read_video,read_seg
from st_spix.spix_utils.evaluate import computeSummary,scoreSpixPoolingQualityByFrame,count_spix,read_spix
from st_spix.spix_utils.evaluate import get_video_names

import bist_cuda

import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def csv_to_th(fname):
    # print(fname)
    return th.from_numpy(pd.read_csv(str(fname),header=None).to_numpy())

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

def get_marked_video(vid,spix,color):
    vid = vid.cuda().contiguous().float()
    spix = spix.cuda().contiguous().int()
    color = color.cuda().contiguous().float()
    marked = bist_cuda.get_marked_video(vid,spix,color)
    return marked

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

def create_bist_seq(root,frame):

    # -- The BIST algorithm --
    frame_str = "%05d"%frame
    names = []

    # -- roots of saved images --
    bndy_root = root / "anim_bndy" / frame_str
    saf_root = root / "anim_saf" / frame_str
    merge_root = root/ "anim_merge" / frame_str
    split_root = root/"anim_split" / frame_str
    relabel_root = root/"anim_relabel" / frame_str

    # -- writing --
    write_root = root/"anim_bist" / frame_str
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
        read_root = root/("anim_%s"%name)/frame_str
        anim_index = link_seq(read_root,write_root,anim_index,inames,name,starts[name],starts[name]+num)
        starts[name]+=num
    # np.savetxt(write_root/"inames.txt",inames,fmt="%s")
    # anim_index = link_seq(read_root,write_root,anim_index,inames,starts["bndy"],starts["bndy"]+1)
    print(anim_index)
    num_anim_frames = anim_index

    #
    # -- write movie --
    #

    import imageio
    import imageio.v3 as iio

    # Define video writer
    write_root = root/"anim_bist_mp4"
    if not write_root.exists(): write_root.mkdir()
    fname = write_root / (frame_str+".mp4")
    fps = 30  # Frames per second
    # writer = iio.imopen(fname, "FFMPEG", mode="w", fps=fps)
    print("Writing mp4 file ",fname)
    writer = imageio.get_writer(fname, fps=fps, codec="libx264")

    # Write frames with correct timing
    frame_durations = {"saf":0.5,"split": 1.0, "merge":1.0, "bndy": 0.15, "relabel":1.0}  # in seconds

    print(num_anim_frames)
    for anim_index in range(num_anim_frames):
        category = inames[anim_index]
        read_fname = root/"anim_bist" / frame_str / ("%05d.png"%anim_index)
        img = iio.imread(read_fname)
        dur = frame_durations[category]
        if anim_index == (num_anim_frames-1): dur = 1.0
        num_frames = int(dur * fps)  # Convert duration to frame count
        for _ in range(num_frames):
            # writer.write(img)
            writer.append_data(img)
    writer.close()

    return names

def shift_tensor(tensor,index):

    # -- reshape --
    B,H,W,C = tensor.shape
    tensor = rearrange(tensor,'b h w c -> h w (b c)')
    shifted = th.zeros_like(tensor).to(tensor.device)

    # Mask out the valid indices
    valid_mask = index != -1
    valid_indices = index[valid_mask]  # Extract valid indices

    # Compute the row/col positions of valid indices in the original image
    rows, cols = th.meshgrid(th.arange(H), th.arange(W), indexing='ij')
    rows = rows.to(tensor.device)
    cols = cols.to(tensor.device)
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]

    # Map the valid indices to their new positions
    shifted[valid_rows, valid_cols] = tensor.reshape(-1, B*C)[valid_indices]

    shifted = rearrange(shifted,'h w (b c) -> b h w c',b=B)
    return shifted

def fill_invalid(tensor,bkg,spix):
    B,H,W,C = tensor.shape
    bkg = repeat(bkg,'1 h w -> 1 (hr h) (wr w) c',hr=2,wr=2,c=3)[:,:H,:W]
    assert bkg.shape[1] == H and bkg.shape[2] == W
    tensor = th.where(spix.unsqueeze(-1)==-1,bkg,tensor)
    return tensor

def get_text_img_from_name(name,ishape):
    fname = Path("./data/text_images")/("%s.png"%name)
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

def get_logged_spix(root):
    spix = []
    for index in range(len(list(root.iterdir()))):
        raw_path = root / ("%05d.csv" % index)
        # spix_t = pd.read_csv(str(raw_path),header=None).to_numpy()
        # th.from_numpy(spix_t))
        spix.append(csv_to_th(raw_path))

    return th.stack(spix)

def show_shift_and_fill(vid,spix,root,save_root,frame):

    # -- create dir --
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- mark --
    img0,img1 = vid[[0]],vid[[1]]
    spix0 = spix[[0]]
    color = th.tensor([1.0,1.0,1.0])*0.7
    # img0 = get_marked_video(img0,spix0,color)
    anim_index = 0
    frame_str = "%05d"%frame

    # -- nice background --
    bkg = tvio.read_image("data/transparent_png.jpg").cuda()/255.

    # -- show anim --
    shifted = get_logged_spix(root/"log"/"shifted"/frame_str).cuda()
    for shift in shifted:
        img_shift = shift_tensor(img0,shift)
        img_shift = fill_invalid(img_shift,bkg,shift)
        spix_shift = shift_tensor(spix0[:,:,:,None],shift)[:,:,:,0]
        img_shift_m = get_marked_video(img_shift,spix_shift,color)
        save_image(img_shift_m[0],save_root,anim_index,"shift",frame)
        anim_index+=1

    # -- shift and fill --
    fill_init = fill_invalid(img1,bkg,shifted[-1])
    fill_init_m = get_marked_video(fill_init,spix_shift,color)
    save_image(fill_init_m[0],save_root,anim_index,"fill",frame)
    anim_index+=1

    spix_filled = get_logged_spix(root/"log"/"filled"/frame_str).cuda()
    for spix in spix_filled:
        fill = fill_invalid(img1,bkg,spix)
        fill_m = get_marked_video(fill,spix,color)
        # img_shift = fill_invalid(img_shift,bkg,shift)
        save_image(fill_m[0],save_root,anim_index,"fill",frame)
        anim_index+=1
    return anim_index

# def show_bass(img0,root,save_root,read_root,iframe):
#     pass

def show_boundary_updates(img1,root,save_root,read_root,iframe,method="bist"):
    if not save_root.exists():
        save_root.mkdir(parents=True)
    color = th.tensor([1.0,1.0,1.0])*0.7
    nsegs = len(list(read_root.iterdir()))
    for ix in range(nsegs):
        fname = read_root / ("%05d.csv"%ix)
        spix = csv_to_th(fname)
        marked = get_marked_video(img1,spix,color)
        save_image(marked[0],save_root,ix,"boundary_updates",iframe,method)
    save_image(marked[0],save_root,nsegs,"complete",iframe,method)
    return nsegs

def get_merge_colors(k=10):
    def adjust_brightness(color, frac):
        """Adjusts the brightness of an RGB color in a given range (0-1 for Matplotlib)."""
        # return tuple(np.clip(c * random.uniform(brightness_range[0], brightness_range[1]), 0, 1) for c in color)
        return tuple(np.clip(c * frac, 0, 1) for c in color)
    lighter_colors = []
    darker_colors = []
    tab10_colors = plt.get_cmap("tab10").colors  # Returns 10 distinct colors
    for base in tab10_colors:  # Loop through Matplotlib 'tab10' colors
        # lighter_colors.append(adjust_brightness(base, (1.2, 1.5)))  # Brighter variation
        # darker_colors.append(adjust_brightness(base, (0.5, 0.8)))  # Darker variation
        lighter_colors.append(adjust_brightness(base, 1.25))  # Brighter variation
        darker_colors.append(adjust_brightness(base, 0.75))  # Darker variation
    return th.tensor(lighter_colors),th.tensor(darker_colors)

def show_merging(img1,merge_root,save_root,merge_ix,max_spix_prev,iframe,method="bist"):

    # -- config output --
    # if save_root.exists():
    #     shutil.rmtree(str(save_root))
    if not save_root.exists():
        save_root.mkdir(parents=True)

    # -- read --
    print(merge_root)
    spix = csv_to_th(merge_root / ("%05d.csv" % (4*merge_ix)))[None,].cuda()
    proposed = csv_to_th(merge_root / ("%05d.csv" % (4*merge_ix+1)))[:,0].cuda()
    accepted_spix = csv_to_th(merge_root / ("%05d.csv" % (4*merge_ix+2)))[None,].cuda()
    # print("num merges: ",th.sum(th.unique(spix[th.where(spix!=accepted_spix)])).item())

    # -- load details --
    merge_details_root = merge_root.parents[1] / "merge_details" / merge_root.name
    details = None
    if merge_details_root.exists():
        details = csv_to_th(merge_details_root/("%05d.csv" % (merge_ix))).cuda()

    # -- list of proposed ones into accepted merges --
    accepted_ref = th.unique(spix[th.where(accepted_spix != spix)])
    accepted_tgt = th.unique(accepted_spix[th.where(accepted_spix != spix)])

    # spix_sq = th.unique(spix[:,45:96,325:356])
    # spix_sq = spix_sq[proposed[spix_sq]==-1]
    # check = th.any(proposed[:,None] == spix_sq[None,:],0)
    # spix_sq = spix_sq[check==0]
    # # -- inspect --
    # if not(details is None):
    #     print(spix_sq)
    #     print(proposed[spix_sq])
    #     check = proposed[:,None] == spix_sq[None,:]
    #     # vals = vals[vals>0]
    #     # ivals = vals[-10:]
    #     # print(ivals)
    #     # merge_ref = spix_to_merge
    #     # merge_tgt = spix_merged_into
    #     # print(merge_ref)
    #     # print(merge_tgt)
    #     # exit()

    # -- correct the proposed arrows for the accepted (ref,tgt) pairs --
    # for ref in accepted_ref:
    #     args = th.logical_and(accepted_spix != spix,spix == ref)
    #     tgt = th.unique(accepted_spix[th.where(args)])
    #     assert len(tgt) == 1
    #     tgt = tgt.item()
    #     print(ref,tgt)
    #     # if (ref > max_spix_prev) and (tgt > max_spix_prev): continue
    #     if (tgt > max_spix_prev): continue
    #     proposed[ref] = tgt
    #     proposed[tgt] = -1
    #     proposed[th.where(proposed == ref)] = -1
    # print(proposed[accepted_tgt])
    # print(proposed[accepted_ref])
    # exit()

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
    # print(accepted_ref)
    # print(accepted_tgt)

    # _accepted_tgt = proposed[accepted_ref]
    # for (ref,tgt) in zip(accepted_ref,_accepted_tgt):
    #     _tmp = th.tensor(th.where(proposed == tgt))
    #     elems = th.isin(accepted_tgt,_tmp)
    #     print(elems)
    #     assert len(elems) == 1
    #     elem = elems[0]
    #     proposed[elem] = ref
    #     proposed[ref] = -1
    #     print(ref,tgt)

    # print(max_spix_prev)
    # print(accepted_ref)
    # print(proposed)
    # print(proposed.shape)
    # print(proposed[1514:1540])
    # -- we don't the original direction of the arrows, so we can't index it like this! --
    # accepted_tgt = proposed[accepted_ref]
    # print(th.where(proposed == 1518))
    # print(accepted_tgt)

    # print(accepted_tgt)
    # print(th.all(accepted_tgt>=0))


    # -- get marked img --
    # color = th.tensor([1.0,1.0,1.0])*0.7
    # img1 = get_marked_video(vid[[1]],spix,color)
    # img1 = vid[[1]].cuda()

    # -- get topk (in terms of size) merged positions --
    counts = th.bincount(spix.ravel(),minlength=proposed.shape[0])
    vals,spix_to_merge = th.topk(counts * (proposed >= 0),k=5000)
    # spix_to_merge = spix_to_merge[vals>0]
    # spix_merged_into = th.unique(proposed[spix_to_merge])
    # spix_to_merge = spix_to_merge[vals>0][-3080:-1080] # REMOVE ME!
    spix_to_merge = spix_to_merge[vals>0]
    # print(len(spix_to_merge))
    # # spix_to_merge = spix_to_merge[:len(spix_to_merge)//2]
    # spix_to_merge = spix_to_merge[len(spix_to_merge)//2:]
    spix_merged_into = th.unique(proposed[spix_to_merge])



    # -- colors --
    colors = color_wheel_tensor(len(spix_merged_into))
    dark = th.clamp(colors * 0.75,0.0,1.0)
    light = th.clamp(colors * 1.25,0.0,1.0)
    # light,dark = get_merge_colors()
    # print(dark.shape,light.shape)
    # exit()

    # # -- are you a selected merge? --
    # check_accepted(spix_id,spix,acc)

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
        # iacc = th.where((spix_id == spix).unsqueeze(-1),(1-alpha)*iacc+(alpha)*color,iacc)
        # iprop = th.where((spix_id == spix).unsqueeze(-1),(1-alpha)*iprop+(alpha)*color,iprop)

        # -- dev only --
        # color = th.tensor([0.0,0.0,1.0]).cuda()[None,None,None,:] * th.ones_like(img1)
        # iprop = th.where(th.isin(spix,spix_sq).unsqueeze(-1),(1-alpha)*iprop+(alpha)*color,iprop)

        # -- show proposed references [the source of the arrow] --
        spix_to_merge = th.where(proposed == spix_id)[0]
        color = light[ix,:].cuda()[None,None,None,:] * th.ones_like(img1)
        for spix_id2 in spix_to_merge: # "references" the spix that dissappears
            # if th.any(spix_merged_into == spix_id2): continue # don't color the tgt spix
            if th.all(accepted_tgt != spix_id2): # [spix_id2] is never the target of an accepted pair
                iprop = th.where((spix_id2 == spix).unsqueeze(-1),(1-alpha)*iprop+(alpha)*color,iprop)
            if th.any(accepted_ref == spix_id2):
                iacc = th.where((spix_id2 == spix).unsqueeze(-1),(1-alpha)*iacc+(alpha)*color,iacc)
            # iprop = th.where((spix_id2 == spix).unsqueeze(-1),(1-alpha)*iprop+(alpha)*color,iprop)
            # iacc = th.where((spix_id2 == spix).unsqueeze(-1),(1-alpha)*iacc+(alpha)*color,iacc)



    # -- save --
    color = th.tensor([1.0,1.0,1.0])*0.7
    iprop = get_marked_video(iprop,spix,color)
    save_image(iprop[0],save_root,3*merge_ix,"merge_proposed",iframe,method)
    iacc_m = get_marked_video(iacc,spix,color)
    save_image(iacc_m[0],save_root,3*merge_ix+1,"merge_accepted",iframe,method)
    iacc_m = get_marked_video(iacc,accepted_spix,color)
    save_image(iacc_m[0],save_root,3*merge_ix+2,"merge_accepted",iframe,method)


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
    # proposed = accepted_spix != init_spix
    # proposed_spix = th.where(th.logical_and(cond0,cond1),accepted_spix,proposed_spix) # correct?
    # spix_ids = th.unique(init_spix[th.where(accepted_spix != init_spix)]) # which ones changed

    # tmp = th.where(accepted_spix != init_spix,accepted_spix,proposed_spix) # correct?
    # proposed_spix = th.where(accepted_spix != init_spix,accepted_spix,proposed_spix) # correct?

    # img1 = vid[[1]].clone().cuda()
    # print(proposed_spix)
    # print(accepted_spix)
    max_spix = init_spix.max().item()

    #
    # -- Get the proposed splits --
    #

    # -- split filled regions on image --
    in_img1 = img1.clone()
    alpha = 0.5
    color = th.tensor([0.0,0.0,1.0]).cuda()*0.7
    # img1 = th.where(max_spix_prev>spix.unsqueeze(-1),img1,(1-alpha)*img1+alpha*color)
    # img1 = th.where(max_spix_prev>proposed_spix.unsqueeze(-1),img1,(1-alpha)*img1+alpha*color)
    img1 = th.where(max_spix>proposed_spix.unsqueeze(-1),img1,(1-alpha)*img1+alpha*color)

    # -- get marked img --
    color = th.tensor([1.0,1.0,1.0]).cuda()*0.7
    img1 = get_marked_video(img1,init_spix,color)

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
    img1_m = get_marked_video(img1,init_spix,color)
    save_image(img1_m[0],save_root,3*split_ix+1,"split_accepted",iframe,method)
    img1_m = get_marked_video(img1,accepted_spix,color)
    save_image(img1_m[0],save_root,3*split_ix+2,"split_accepted",iframe,method)
    split_ix+=1

    return split_ix

def show_relabel(img1,read_root,save_root,ix,max_spix_prev,iframe,method="bist"):

    # -- config output --
    if not save_root.exists():
        save_root.mkdir(parents=True)

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
    img1_m = get_marked_video(img1,init_spix,color)
    save_image(img1_m[0],save_root,ix,"relabel",iframe,method)

def animate_bass(img,spix,root):


    #
    # -- Save BASS Boundary Updates --
    #

    frame = 0
    frame_str = "%05d" % frame
    bndy_root = root / "anim_bndy" / frame_str
    merge_root = root/ "anim_merge" / frame_str
    split_root = root/"anim_split" / frame_str

    # -- create update frames --
    read_merge_root = root/"log/merge/"/frame_str
    read_split_root = root/"log/split/" / frame_str
    niters_div_4 = (25-1)//4+1
    max_spix_prev = -1
    for ix in range(niters_div_4):
        show_splitting(img,read_split_root,split_root,ix,max_spix_prev,frame,"bass")
        if ix == (niters_div_4-1): continue
        show_merging(img,read_merge_root,merge_root,ix,max_spix_prev,frame,"bass")
    read_bndy_root = root/"log/bndy/" / frame_str
    show_boundary_updates(img,root,bndy_root,read_bndy_root,frame,"bass")

    # -- writing --
    write_root = root/"anim_bist" / frame_str
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
        read_root = root/("anim_%s"%name)/frame_str
        anim_index = link_seq(read_root,write_root,anim_index,inames,name,starts[name],starts[name]+num)
        starts[name]+=num
    # np.savetxt(write_root/"inames.txt",inames,fmt="%s")
    # anim_index = link_seq(read_root,write_root,anim_index,inames,starts["bndy"],starts["bndy"]+1)
    print(anim_index)
    num_anim_frames = anim_index
    # num_anim_frames = 366

    #
    # -- write movie --
    #

    import imageio
    import imageio.v3 as iio

    # Define video writer
    write_root = root/"anim_bist_mp4" # even tho bass
    if not write_root.exists(): write_root.mkdir()
    fname = write_root / (frame_str+".mp4")
    fps = 30  # Frames per second
    # writer = iio.imopen(fname, "FFMPEG", mode="w", fps=fps)
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
        read_fname = root/"anim_bist" / frame_str / ("%05d.png"%anim_index)
        img = iio.imread(read_fname)

        dur = frame_durations[category]
        if anim_index == (num_anim_frames-1): dur = 1.5
        num_frames = int(dur * fps)  # Convert duration to frame count
        for _ in range(num_frames):
            # writer.write(img)
            writer.append_data(img)
    writer.close()


def create_mp4(dname,vname,pname):

    # -- config --
    # dname = "davis"
    # vname = "bike-packing"
    # vname = "soapbox"
    # vname = "blackswan"
    # vname = "dance-twirl"
    # pname = "param0"
    offset = 0 if dname == "davis" else 1
    # frames = np.arange(30)+1
    # frames = np.arange(1)+1
    # frames = [1,2,3]
    # frames = np.arange(2)+10
    # frames = np.arange(10)
    frames = np.arange(5)
    frames = np.arange(3)
    root = Path("result/davis/bist/logged/%s/%s/"%(pname,vname))

    # -- load video --
    _vid = read_video(dname,vname)
    # _seg = read_seg(dname,vname)
    _spix = read_spix(root,vname,offset)
    _vid = th.from_numpy(_vid).cuda()
    # seg = th.from_numpy(seg)
    _spix = th.from_numpy(_spix).cuda()

    # -- show bass --
    # animate_bass(_vid[[0]],_spix[[0]],root)

    for frame in frames:
        # continue
        if frame == 0: continue

        # break
        # -- patchs with frame --
        frame_str = "%05d" % frame
        bndy_root = root / "anim_bndy" / frame_str
        saf_root = root / "anim_saf" / frame_str
        merge_root = root/ "anim_merge" / frame_str
        split_root = root/"anim_split" / frame_str
        relabel_root = root/"anim_relabel" / frame_str

        # saf_root = Path("result/davis/logged/%s/%05d/"%(vname,frame))/"anim_saf"
        # csv_root = Path("result/davis/logged/%s/%05d/"%(vname,frame))/"anim_csv"
        # saf_root = Path("result/davis/logged/%s/%05d/"%(vname,frame))/"anim_saf"

        # -- index --
        vid = _vid[frame-1:frame+1] # at frame 1 index [0:2]
        # seg = _seg[frame-1:frame+1]
        spix = _spix[frame-1:frame+1]
        max_spix_prev = spix[[0]].max().item()

        # -- show relabel --
        read_relabel_root = root/"log/relabel/"/frame_str
        relabel_ix = 0
        show_relabel(vid[[1]],read_relabel_root,relabel_root,relabel_ix,max_spix_prev,frame)
        relabel_ix = 1
        show_relabel(vid[[1]],read_relabel_root,relabel_root,relabel_ix,max_spix_prev,frame)

        # -- show merging --
        read_merge_root = root/"log/merge/"/frame_str
        merge_ix = 0
        show_merging(vid[[1]],read_merge_root,merge_root,merge_ix,max_spix_prev,frame)
        merge_ix = 1
        show_merging(vid[[1]],read_merge_root,merge_root,merge_ix,max_spix_prev,frame)

        # -- show split --
        read_split_root = root/"log/split/" / frame_str
        split_ix = 0
        show_splitting(vid[[1]],read_split_root,split_root,split_ix,max_spix_prev,frame)
        split_ix = 1
        show_splitting(vid[[1]],read_split_root,split_root,split_ix,max_spix_prev,frame)
        # exit()

        # -- viz show and fill --
        show_shift_and_fill(vid,spix,root,saf_root,frame)

        # -- viz boundary updates --
        read_bndy_root = root/"log/bndy/" / frame_str
        show_boundary_updates(vid[[1]],root,bndy_root,read_bndy_root,frame)

        # -- save bist sequence --
        create_bist_seq(root,frame)

        # -- [optionally] clean up the pngs --

    #
    # -- concat full movie --
    #

    mp4_root = root/"anim_bist_mp4"
    write_fname = (root/"anim_bist_mp4"/"file_names.txt").resolve()
    np.savetxt(write_fname,["file "+str((mp4_root/("%05d.mp4"%i)).resolve()) for i in frames],fmt="%s")

    write_mp4_dir = Path("result/davis/bist_mp4/")
    if not write_mp4_dir.exists():
        write_mp4_dir.mkdir()
    write_mp4 = write_mp4_dir / ("%s.mp4"%vname)
    cmd = "ffmpeg -y -f concat -safe 0 -i %s -c copy %s" % (write_fname,write_mp4.resolve())
    print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

def main():
    dname = "davis"
    pname = "param0"
    vnames = get_video_names(dname)
    # vnames = ["breakdance"]
    vnames = ["kid-football"]
    # vnames = ["car-roundabout"]
    vnames = ["kite-surf","horsejump-high"]
    # vnames = ["bike-packing"]
    # vnames = ["bmx-trees"]
    for vname in vnames:
        create_mp4(dname,vname,pname)


if __name__ == "__main__":
    main()

