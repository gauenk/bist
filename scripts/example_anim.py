import bist
import torch as th

# Select Video & Aesthetic Threshold
vname,img_ext,thresh_new = "kid-football","jpg",0.05
#vname,img_ext,thresh_new = "hummingbird","png",0.1

# Read the video & optical flow
vid_root  = "data/examples/%s/imgs"%vname
flow_root = "data/examples/%s/flows"%vname
spix_root = 'results/%s'%vname

# Run BIST
kwargs = {"n":25,"read_video":True,"iperc_coeff":4.0,"thresh_new":thresh_new,'rgb2lab':True,"nimgs":3,"logging":1,"verbose":True}
bist.run_bin(vid_root,flow_root,spix_root,img_ext,**kwargs)

# Create the "BIST in Action" animation
vid  = bist.utils.read_video(vid_root).cuda()/255.
spix = bist.utils.read_spix(spix_root).cuda()
log_root = 'results/%s/log'%vname
anim_root = 'results/%s/anim'%vname
bist.animate.run(vid,spix,log_root,anim_root,[0,1,2])

