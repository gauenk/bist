import bist
import torch as th

# Select Video & Aesthetic Threshold
vname,thresh_new = "kid-football",0.05
#vname,thresh_new = "hummingbird",0.1

# Read the video & optical flow
vid = bist.utils.read_video("data/examples/%s/imgs"%vname).cuda()
flows = bist.utils.read_flows("data/examples/%s/flows"%vname).cuda()
# vid.shape = (T,H,W,C)
# flows.shape = (T-1,H,W,2)

# Run BIST
kwargs = {"n":25,"read_video":True,"iperc_coeff":4.0,"thresh_new":thresh_new,'rgb2lab':True}
spix = bist.run(vid,flows,**kwargs)

# Mark the image with the superpixels
color = th.tensor([0.0,0.0,1.0]) # color of border
marked = bist.get_marked_video(vid,spix,color)

# Computer the superpixel-pooled video
pooled,downsampled = bist.get_pooled_video(vid,spix)
# downsampled.shape = (T,MAX_NSPIX,C)

# Save the superpixel info
bist.utils.save_spix(spix, 'results/%s'%vname,"%05d.csv")
bist.utils.save_video(marked, 'results/%s'%vname,"border_%05d.png")
bist.utils.save_video(pooled, 'results/%s'%vname,"pooled_%05d.png")
