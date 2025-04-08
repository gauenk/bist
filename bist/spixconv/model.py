"""

   Conv Denoising Network

"""

# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from collections import OrderedDict

# -- utils --
import bist
from bist.utils import extract,extract_self
from torch.nn import Unfold

# -- neighborhood search --
try:
    import stnls # see github.com/gauenk/stnls
except:
    pass


class SuperpixelConv(nn.Module):

    defs = {"conv_type":"sconv",
            "sims_norm_scale":10.0,
            "sconv_kernel_size":7,
            "sconv_reweight_source":"sims",
            # -- normalization --
            "sconv_norm_type":"max",
            "sconv_norm_scale":0.0,
            "kernel_norm_type":"none",
            "kernel_norm_scale":0.0}

    def __init__(self, dim, **kwargs):
        super().__init__()

        # -- init --
        extract_self(self,kwargs,self.defs)
        in_chnls = dim
        out_chnls = dim
        stride = 1
        kernel_size = self.sconv_kernel_size
        self.in_dim = dim
        self.out_dim = dim
        self.stride = stride
        self.padding = (kernel_size-1)//2

        def with_pads(video):
            pad = (kernel_size-1)//2
            video = F.pad(video, (pad, pad, pad, pad), mode='reflect')
            return video
        self.with_pads = with_pads
        def without_pads(video):
            return video[...,pad:-pad,pad:-pad]
        self.without_pads = without_pads

        def unfold_reflect(video):
            unfold = Unfold(kernel_size=kernel_size,
                            stride=stride,padding=0)
            pad = (kernel_size-1)//2
            video = F.pad(video, (pad, pad, pad, pad), mode='reflect')
            return unfold(video)
        self.unfold = unfold_reflect
        # self.unfold = Unfold(kernel_size=kernel_size,
        #                      stride=self.stride,padding=self.padding)
        self.linear = nn.Linear(in_chnls*kernel_size *kernel_size, out_chnls)
        self.scale_net = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1)
        self.scale_act = nn.Softplus()
        nn.init.kaiming_uniform_(self.scale_net.weight, a=0.1)
        self.scale_net.bias.data.zero_()
        self.rweight_mean = 0
        self.rweight_std = 0

    def forward(self, x, spix):
    # def forward(self, x, sims, pix):

        # Extract patches and flatten them
        patches = self.unfold(x)  # Shape: (batch_size, in_channels * kernel_size * kernel_size, L)
        patches = patches.transpose(1, 2)  # Shape: (batch_size, L, in_channels * kernel_size * kernel_size)

        # -- unpack weights --
        batchsize = len(x)
        kernel = self.linear.weight
        out_dim,total_ksize = kernel.shape
        kernel = kernel.unsqueeze(0).expand(batchsize,out_dim,total_ksize)
        bias = self.linear.bias
        kernel_size = self.sconv_kernel_size

        # -- get scale --
        scs = self.sconv_reweight_source
        scale = 10.*self.scale_act(self.scale_net(x/(x.norm(dim=1,keepdim=True)+1e-10)))
        # x_max = th.amax(x,(1,2,3),keepdim=True)
        # print(x_max.shape,x.shape)
        # exit()
        # scale = 10.*self.scale_act(self.scale_net(x/x_max+1e-10))
        # scale = 30.*self.scale_act(self.scale_net(x/x_max+1e-10))
        # if scs == "ftrs":
        #     scale = 10.*self.scale_act(self.scale_net(x/(x.norm(dim=1,keepdim=True)+1e-10)))
        # else:
        #     scale = 10.
        #     # scale = 50.

        # -- reweight with sims --
        if self.conv_type == "sconv":

            # -- get reweight term for kernel --
            # sims,scs = None,self.sconv_reweight_source
            scs = self.sconv_reweight_source
            ntype,nscale = self.sconv_norm_type,self.sconv_norm_scale
            # if self.conv_type == "sconv" and "sims" in scs:
            #     sims = get_sims(x,spix,self.sims_norm_scale)
            #     # sims = get_sims_v1(x,spix,self.unfold,self.sims_norm_scale)
            #     # print(sims.shape)

            if scs == "sims+ftrs":
                rweight_s = self.get_reweight(x,sims,"sims")
                # rweight_s = apply_norm(rweight_s,"max",0.0)
                # rweight_s = rearrange(sims,'t k h w -> t 1 1 h w k')
                # rweight_s = apply_norm(rweight_s,ntype,nscale)
                # rweight_f = self.get_reweight(x,sims,"ftrs")
                # rweight_f = apply_norm(rweight_f,"exp_max",nscale)
                # rweight_s = rearrange(sims,'t k h w -> t 1 1 h w k')
                # print("s: ",rweight_s[0,0,0,64,64])
                rweight_f = self.get_reweight(x,sims,"ftrs")
                # print("f: ",rweight_f[0,0,0,64,64])
                rweight = rweight_f * rweight_s
                rweight = apply_norm(rweight,"exp_max",nscale)
                # print("c: ",rweight[0,0,0,64,64])
            elif scs == "sims":
                sims = get_sims_v1(x,spix,self.unfold,scale)
                sims = rearrange(sims,'t k h w -> t 1 1 h w k')
                scale = rearrange(scale,'t 1 h w -> t 1 1 h w 1')
                rweight = apply_norm(sims,"exp_max",scale)
            elif scs == "pftrs":
                means,_ = bist.get_pooled_video(x, spix, cdim=1)
                attn = self._run_attn(x,means,"l2")
                scale = rearrange(scale,'t 1 h w -> t 1 1 h w 1')
                rweight = apply_norm(attn,"exp_max",scale)
            elif scs == "ftrs":
                # rweight = self.get_reweight(x,None,"ftrs")
                rweight = self._run_attn(x,x,"l2")
                scale = rearrange(scale,'t 1 h w -> t 1 1 h w 1')
                # scale = nscale * scale
                # print(rweight.shape,scale.shape)
                # exit()
                rweight = apply_norm(rweight,ntype,scale)
            else:
                raise ValueError(f"Uknown reweight term [{scs}]")
            # rweight = apply_norm(rweight,self.sconv_norm_type,self.sconv_norm_scale)

            # -- apply the reweighting term --
            self.rweight_mean = rweight.mean().item()
            self.rweight_std = rweight.std().item()
            in_dim,out_dim = self.in_dim,self.out_dim
            ksize2 = kernel_size*kernel_size
            kernel = kernel.reshape(batchsize,out_dim,in_dim,ksize2)
            kernel = rearrange(kernel,'b od id k -> b od id 1 1 k')
            kernel = kernel * rweight
            kernel = rearrange(kernel,'b od id h w k -> b od (h w) (id k)')

            # -- renormalize each kernel --
            kernel = apply_norm(kernel,self.kernel_norm_type,self.kernel_norm_scale)

            # -- apply kernel --
            out = th.sum(patches.unsqueeze(1) * kernel,-1).transpose(1,2) + bias

        else:

            # -- apply kernel --
            out = th.bmm(patches,kernel.transpose(1,2)) + bias

        # Reshape back to (batch_size, out_channels, H_out, W_out)
        batch_size, num_patches, _ = out.shape
        H_out = int((x.size(2) + 2 * self.padding - kernel_size) / self.stride + 1)
        W_out = int((x.size(3) + 2 * self.padding - kernel_size) / self.stride + 1)

        out = out.transpose(1, 2).reshape(batch_size, -1, H_out, W_out)
        return out

    def get_reweight(self,ftrs,sims,source):
        if source == "sims":
            data,dist_type = sims,"prod"
        elif source == "ftrs":
            data,dist_type = ftrs,"l2"
        else:
            stype = self.sconv_reweight_source
            raise ValueError(f"Uknown reweight source: {stype}")
        # return self._get_reweight(data,dist_type)
        return self._run_attn(data,dist_type)

    def _run_attn(self,x,y,dist_type):
        # -- compute \sum_s p(s_i=s)p(s_j=s) --
        ws = self.sconv_kernel_size
        x = self.with_pads(x)[None,:].contiguous()
        y = self.with_pads(y)[None,:].contiguous()
        NonLocalSearch = stnls.search.NonLocalSearch
        search = NonLocalSearch(ws,0,dist_type=dist_type,
                                itype="int",full_ws=False,
                                reflect_bounds=True)
        T,B,F,H,W = x.shape
        flows = th.zeros((B,1,T,1,2,H,W),device=x.device)
        attn = search(x,y,flows)[0]
        pad = self.padding
        attn = attn[...,pad:-pad,pad:-pad,:] # remove pads
        attn = rearrange(attn,'1 1 t h w k -> t 1 1 h w k')
        return attn

    def _get_reweight(self,tensor,dist_type):
        # -- compute \sum_s p(s_i=s)p(s_j=s) --
        ws = self.sconv_kernel_size
        tensor = tensor[None,:].contiguous()
        if dist_type == "l2":
            search = stnls.search.NonLocalSearch(ws,0,dist_type=dist_type,itype="int")
        else: # replace with NATTEN
            search = stnls.search.NonLocalSearch(ws,0,dist_type=dist_type,itype="int")
        T,B,F,H,W = tensor.shape
        flows = th.zeros((B,1,T,1,2,H,W),device=tensor.device)
        assert not(th.any(th.isnan(tensor)).item()),"[0] Must be no nan."
        attn = search(tensor,tensor,flows)[0]
        attn = rearrange(attn,'1 1 t h w k -> t 1 1 h w k')
        return attn


def apply_norm(tensor,norm_type,norm_scale):
    if norm_type == "sum":
        tensor = tensor/(1e-5+tensor.sum(-1,keepdim=True))
    elif norm_type == "max":
        tensor = tensor/(1e-5+tensor.max(-1,keepdim=True).values)
    elif norm_type == "sm":
        tensor = th.softmax(-norm_scale*tensor,-1)
    elif norm_type == "exp_max":
        tensor = th.exp(-norm_scale*tensor)
        tensor = tensor/(1e-5+tensor.max(-1,keepdim=True).values)
    elif norm_type == "none":
        pass
    else:
        raise ValueError(f"Uknown normalization method [{norm_type}]")
    return tensor

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#         Wrapper to compute superpixels
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class SuperpixelWrapper(nn.Module):
    defs = {"spix_method":"bist"}
    defs.update(bist.default_params())

    def __init__(self, **kwargs):
        super().__init__()
        extract_self(self,kwargs,self.defs)
        self.proj = SpixFeatureProjection(3)
        self.spix_params = extract(kwargs,bist.default_params())
        self.spix_params['video_mode'] = self.spix_method == "bist"

    def forward(self, x, flow, rgb2lab=False):

        # -- all pixels are an spix :D --
        if self.spix_method == "exh":
            T,_,H,W = x.shape
            spix = th.arange(H*W).view(1,H,W).repeat(T,1,1).to(x.device)
            return spix

        # -- unpack --
        B,F,H,W = x.shape
        with th.no_grad():

            # -- prepare video --
            y = self.proj(x)
            y = rearrange(y,'t c h w -> t h w c')
            if self.spix_params['rgb2lab'] is True:
                y = y - y.mean((1,2),keepdim=True)
                y = y/y.std((1,2),keepdim=True)
            y = y.contiguous()
            # print(self.spix_params)
            # exit()

            # -- run spix --
            self.spix_params['rgb2lab'] = rgb2lab
            spix = bist.run(y,flow,**self.spix_params)

            # -- alt spix --
            # import bist_cuda
            # flow = th.clamp(flow,-25,25)
            # fxn = bist_cuda.bist_forward
            # spix = fxn(y,flow,20,15,1.0,0.1,0.0,2.0,0,True,False)

        return spix


class SpixFeatureProjection(nn.Module):

    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.svd_with_batch = False

    def forward(self,x):

        # -- edge case --
        if x.shape[1] == self.out_dim:
            return x

        # -- unpack --
        out_dim = self.out_dim
        B,F,H,W = x.shape
        if self.svd_with_batch:
            x = rearrange(x,'b f h w -> 1 (b h w) f')
        else:
            x = rearrange(x,'b f h w -> b (h w) f')

        # -- normalize --
        x = x - x.mean(-1,keepdim=True)
        x = x/x.std(-1,keepdim=True)

        # -- svd --
        U, S, V = th.linalg.svd(x, full_matrices=True)
        U_k = U[:, :, :out_dim]  # First k columns of U
        S_k = S[:,:out_dim]     # First k singular values
        V_k = V[:,:, :out_dim]  # First k columns of V

        # -- reconstruct --
        xr = th.bmm(U_k, th.diag_embed(S_k))  # Reduced features

        # -- reshape --
        if self.svd_with_batch:
            xr = rearrange(xr,'1 (b h w) f -> b f h w',b=B,h=H)
        else:
            xr = rearrange(xr,'b (h w) f -> b f h w',h=H)

        return xr




# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#       Feature Extraction for Re-weight Term
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SimNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32, softmax=True):
        super(SimNet, self).__init__()
        self.softmax = softmax
        features = init_features
        self.encoder1 = SimNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = SimNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = SimNet._block(features * 2, features * 4, name="bottleneck")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = SimNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = SimNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # -- encode --
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)
        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            padding_mode='replicate'
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=False)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            padding_mode='replicate'
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=False)),
                ]
            )
        )



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#                  Main Network
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def expand_ndarrays(size,*ndarrays):
    out = []
    for ndarray in ndarrays:
        ndarray_e = -np.ones(size)
        ndarray_e[:len(ndarray)] = ndarray
        out.append(ndarray_e)
    return out

def get_sims_v1(vid,spix,unfold,scale=1.):

    # -- get compact spix ids --
    T,F,H,W = vid.shape
    # _,spix = th.unique(spix, return_inverse=True)

    # -- downsample --
    vid = rearrange(vid,'t f h w -> t h w f')
    means,down = bist.get_pooled_video(vid, spix)

    # -- only keep the "down" from this video subsequence --
    spids = th.arange(down.shape[1]).to(vid.device)
    comp = spix.unsqueeze(-1) == spids.view(1, 1, 1, -1)
    vmask = comp.any((0,1,2))
    spids = spids[vmask]
    down = down[:,vmask]

    # -- pwd --
    pwd = th.cdist(down,down)**2 # sum-of-squared differences
    tgrid = th.arange(down.shape[0]).unsqueeze(1).unsqueeze(2)
    pwd = pwd[tgrid,spix]
    pwd = rearrange(pwd,'t h w s -> t s h w')

    # -- valid indexing strat --
    # rows, cols = th.meshgrid(th.arange(H), th.arange(W), indexing='ij')
    # rows, cols = rows.cuda(), cols.cuda()
    # inds = th.stack([rows,cols],0)[None,:]
    # inds = rearrange(unfold(inds*1.),'t (f k) (h w) -> t h w k f',h=H,f=2)
    # print(inds[0,64,64])
    # print(inds[0,65,64,K//2])

    # -- kernel indexing --
    inds = rearrange(unfold(spix[:,None]*1.),'t k (h w) -> t k h w',h=H).long()
    pwd = th.gather(pwd,dim=1,index=inds)
    # mask = comp.any((1, 2),keepdim=True).repeat(1,H,W,1)
    # mask = rearrange(mask,'t h w s -> t s h w')
    # mask = th.gather(mask,dim=1,index=inds)

    # -- normalize --
    # sims = mask*th.exp(-scale*pwd)
    # sims = th.exp(-scale*pwd)

    # return sims
    return pwd


def get_sims_v2(vid,spix,spids,scale=1.0):

    # -- get compact spix ids --
    T,F,H,W = vid.shape
    # _,spix = th.unique(spix, return_inverse=True)

    # -- downsample --
    # vid = rearrange(vid,'t f h w -> t h w f')
    down = bist.get_pooled_video(vid, spix, return_pool=False)

    # -- only keep the "down" from this video subsequence --
    # spids = th.arange(down.shape[1]).to(vid.device)
    # vmask = (spix.unsqueeze(-1) == spids.view(1, 1, 1, -1)).any((0,1,2))
    # spids = spids[vmask]
    # down = down[:,vmask]

    # -- pwd --
    pwd = th.cdist(down,down)**2 # sum-of-squared differences
    # tgrid = th.arange(down.shape[0]).unsqueeze(1).unsqueeze(2)
    # pwd = pwd[tgrid,spix]
    # pwd = rearrange(pwd,'t h w s -> t s h w')

    # -- mask invalid ["empty" spix in down have "0" value] --
    # mask = ~(spix.unsqueeze(-1) == spids.view(1, 1, 1, -1)).any((1, 2))
    # # print(pwd.shape,spix.shape,spids.shape,mask.shape)
    # pwd[mask] = th.inf

    pwd = th.exp(-scale*pwd)
    return pwd

def get_sims_v3(vid,spix,scale=1.0):
    # -- get compact spix ids --
    T,F,H,W = vid.shape
    # _,spix = th.unique(spix, return_inverse=True)

    # -- downsample --
    vid = rearrange(vid,'t f h w -> t h w f')
    means,down = bist.get_pooled_video(vid, spix)

    # -- only keep the "down" from this video subsequence --
    spids = th.arange(down.shape[1]).to(vid.device)
    vmask = (spix.unsqueeze(-1) == spids.view(1, 1, 1, -1)).any((0,1,2))
    spids = spids[vmask]
    down = down[:,vmask]

    # -- pwd --
    pwd = th.cdist(down,down)**2 # sum-of-squared differences
    tgrid = th.arange(down.shape[0]).unsqueeze(1).unsqueeze(2)
    pwd = pwd[tgrid,spix]
    pwd = rearrange(pwd,'t h w s -> t s h w')

    # -- mask invalid ["empty" spix in down have "0" value] --
    mask = ~(spix.unsqueeze(-1) == spids.view(1, 1, 1, -1)).any((1, 2))
    # # print(pwd.shape,spix.shape,spids.shape,mask.shape)
    pwd[mask] = th.inf

    # -- normalize --
    sims = th.exp(-scale*pwd)
    return sims


def get_sims(vid,spix,scale=1.):

    # -- get compact spix ids --
    T,F,H,W = vid.shape
    _,spix = th.unique(spix, return_inverse=True)

    # -- downsample --
    vid = rearrange(vid,'t f h w -> t h w f')
    means,down = bist.get_pooled_video(vid, spix)

    # -- only keep the "down" from this video subsequence --
    spids = th.arange(down.shape[1]).to(vid.device)
    comp = spix.unsqueeze(-1) == spids.view(1, 1, 1, -1)
    vmask = comp.any((0,1,2))
    spids = spids[vmask]
    down = down[:,vmask]

    # -- pwd --
    vid = rearrange(vid,'t h w f -> t (h w) f')
    pwd = th.cdist(vid,down)**2 # sum-of-squared differences
    pwd = rearrange(pwd,'t (h w) s -> t s h w',t=T,h=H)

    # -- mask invalid ["empty" spix in down have "0" value] --
    # mask = ~comp.any((1,2))
    # # mask = ~(comp).any((1, 2))
    # pwd[mask] = th.inf

    # -- mask --
    mask = comp.any((1,2),keepdim=True).repeat(1,H,W,1)
    mask = rearrange(mask,'t h w s -> t s h w')
    pwd = pwd.masked_fill(mask == 0, float('-inf'))

    # -- normalize --
    sims = mask*th.softmax(-scale*pwd,1)

    return sims


class SpixConvNetwork(nn.Module):

    defs = dict(SuperpixelWrapper.defs)
    defs.update(SuperpixelConv.defs)
    _defs = {"dim":6,"net_depth":3,"out_dims":3,
             "conv_type":"sconv",
             "sconv_reweight_source":"sim",
             "sconv_kernel_size":7,
             "conv_kernel_size":3,
             "sims_norm_scale":10.0, # <- can remove this one
             "use_spixftrs_net":False,"spixftrs_dim":0,
             "task":"deno"}
    defs.update(_defs)

    def __init__(self, **kwargs):
        super().__init__()

        # -- unpack --
        extract_self(self,kwargs,self._defs)

        # -- conv layers --
        dim = self.dim
        out_dims = 3 if self.task == "deno" else 1
        D = self.net_depth
        conv_ksize = self.conv_kernel_size
        conv_ksize = self.unpack_conv_ksize(conv_ksize,self.net_depth)
        init_conv = lambda d0,d1,ksize,g: nn.Conv2d(d0,d1,ksize,padding="same",groups=g)
        self.conv0 = init_conv(3,dim,conv_ksize[0],3)
        self.conv1 = init_conv(dim,out_dims,conv_ksize[-1],1)

        # -- learn attn scale --
        self.mid0 = nn.ModuleList([init_conv(dim,dim,conv_ksize[d+1],dim) for d in range(D-1)])
        self.mid1 = nn.ModuleList([init_conv(dim,dim,conv_ksize[d+1],dim) for d in range(D-1)])
        akwargs = extract(kwargs,SuperpixelConv.defs)
        self.sconv = nn.ModuleList([SuperpixelConv(dim,**akwargs) for _ in range(D)])

        # -- superpixel network with projection --
        kwargs = extract(kwargs,SuperpixelWrapper.defs)
        self.spix_net = SuperpixelWrapper(**kwargs)

        # -- superpixel feature network --
        if self.use_spixftrs_net:
            self.spixftrs_net = SimNet(out_channels=self.spixftrs_dim)
        else:
            self.spixftrs_net = nn.Identity()

    def unpack_conv_ksize(self,ksize,depth):
        if hasattr(ksize,"__len__"):
            if len(ksize) == 1:
                ksize = ksize*(depth+1)
            else:
                assert len(ksize) == (depth+1),"Must be equal."
                return ksize
        else:
            return [ksize,]*(depth+1)

    def get_spix(self, x, flow, rgb2lab=False):
        if flow.shape[1] == 2:
            flow = rearrange(flow,'t c h w -> t h w c')
        spix = self.spix_net(x,flow,rgb2lab)
        _,spix = th.unique(spix, return_inverse=True) # compactify
        return spix

    def forward(self, x, spix):
        """

        Forward function.

        """

        # -- unpack --
        H,W = x.shape[-2:]
        spix = th.unique(spix, return_inverse=True)[1] # compactify

        # -- conv layers --
        ftrs = self.conv0(x)
        if self.net_depth >=1:
            ftrs = ftrs+self.sconv[0](ftrs,spix)
        for d in range(self.net_depth-1):
            ftrs = self.mid0[d](ftrs)+ftrs
            ftrs = self.pooling_layer(ftrs,spix,self.mid1[d])
            ftrs = ftrs+self.sconv[d+1](ftrs,spix)

        # -- output --
        if self.task == "deno":
            out = x + self.conv1(ftrs)
        else:
            out = self.conv1(ftrs)
            out = self.apply_pooling_layer(deno,spix_x)[:,0]
        return out

    def crop_forward(self,size_t,size_s,overlap_s,*inputs):
        fwd_fxn = self.forward
        deno = run_chunks(fwd_fxn,size_t,size_s,overlap_s,*inputs)
        return deno

    def apply_pooling_layer(self,vid,spix):
        vid = rearrange(vid,'t f h w -> t h w f')
        means,down = bist.get_pooled_video(vid, spix)
        means = rearrange(means,'t h w f -> t f h w')
        return means

    def pooling_layer(self,vid,spix,xform_layer):
        xform = xform_layer(vid)
        xform = rearrange(xform,'t f h w -> t h w f')
        means,down = bist.get_pooled_video(xform, spix)
        means = rearrange(means,'t h w f -> t f h w')
        return means + vid

    def get_rweight_stats(self):
        means,stds = [],[]
        for sconv in self.sconv:
            means.append(sconv.rweight_mean)
            stds.append(sconv.rweight_std)
        return np.array(means),np.array(stds)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#       Run model on spatial-time crops of the video
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# -- simpler one --
def run_chunks(fwd_fxn,size_t,size_s,overlap_s,*inputs):

    # -- unpack --
    device = inputs[0].device
    shape = inputs[0].shape
    T,F,H,W = shape

    # -- alloc --
    deno = th.zeros(shape,device=device)
    count = th.zeros((T,1,H,W),device=device)

    # -- get chunks --
    t_chunks = get_chunks(T,size_t,0.0)
    h_chunks = get_chunks(H,size_s,overlap_s)
    w_chunks = get_chunks(W,size_s,overlap_s)

    def chunks_em(*inputs):
        slices = (slice(t_chunk,t_chunk+size_t),
                  Ellipsis,
                  slice(h_chunk,h_chunk+size_s),
                  slice(w_chunk,w_chunk+size_s))
        return [inp[slices].contiguous() for inp in inputs]

    def add_chunk(agg,inp,t_chunk,h_chunk,w_chunk,sizeT,sizeH,sizeW):
        sizeH,sizeW = deno_chunk.shape[-2:]
        slices = (slice(t_chunk,t_chunk+sizeT),
                  Ellipsis,
                  slice(h_chunk,h_chunk+sizeH),
                  slice(w_chunk,w_chunk+sizeW))
        agg[slices] += inp

    # -- loop --
    tt,hh,ww = np.meshgrid(t_chunks,h_chunks,w_chunks,indexing='ij')
    for chunks in zip(tt.ravel(), hh.ravel(), ww.ravel()):

        # -- unpack --
        t_chunk,h_chunk,w_chunk = chunks

        # -- forward --
        chunks = chunks_em(*inputs)
        deno_chunk = fwd_fxn(*chunks)

        # -- fill --
        sT,_,sH,sW = deno_chunk.shape
        add_chunk(deno,deno_chunk,t_chunk,h_chunk,w_chunk,sT,sH,sW)
        add_chunk(count,1,t_chunk,h_chunk,w_chunk,sT,sH,sW)

    # -- normalize --
    deno = deno / count
    return deno


def get_chunks(size,chunk_size,overlap):
    """

    Thank you to https://github.com/Devyanshu/image-split-with-overlap/

    args:
      size = original size
      chunk_size = size of output chunks
      overlap = percent (from 0.0 - 1.0) of overlap for each chunk

    This code splits an input size into chunks to be used for
    split processing

    Will overlap at the bottom-right to ensure all chunks are size "chunk_size"

    """
    overlap = handle_int_overlap(size,overlap)
    points = [0]
    stride = max(int(chunk_size * (1-overlap)),1)
    if size <= chunk_size: return [0]
    assert stride > 0
    counter = 1
    while True:
        pt = stride * counter
        if pt + chunk_size >= size:
            points.append(size - chunk_size)
            break
        else:
            points.append(pt)
        counter += 1
    points = list(np.unique(points))
    return points

def handle_int_overlap(size,overlap):
    if overlap >= 0 and overlap < 1:
        return overlap
    elif overlap >= 1:
        if isinstance(overlap,int):
            return (1.*overlap)/size
    else:
        raise ValueError("Uknown behavior for overlap as a float greater than 1 or less than 0.")
