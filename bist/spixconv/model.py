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
from einops import rearrange
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
        kernel_size = self.sconv_kernel_size
        self.in_dim = dim
        self.out_dim = dim
        self.stride = 1
        self.padding = (kernel_size-1)//2
        self.unfold = Unfold(kernel_size=kernel_size,
                             stride=self.stride,padding=self.padding)
        self.linear = nn.Linear(in_chnls*kernel_size *kernel_size, out_chnls)

    def forward(self, x, spix):

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

        # -- reweight with sims --
        if self.conv_type == "sconv":

            # -- get reweight term for kernel --
            sims,scs = None,self.sconv_reweight_source
            if self.conv_type == "sconv" and scs == "sims":
                sims = get_sims(x,spix,self.sims_norm_scale)
            # sims = get_sims(x,spix,self.sims_norm_scale)
            rweight = self.get_reweight(x,sims)
            rweight = apply_norm(rweight,self.sconv_norm_type,self.sconv_norm_scale)

            # -- apply the reweighting term --
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

    def get_reweight(self,ftrs,sims):
        if self.sconv_reweight_source == "sims":
            data,dist_type = sims,"prod"
        elif self.sconv_reweight_source == "ftrs":
            data,dist_type = ftrs,"l2"
        else:
            stype = self.sconv_reweight_source
            raise ValueError(f"Uknown reweight source: {stype}")
        return self._get_reweight(data,dist_type)

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

    def forward(self, x, flow):

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

def get_sims(vid,spix,scale=1.):
    T,F,H,W = vid.shape

    # import st_spix
    # # print("vid.shape, spix.shape: ",vid.shape,spix.shape)
    # _sims = st_spix.models.bass.get_sims(vid,spix[:,None],scale)
    # # print("_sims.shape: ",_sims.shape)
    # return _sims
    # # exit()

    # -- get compact spix ids --
    _,spix = th.unique(spix, return_inverse=True)

    # -- downsample --
    vid = rearrange(vid,'t f h w -> t h w f')
    means,down = bist.get_pooled_video(vid, spix)

    # -- only keep the "down" from this video subsequence --
    spids = th.arange(down.shape[1]).to(vid.device)
    vmask = (spix.unsqueeze(-1) == spids.view(1, 1, 1, 1, -1)).any((0,1,2,3))
    spids = spids[vmask]
    down = down[:,vmask]

    # -- pwd --
    vid = rearrange(vid,'t h w f -> t (h w) f')
    pwd = th.cdist(vid,down)**2 # sum-of-squared differences
    pwd = rearrange(pwd,'t (h w) s -> t s h w',t=T,h=H)

    # -- mask invalid ["empty" spix in down have "0" value] --
    mask = ~(spix.unsqueeze(-1) == spids.view(1, 1, 1, -1)).any((1, 2))
    pwd[mask] = th.inf

    # -- normalize --
    sims = th.softmax(-scale*pwd,1)

    return sims


class SpixConvDenoiser(nn.Module):

    defs = dict(SuperpixelWrapper.defs)
    defs.update(SuperpixelConv.defs)
    _defs = {"dim":6,"net_depth":3,
             "conv_type":"sconv",
             "sconv_reweight_source":"sim",
             "conv_kernel_size":3,
             "sims_norm_scale":10.0, # <- can remove this one
             "use_spixftrs_net":False,"spixftrs_dim":0}
    defs.update(_defs)

    def __init__(self, **kwargs):
        super().__init__()

        # -- unpack --
        extract_self(self,kwargs,self._defs)

        # -- conv layers --
        dim = self.dim
        D = self.net_depth
        conv_ksize = self.conv_kernel_size
        conv_ksize = self.unpack_conv_ksize(conv_ksize,self.net_depth)
        init_conv = lambda d0,d1,ksize: nn.Conv2d(d0,d1,ksize,padding="same")
        self.conv0 = init_conv(3,dim,conv_ksize[0])
        self.conv1 = init_conv(dim,3,conv_ksize[-1])

        # -- learn attn scale --
        self.mid = nn.ModuleList([init_conv(dim,dim,conv_ksize[d+1]) for d in range(D-1)])
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

    def get_spix(self, x, flow):
        # ftrs = self.conv0(x)
        # if self.use_spixftrs_net:
        #     spix_ftrs = self.spixftrs_net(x)
        # else:
        #     spix_ftrs = x
        # spix = self.spix_net(spix_ftrs,flow)
        spix = self.spix_net(x,flow)
        return spix

    def forward(self, x, spix):
        """

        Forward function.

        """

        # -- unpack --
        H,W = x.shape[-2:]

        # -- conv layers --
        ftrs = self.conv0(x)
        if self.net_depth >=1:
            ftrs = ftrs+self.sconv[0](ftrs,spix) # the other slow part
        for d in range(self.net_depth-1):
            ftrs = self.mid[d](ftrs)
            ftrs = ftrs+self.sconv[d+1](ftrs,spix)

        # -- output --
        deno = x + self.conv1(ftrs)
        return deno

    def crop_forward(self,x,spix,size_t,size_s,overlap_s):
        fwd_fxn = self.forward
        deno = run_chunks(fwd_fxn,size_t,size_s,overlap_s,x,spix)
        return deno


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
        return [inp[slices] for inp in inputs]

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
