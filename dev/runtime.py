
import bist
import torch as th
import numpy as np
from einops import rearrange,repeat
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from natten.functional import na2d_av, na2d_qk_with_bias

import bin.bist_cuda as bist_cuda

def run_spix(x,spix,spids,kernel_size):
    attn = bist.spixconv.model.get_sims_v2(x,spix,spids,10.0)
    return attn

def run_pix_spix(x,spix,spids,spix_read,kernel_size):
    down = bist_cuda.downpool(x,spix)
    pwd = bist_cuda.sparse_pwd(x,down,spix_read)
    return pwd

def run_nat(x,kernel_size):
    dilation = 1
    attn = na2d_qk_with_bias(x, x, None, kernel_size, dilation)
    # attn = rearrange(attn,'t hd h w k -> 1 hd t h w k')
    return attn

def main():

    # -- data --
    # T,H,W,F = 2,256,256,32
    T,H,W,F = 2,480,480,32
    # T,H,W,F = 2,128,128,18
    data = bist.spixconv.davis.get_davis_dataset(T,False)
    flow_fxn = bist.spixconv.spynet.get_flow_fxn()
    video = data.tr[0][0][:,:,:H,:W].cuda()
    fflow,_ = flow_fxn(video)
    _video = repeat(video,'t f h w -> t h w f').contiguous()
    spix = bist.run(_video,fflow)
    print([len(th.unique(s)) for s in spix])
    spids = th.arange(spix.max()).cuda()
    video = repeat(video[:,:1],'t 1 h w -> t f h w',f=F)
    kernel_size = 15
    nreps = 30
    K = kernel_size*kernel_size

    # -- prepare for sparse opt --
    stride = 1
    padding = (kernel_size-1)//2
    unfold = th.nn.Unfold(kernel_size=kernel_size,
                          stride=stride,padding=padding)
    spix_write = rearrange(unfold(spix[:,None]*1.),'t k (h w) -> t k h w',h=H).long()
    # spix_read = spix_write.clone()
    spix_read = spix_write[:,:8].contiguous()

    # -- prep shapes --
    video = rearrange(video,'t f h w -> t h w f').contiguous()
    video_nat = rearrange(video,'t f h w -> t 1 h w f').contiguous()
    spix_read = spix_read.contiguous().int()
    spix_read = rearrange(spix_read,'t k h w -> t h w k').contiguous()

    # _spix_flat = spix_write.permute(0, 2, 3, 1).reshape(-1, K)
    # U = max(th.unique(row).size(0) for row in _spix_flat)
    # spix_read = -th.ones((_spix_flat.shape[0],U)).to(video.device)
    # for i in range(_spix_flat.size(0)):
    #     uniques = th.unique(_spix_flat[i])
    #     spix_read[i, :uniques.size(0)] = uniques
    # spix_read = spix_read.view(T, H, W, U).permute(0, 3, 1, 2)  # (T, U, H, W)
    # print(spix_write.shape)
    # print(spix_read.shape)
    # print(spix_write[0,:,64,64])
    # print(spix_read[0,:,64,64])
    # exit()
    th.cuda.empty_cache()


    # -- init --
    run_spix(video,spix,spids,kernel_size)
    run_pix_spix(video,spix,spids,spix_read,kernel_size)
    run_nat(video_nat,kernel_size)
    th.cuda.empty_cache()

    # -- spix --
    timer = ExpTimer()
    for i in range(nreps):
        with TimeIt(timer,'%d'%i):
            run_spix(video,spix,spids,kernel_size)
        th.cuda.empty_cache()
    times = np.array([timer["%d"%i] for i in range(10)])
    print("spix: ",np.median(times),times.std())

    # -- pix-spix --
    timer = ExpTimer()
    for i in range(nreps):
        with TimeIt(timer,'%d'%i):
            run_pix_spix(video,spix,spids,spix_read,kernel_size)
        th.cuda.empty_cache()
    times = np.array([timer["%d"%i] for i in range(10)])
    pixspix_m = np.median(times)
    print("pix-spix: ",np.median(times),times.std())

    # -- nat --
    timer = ExpTimer()
    for i in range(nreps):
        with TimeIt(timer,'%d'%i):
            run_nat(video_nat,kernel_size)
        th.cuda.empty_cache()
    times = np.array([timer["%d"%i] for i in range(10)])
    nat_m = np.median(times)
    print("nat: ",np.median(times),times.std())
    print("ratio: ",nat_m/pixspix_m)


if __name__ == "__main__":
    main()
