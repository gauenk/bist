
import tqdm
import math

import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange

import seaborn as sns
import matplotlib.pyplot as plt

import glob
from pathlib import Path
from run_eval import read_video,read_seg,get_video_names
from st_spix.utils import rgb2lab
from st_spix.spix_utils.evaluate import computeSummary,scoreSpixPoolingQualityByFrame,count_spix,read_spix


def read_cache(cache_root,dname,method):
    cache_fn = cache_root / ("%s_%s.csv"%(dname,method))
    # print(cache_fn,cache_fn.exists())
    if not cache_fn.exists(): return None
    else: return pd.read_csv(cache_fn)

def save_cache(summs,cache_root,dname,method):
    if not cache_root.exists():cache_root.mkdir(parents=True)
    cache_fn = cache_root / ("%s_%s.csv"%(dname,method))
    # print(summs)
    # print(pd.DataFrame(summs))
    # exit()
    pd.DataFrame(summs).to_csv(cache_fn)



def controlled_tex():

    # -- get results --
    dname = "davis"
    vnames = ["bmx-trees"]
    pnames = ["param%d"%i for i in range(200,200+12)]
    df = eval_experiments(dname,vnames,pnames,refresh=False)

    # -- summary stats --
    df = df[df['param'].isin(["param%d" % i for i in range(200,212)])]
    df = df.reset_index(drop=True)
    df = df.drop(["name","method","param"],axis=1)
    df_mean = df.groupby(["thresh_new"]).mean().reset_index()
    df_std = df.groupby(["thresh_new"]).std().reset_index()

    # -- unpack --
    xgrid = df_mean['thresh_new']
    ave_tex_m = df_mean['tex']
    ave_tex_s = df_std['tex']#/math.sqrt(3)
    print(xgrid)
    print(ave_tex_m)
    print(ave_tex_s)

    # Sample data with uncertainties
    data = pd.DataFrame({
        'thresh': xgrid,
        'tex': ave_tex_m,
        'std_dev': ave_tex_s
    })

    # Define pastel colors
    # colors = ['#6EA4BF', '#B48EEC', '#FF8BA0']  # Vibrant pastel blue, purple, and red
    colors = ['#6EA4FF', '#B48EFF', '#FF8BA0']  # Vibrant pastel blue, purple, and red
    shadow_color = '#555555'  # Gray shadow color

    # Create the plot
    ginfo = {'wspace':0.01, 'hspace':0.01,
             "top":0.88,"bottom":0.20,"left":.15,"right":0.98}
    fig, ax = plt.subplots(figsize=(6, 3),gridspec_kw=ginfo,dpi=200)

    # Convert categorical x values to numerical positions
    x_positions = np.arange(len(data))

    # Plot shadow bars (slightly offset and larger)
    # for i, y in enumerate(data['Average # SPIX']):
    #     ax.bar(x_positions[i], y, color=shadow_color, alpha=0.4, width=0.6, zorder=1)

    # Seaborn bar plot (main bars)
    bars = sns.barplot(x='thresh', y='tex', data=data, palette=colors, ax=ax, width=0.4, zorder=2)

    # Add right-side shadow effect
    for bar in bars.patches:
        x = bar.get_x() + bar.get_width() - 0.008  # Shift slightly to the right
        y = bar.get_y() - 0.0008
        width = 0.008  # Thin shadow width
        height = bar.get_height()
        ax.add_patch(plt.Rectangle((x, y), width, height, color=shadow_color, alpha=1.0, zorder=1))

    # Add error bars manually using Matplotlib
    for i, (y, err) in enumerate(zip(data['tex'], data['std_dev'])):
        plt.errorbar(i, y, yerr=err, fmt='none', ecolor='black', capsize=5, capthick=1.5, linewidth=1.5, zorder=3)

    # Labels with LaTeX formatting
    ymax = ax.get_ylim()[1]
    ax.set_title("Controlling the Lifespan of Superpixels",fontsize=16)
    ax.set_ylim([0,ymax])
    ax.set_yticklabels(["%d%%"%(100*y) for y in ax.get_yticks()])
    plt.xlabel(r"Relabeling Threshold $(\varepsilon_{\text{new}})$", fontsize=16)
    plt.ylabel("Temporal Extent (TEX)", fontsize=16)

    # Adjust tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # -- save --
    save_root = Path("output/plots")
    if not save_root.exists():
        save_root.mkdir(parents=True)
    fname = save_root / "controlled_tex.png"
    plt.savefig(fname,transparent=True)
    plt.close("all")

def controlled_nspix():

    # -- get results --
    dname = "davis"
    vnames = ["bmx-trees"]
    pnames = ["param%d"%i for i in range(400,400+3*3)]
    df = eval_experiments(dname,vnames,pnames,refresh=False)

    # -- summary stats --
    df = df[df['param'].isin(["param%d" % i for i in range(400,430)])]
    df = df.reset_index(drop=True)
    df = df.drop(["name","method","param"],axis=1)
    df_mean = df.groupby(["iperc_coeff"]).mean().reset_index()
    df_std = df.groupby(["iperc_coeff"]).std().reset_index()

    # -- unpack --
    iperc = df_mean['iperc_coeff']
    ave_nspix_m = df_mean['ave_nsp']
    ave_nspix_s = df_std['ave_nsp']#/math.sqrt(3)
    print(iperc)
    print(ave_nspix_m)
    print(ave_nspix_s)

    # Sample data with uncertainties
    data = pd.DataFrame({
        'gamma': iperc,
        'Average # SPIX': ave_nspix_m,
        'std_dev': ave_nspix_s
    })

    # Define pastel colors
    # colors = ['#6EA4BF', '#B48EEC', '#FF8BA0']  # Vibrant pastel blue, purple, and red
    colors = ['#6EA4FF', '#B48EFF', '#FF8BA0']  # Vibrant pastel blue, purple, and red
    shadow_color = '#555555'  # Gray shadow color

    # Create the plot
    ginfo = {'wspace':0.01, 'hspace':0.01,
             "top":0.88,"bottom":0.20,"left":.15,"right":0.98}
    fig, ax = plt.subplots(figsize=(6, 3),gridspec_kw=ginfo,dpi=200)

    # Convert categorical x values to numerical positions
    x_positions = np.arange(len(data))

    # Plot shadow bars (slightly offset and larger)
    # for i, y in enumerate(data['Average # SPIX']):
    #     ax.bar(x_positions[i], y, color=shadow_color, alpha=0.4, width=0.6, zorder=1)


    # Seaborn bar plot (main bars)
    bars = sns.barplot(x='gamma', y='Average # SPIX', data=data, palette=colors, ax=ax, width=0.4, zorder=2)

    # Add right-side shadow effect
    for bar in bars.patches:
        x = bar.get_x() + bar.get_width() - 0.08  # Shift slightly to the right
        y = bar.get_y() - 0.08
        width = 0.08  # Thin shadow width
        height = bar.get_height()
        ax.add_patch(plt.Rectangle((x, y), width, height, color=shadow_color, alpha=1.0, zorder=1))

    # Add error bars manually using Matplotlib
    # for i, (y, err) in enumerate(zip(data['Average # SPIX'], data['std_dev'])):
    #     plt.errorbar(i, y, yerr=err, fmt='none', ecolor='black', capsize=5, capthick=1.5, linewidth=1.5, zorder=3)

    # Labels with LaTeX formatting
    ymax = ax.get_ylim()[1]
    ax.set_title("Controlling the Number of Superpixels",fontsize=16)
    ax.set_ylim([0,ymax])
    plt.xlabel(r'Split Step Hyperparamter ($\gamma$)', fontsize=16)
    plt.ylabel('Average # SPIX', fontsize=16)

    # Adjust tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # -- save --
    save_root = Path("output/plots")
    if not save_root.exists():
        save_root.mkdir(parents=True)
    fname = save_root / "controlled_nspix.png"
    plt.savefig(fname,transparent=True)
    plt.close("all")

def eval_experiments(dname,vnames,pnames=None,refresh=False):

    # -- data info --
    if dname == "davis": offset = 0
    else: offset = 1
    method = "bist"
    # method = "bass"

    # -- init --
    cache_root = Path("./output/eval_ablation/cache/")
    cache_root = Path("./output/eval_ablation/cache_v2/")
    # cache_root = Path("./output/eval_ablation/cache_v3/")
    # cache_root = Path("./output/eval_ablation/cache_v4/")
    if method == "bass":
        cache_root = cache_root / "bass"

    # -- run --
    summs_agg = []
    # vnames = ["bmx-trees"]
    # vnames = vnames[:2]
    # vnames = vnames[:4]
    # vnames = vnames[:8]
    # vnames = vnames[:3]
    # vnames = ["dogs-jump"]
    # vnames = ["dogs-jump","blackswan"]
    # vnames = ["bmx-trees"]

    if pnames is None:
        pnames = []

        # pnames = ["param10","param11","param20","param21","param22","param23",
        #           "param30","param31","param32"]
        # pnames = ["param30","param31","param32"]
        # pnames = ["param0","param20","param21","param22"]

        # pnames = ["param20"]
        # pnames = ["param0","param10","param11"]
        # pnames = ["param0",]
        # vnames = vnames[:1]
        # vnames = ["bike-packing","bmx-trees","car-roundabout"]

        # -- ablation for boundary updates --
        # pnames = ["param%d"%i for i in range(10,10+4)]

        # -- ablation for boundary updates [v2] --
        # pnames = ["param%d"%i for i in range(60,60+4)]

        # -- ablation for split --
        pnames += ["param%d"%i for i in range(40,40+18)]
        # pnames += ["param%d"%i for i in np.arange(49,57+1)]

        # -- ablation for flow --
        pnames += ["param%d"%i for i in range(80,80+3)]
        # pnames += ["param%d"%i for i in range(80,80+6)]
        # pnames = ["param100","param101"]
        # pnames = ["param0","param101"]
        # pnames = ["param101"]
        # pnames = ["param0"] + ["param%d"%i for i in range(101,101+4)]
        # pnames = ["param%d"%i for i in range(101,101+8)]
        # pnames = ["param%d"%i for i in range(101,101+16)]
        # pnames = ["param%d"%i for i in range(120,120+16)]
        # pnames = ["param%d"%i for i in range(140,140+12)]
        # pnames = ["param%d"%i for i in range(140,140+4)]
        # pnames = ["param101"]
        # pnames = ["param0"]
        # pnames = ["param1"]
        # pnames = ["param0"]

        # -- bist explore davis params --
        # pnames = ["param%d"%i for i in range(100,100+3)]

        # -- BIST [relabel grid] --
        # pnames += ["param%d"%i for i in range(200,200+16)]

        # -- BIST [potts/sigma2] --
        pnames += ["param%d"%i for i in range(300,300+8)]

        # -- BIST [split] --
        pnames += ["param%d"%i for i in range(400,400+3)]

        # -- bass explore davis params --
        # pnames += ["param%d"%i for i in range(100,100+6)]

        # -- ... --
        pnames += ["param0"]
        # pnames += ["param0","param1"]
        pnames = ["param0","param1"]
        # pnames = ["param1","param2"]
        # pnames = ["param0"]
        # print(len(pnames))
        # exit()
        # pnames = ["param60","param63"]
        # pnames = ["param%d"%i for i in range(600,640)]
        # pnames = ["param63"]

        # pnames = ["param60","param61","param62"]
        # pnames = ["param61","param62"]

        # pnames = ["param0",]+["param%d"%i for i in range(200,200+20)]
        # pnames = ["param%d"%i for i in range(400,400+3*10)]
        # pnames = ["param%d"%i for i in range(400,400+3*3)]
        # pnames = ["param0",]+["param%d"%i for i in range(200,200+12)]

    for pname in tqdm.tqdm(pnames,position=0,leave=False):

        # -- read and append info --
        fn = "result/%s/%s/info/%s.csv"%(dname,method,pname)
        # print(fn)
        # exit()
        info = pd.read_csv(fn).to_dict(orient='records')[0]

        # -- reading cache --
        summs = read_cache(cache_root,dname,pname)
        if not(summs is None) and (refresh is False):
            for k,v in info.items(): summs[k] = v
            summs = summs[summs['name'].isin(vnames)]
            summs_agg.append(summs)
            continue
        else:
            summs = []

        for vname in tqdm.tqdm(vnames,position=1,leave=False):

            # -- read --
            vid = read_video(dname,vname)
            seg = read_seg(dname,vname)
            nframes = len(vid)
            root = Path("result/%s/%s/%s/%s"%(dname,method,pname,vname))
            # print(root)
            spix = read_spix(root,vname,offset)

            # -- eval --
            _summ = computeSummary(vid,seg,spix)
            _summ.name = vname
            _summ.method = method
            _summ.nspix = len(np.unique(spix))
            _summ.param = pname
            summs.append(_summ)

        # -- caching --
        save_cache(summs,cache_root,dname,pname)

        # -- agg --
        for k,v in info.items():
            for summ in summs: summ[k] = v
        summs_agg.append(pd.DataFrame(summs))

    # print(summs_agg)
    return pd.concat(summs_agg)

def main():

    dname = "davis"
    vnames = get_video_names(dname)

    # dname = "segtrackerv2"
    # df = eval_experiments(dname,vnames,refresh=True)
    # df = eval_experiments(dname,vnames,pnames=None,refresh=False)
    # df = None

    # -- plots --
    # controlled_nspix()
    controlled_tex()
    return

    # print(df)

    # # df = df[df['param'].isin(['param20','param23'])]
    # df = df[df['param'].isin(['param0','param10','param11'])]
    # print(df)
    # print(df[['param','name','tex','pooling']])
    # exit()

    # df.drop(["name","method","flow_name"],axis=1,inplace=True)
    df.drop(["name","method"],axis=1,inplace=True)
    # df.drop(["name","method","param"],axis=1,inplace=True)
    df = df.groupby(["param"]).mean().reset_index()
    # df_mean = df.groupby(["prop_nc","prop_icov"]).mean().reset_index()
    # df_std = df.groupby(["prop_nc","prop_icov"]).std().reset_index()/math.sqrt(10.)
    # df = df_mean
    # cols = ['param','ave_nsp','pooling','ue2d','sa2d','ue3d','sa3d','tex']
    # cols = ['param','ave_nsp','pooling','ue2d','sa2d','tex','szv']
    cols = ['param','pooling','sa2d','sa3d','ue2d','ue3d']
    # cols = ["prop_nc","prop_icov",'pooling','sa2d','sa3d','ue2d','ue3d']
    print(df[cols])
    # print(df_std[cols])

    cols = ["param","ave_nsp","ev","ev3d","tex","szv"]
    # cols = ["prop_nc","prop_icov","ave_nsp","ev","ev3d","tex","szv"]
    print(df[cols])
    # print(df_std[cols])

    # -- const --
    cols = ['param','strred0','strred1']
    # cols = ['param','alpha','split_alpha']
    print(df[cols])

    # -- params [40 - 52] --
    cols = ['param','alpha','split_alpha','iperc_coeff']
    # cols = ['param','alpha','split_alpha']
    print(df[cols])

    # -- params [10 - 13] --
    # cols = ['param','prop_nc','prop_icov','strred0','strred1']
    # print(df[cols])

    # -- params [100 - 1??] --
    # cols = ['param','alpha','split_alpha']
    # print(df[cols])
    cols = ['param','thresh_relabel','thresh_new']
    print(df[cols])




if __name__ == "__main__":
    main()
