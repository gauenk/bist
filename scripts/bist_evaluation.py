"""

  Evaluate the BIST Experiments

"""

import bist

import math
import pandas as pd
import shutil # centered table titles
from tabulate import tabulate # pretty tables
from pathlib import Path

def format_df(df, float_format="{:.4f}"):
    df_copy = df.copy()  # Avoid modifying the original DataFrame
    for col in df_copy.select_dtypes(include=['float', 'int']).columns:
        if str(col).startswith("ue"): _float_format = "{:0.6f}"
        elif str(col).startswith("pooling"): _float_format = "{:2.2f}"
        elif str(col).startswith("ssim"): _float_format = "{:0.3f}"
        elif str(col).startswith("ave_nsp"): _float_format = "{:0.1f}"
        else: _float_format = float_format
        df_copy[col] = df_copy[col].map(lambda x: _float_format.format(x))
    return df_copy

def view_table(df):
    print(tabulate(format_df(df), headers='keys', tablefmt='pretty'))

def bist_bass_exps(results):

    # -- prepare results --
    N = 1.*len(pd.unique(results['vname']))
    results = results[results['group'].isin(["bist","bass","tsp"])]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep","flow"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = (results.groupby(["group","id"]).std()/math.sqrt(N)).reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Default Parameters.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id', 'group', 'ave_nsp', 'tex', 'szv']])
    view_table(results_m[['id','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])
    print("Standard Error:")
    view_table(results_s[['id', 'group', 'ave_nsp', 'tex', 'szv']])
    view_table(results_s[['ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])


def split_step_exps(results):

    # -- prepare results --
    N = 1.*len(pd.unique(results['vname']))*len(pd.unique(results['rep']))
    print("N: ",N)
    # N = 1.*len(pd.unique(results['vname']))
    # results = results[results['group'] == "split"]
    split_groups = ['split_g0','split_g1','split_g2']
    bass_res = results[results['group'].isin(["bass"])]
    bass_res['tgt_nsp'] = bass_res['ave_nsp']
    bass_res = bass_res[['vname','tgt_nsp']]
    # results = results[results['group'].isin(split_groups)]
    results = results[results['group'].str.startswith("rsplit")]
    results = results.rename(columns={"gamma":"gamma",
                                      "split_alpha":"alpha_s"})
    if not('alpha_s' in results.columns): results['alpha_s'] = -1
    nrep = len(results['rep'].unique())
    results = results.merge(bass_res,on='vname',how='left')
    results['error'] = (results['ave_nsp'] - results['tgt_nsp']).abs()

    results['error'] = results['error'].astype(int)
    results['ave_nsp'] = results['ave_nsp'].astype(int)
    results['tex'] = (100*results['tex']).round(decimals=0).astype(int)
    results['pooling'] = results['pooling'].round(decimals=2)
    results['ue2d'] = (1000*results['ue2d']).round(decimals=2)
    results['ue3d'] = (1000*results['ue3d']).round(decimals=2)

    # # -- inspect [dev] --
    # for vname,vdf in results.groupby("vname"):
    #     df_a = vdf[vdf['group'].isin(["split_g2"])].sort_values("alpha")
    #     df_b = vdf[vdf['gamma'] == 4]
    #     df_c = vdf[vdf['gamma'] == 8]
    #     print("-"*10)
    #     print(vname)
    #     print(df_a[['gamma','alpha','error','tex','sa2d','sa3d','ue2d','ue3d']])
    #     print(df_b[['gamma','alpha','error','tex','sa2d','sa3d','ue2d','ue3d']])
    #     print(df_c[['gamma','alpha','error','tex','sa2d','sa3d','ue2d','ue3d']])
    #     # print(bass_v[['alpha','ave_nsp','tex','pooling']])


    results = results.drop(columns=["vname","rep","flow"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = (results.groupby(["group","id"]).std()/math.sqrt(N)).reset_index()
    results_s['gamma'] = results_m['gamma']
    results_s['alpha'] = results_m['alpha']
    results_s['alpha_s'] = results_m['alpha_s']
    results_m = results_m.sort_values(["gamma","alpha","alpha_s"])
    results_s = results_s.sort_values(["gamma","alpha","alpha_s"])


    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Split Step.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','gamma','alpha','alpha_s','ave_nsp','std_nsp','szv']])
    view_table(results_m[['id','tex','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])
    print("Standard Error:")
    view_table(results_s[['id','gamma','alpha','alpha_s','ave_nsp','std_nsp','szv']])
    view_table(results_s[['id','tex','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])

# def splitN_step_exps(results):

#     # -- prepare results --
#     N = 1.*len(pd.unique(results['vname']))
#     results = results[results['group'].isin(["split2","split5","split10"])]
#     nrep = len(results['rep'].unique())
#     results = results.drop(columns=["vname","rep","flow"],axis=1)
#     results_m = results.groupby(["group","id"]).mean().reset_index()
#     results_s = (results.groupby(["group","id"]).std()/math.sqrt(N)).reset_index()
#     results_s['gamma'] = results_m['gamma']
#     results_s['alpha'] = results_m['alpha']
#     results_s['nimgs'] = results_m['nimgs']
#     results_m = results_m.sort_values(["gamma","alpha","nimgs"])
#     results_s = results_s.sort_values(["gamma","alpha","nimgs"])

#     # -- view results --
#     terminal_width = shutil.get_terminal_size().columns
#     print("\n"*2)
#     print("Split [10] Step.".center(terminal_width))
#     print("Averages:")
#     view_table(results_m[['id','gamma','alpha','nimgs','ave_nsp','tex','szv']])
#     view_table(results_m[['id','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])
#     print("Standard Error:")
#     view_table(results_s[['id','gamma','alpha','nimgs','ave_nsp','tex','szv']])
#     view_table(results_s[['id','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])

def relabeling_exps(results):

    # -- prepare results --
    N = 1.*len(pd.unique(results['vname']))
    results = results[results['group'] == "relabel"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep","flow"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = (results.groupby(["group","id"]).std()/math.sqrt(N)).reset_index()
    results_s['epsilon_new'] = results_m['epsilon_new']
    results_s['epsilon_reid'] = results_m['epsilon_reid']
    results_m = results_m.sort_values(["epsilon_new","epsilon_reid"])
    results_s = results_s.sort_values(["epsilon_new","epsilon_reid"])
    results_m['epsilon_reid'] = 1e5*results_m['epsilon_reid']

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Relabeling.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','epsilon_new','epsilon_reid','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])
    print("Standard Error:")
    view_table(results_s[['id','epsilon_new','epsilon_reid','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d','ue3d','sa2d','sa3d','pooling']])

    # for (group,gdf) in results_m.groupby("epsilon_reid"):
    #     gdf = gdf.sort_values("epsilon_new")
    #     print("Threshold to Relabel a Current Spix as a Previous One: ",group)
    #     view_table(gdf[['id','epsilon_new','ave_nsp','tex','szv']])
    #     view_table(gdf[['id','ue2d','ue3d','sa2d','sa3d','pooling']])

def boundary_shape_exps(results):

    # -- prepare results --
    N = 1.*len(pd.unique(results['vname']))
    results = results[results['group'] == "bshape"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep","flow"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = (results.groupby(["group","id"]).std()/math.sqrt(N)).reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Boundary Shape.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','sigma_app','potts','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])
    print("Standard Error:")
    view_table(results_s[['id','sigma_app','potts','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d','ue3d','sa2d','sa3d','pooling']])


def conditioned_boundary_updates_exps(results):

    # -- prepare results --
    N = 1.*len(pd.unique(results['vname']))*len(pd.unique(results['rep']))
    print("N: ",N)
    results = results[results['group'] == "cboundary"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep","flow"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = (results.groupby(["group","id"]).std()/math.sqrt(N)).reset_index()
    if not("prop_nc" in results.columns): return

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Conditioned Boundary Updates.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','prop_nc','prop_icov','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])
    print("Standard Error:")
    view_table(results_s[['id','prop_nc','prop_icov','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d','ue3d','sa2d','sa3d','pooling']])


def optical_flow_exps(results):

    # -- prepare results --
    N = 1.*len(pd.unique(results['vname']))
    results = results[results['group'] == "flow"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep"],axis=1)
    results_m = results.groupby(["group","id","flow"]).mean().reset_index()
    results_s = (results.groupby(["group","id","flow"]).std()/math.sqrt(N)).reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Optical Flow.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','flow','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])
    print("Standard Error:")
    view_table(results_s[['id','flow','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d','ue3d','sa2d','sa3d','pooling']])

def overlap_exps(results):

    # -- prepare results --
    N = 1.*len(pd.unique(results['vname']))*len(pd.unique(results['rep']))
    print("N: ",N)
    results = results[results['group'] == "overlap"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep","flow"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = (results.groupby(["group","id"]).std()/math.sqrt(N)).reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Overlap Terms.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','overlap','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])
    print("Standard Error:")
    view_table(results_s[['id','overlap','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d','ue3d','sa2d','sa3d','pooling']])

def read_exps(save_root):
    exps = []
    for fname in save_root.iterdir():
        if not str(fname).endswith(".csv"): continue
        exp = pd.read_csv(fname).to_dict(orient='records')[0]
        exps.append(exp)
    return exps

#
# -- Primary Evaluation Function --
#

def evaluate_exp(dname,exp,spix_root,eval_root,rep,refresh=False):

    # -- compute results --
    results = []
    vnames = bist.utils.get_video_names(dname)
    image_root = bist.utils.get_image_root(dname)
    anno_root = bist.utils.get_anno_root(dname)
    cache_root = eval_root / "cache"

    print("Evaluating experiment: ")
    print(exp)

    for vname in vnames:

        # print(vname)
        # refresh = vname == "bike-packing"

        # -- read cache --
        res = bist.utils.read_cache(cache_root,dname,vname,exp['group'],exp['id'],rep)
        if not(res is None) and not(refresh):
            results.append(res)
            continue

        # -- read video, gt-segmentation, & spix --
        vid = bist.utils.read_video(image_root/vname)/255.
        anno = bist.utils.read_anno(anno_root/vname)
        spix = bist.utils.read_spix(spix_root/vname)

        # -- [optional] read bass spix --
        bass_root = spix_root.parents[2] / "id_0/rep0" / vname
        ref_spix = None
        if bass_root.exists():
            ref_spix = bist.utils.read_spix(bass_root)

        # -- optionally subset --
        read_nimgs = ('nimgs' in exp) and (exp['nimgs']>0)
        T = exp['nimgs'] if read_nimgs else spix.shape[0]
        vid = vid[:T].contiguous()
        anno = anno[:T].contiguous()

        # -- evaluate --
        res = bist.evaluate.run(vid,anno,spix,ref_spix)
        res['vname'] = vname
        for k,v in exp.items(): res[k] = v
        results.append(pd.DataFrame([res]))

        # -- save cache --
        bist.utils.save_cache([res],cache_root,dname,vname,exp['group'],exp['id'],rep)

    results = pd.concat(results).reset_index(drop=True)
    return results



def main():

    # -- read experiment configs --
    dname = "davis"
    dname = "segtrackv2"
    # save_root = Path("results/")/dname
    save_root = Path("results_v2/")/dname
    # save_root = Path("results_v3/")/dname
    exps = read_exps(save_root / "info")
    refresh = False

    # -- [optionally] evaluate only a group (or some subset) --
    # exps = [exp for exp in exps if (exp['group'] in ['bass','bist'] )]
    # exps = [exp for exp in exps if (exp['group'] in ['tsp'] )]
    # split_groups = ['split_g0','split_g1','split_g2'] + ['bass',]
    # split_groups = ['split_g1',]# + ['bass',]
    # split_groups = ['split_g2']
    # _exps0 =[exp for exp in exps if exp['group'].startswith("rsplit")]
    # _exps1 = [exp for exp in exps if (exp['group'] in ["split_g5"])]
    # _exps2 = [exp for exp in exps if (exp['group'] in ["bass"])]
    # exps = _exps0# + _exps1# + _exps2
    # exps = _exps1
    # exps = [ exp for exp in exps if exp['group'] == "overlap" ]
    exps = [ exp for exp in exps if exp['group'] == "flow" ]
    # exps = [e for e in exps if e['overlap'] == 0]
    # exps = [e for e in exps if e['overlap'] == 1]
    print(exps)
    # exps = [exp for exp in exps if (exp['group'] in ['cboundary'] )]
    # exps = [exp for exp in exps if (exp['group'] in ['flow'] )]

    # -- evaluate over a grid --
    eval_root = save_root / "eval"
    nreps = 10
    nreps = 3
    results = []
    for exp in exps:
        for rep in range(nreps):
            spix_root = save_root / exp['group'] / exp['id'] / ("rep%d"%rep)
            if not spix_root.exists(): continue
            res = evaluate_exp(dname,exp,spix_root,eval_root,rep,refresh)
            res['rep'] = rep
            results.append(res)
    results = pd.concat(results)

    # -- view results --
    # bist_bass_exps(results)
    # split_step_exps(results)
    # relabeling_exps(results)
    # boundary_shape_exps(results)
    # overlap_exps(results)
    # conditioned_boundary_updates_exps(results)
    optical_flow_exps(results) # run for segtrackv2


if __name__ == "__main__":
    main()
