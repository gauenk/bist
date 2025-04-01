"""

  Evaluate the BIST Experiments

"""

import bist

import pandas as pd
import shutil # centered table titles
from tabulate import tabulate # pretty tables
from pathlib import Path

def read_exps(save_root):
    exps = []
    for fname in save_root.iterdir():
        if not str(fname).endswith(".csv"): continue
        exp = pd.read_csv(fname).to_dict(orient='records')[0]
        exps.append(exp)
    return exps

def evaluate_exp(dname,exp,spix_root,eval_root,refresh=False):

    # -- compute results --
    results = []
    vnames = bist.utils.get_video_names(dname)
    image_root = bist.utils.get_image_root(dname)
    anno_root = bist.utils.get_anno_root(dname)
    cache_root = eval_root / "cache"

    for vname in vnames[:2]:

        # -- read cache --
        res = bist.utils.read_cache(cache_root,dname,vname,exp['group'],exp['id'])
        if not(res is None) and not(refresh):
            results.append(res)
            continue

        # -- read video, gt-segmentation, & spix --
        vid = bist.utils.read_video(image_root/vname)/255.
        anno = bist.utils.read_anno(anno_root/vname)
        spix = bist.utils.read_spix(spix_root/vname)

        # -- subset --
        T = spix.shape[0]
        vid = vid[:T].contiguous()
        anno = anno[:T].contiguous()

        # -- evaluate --
        res = bist.evaluate.run(vid,anno,spix)
        res['vname'] = vname
        for k,v in exp.items(): res[k] = v
        results.append(pd.DataFrame([res]))

        # -- save cache --
        bist.utils.save_cache([res],cache_root,dname,vname,exp['group'],exp['id'])

    results = pd.concat(results).reset_index(drop=True)
    return results


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
    results = results[results['group'].isin(["bist","bass"])]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = results.groupby(["group","id"]).std().reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Default Parameters.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id', 'group', 'ave_nsp', 'tex', 'szv']])
    view_table(results_m[['id','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])
    print("Standard Deviation:")
    view_table(results_s[['id', 'group', 'ave_nsp', 'tex', 'szv']])
    view_table(results_s[['ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])


def split_step_exps(results):

    # -- prepare results --
    results = results[results['group'] == "split"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = results.groupby(["group","id"]).std().reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Split Step.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','iperc_coeff','alpha','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])
    print("Standard Deviation:")
    view_table(results_s[['id','iperc_coeff','alpha','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d', 'ue3d', 'sa2d', 'sa3d', 'pooling']])

def relabeling_exps(results):

    # -- prepare results --
    results = results[results['group'] == "relabel"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = results.groupby(["group","id"]).std().reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("Relabeling.".center(terminal_width))
    for (group,gdf) in results.groupby("thresh_relabel"):
        print("Threshold to Relabel a Current Spix as a Previous One: ",group)
        view_table(results_m[['id','thresh_new','ave_nsp','tex','szv']])
        view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])

def boundary_shape_exps(results):

    # -- prepare results --
    results = results[results['group'] == "bshape"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = results.groupby(["group","id"]).std().reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Boundary Shape.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','sigma_app','potts','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])

def conditioned_boundary_updates_exps(results):

    # -- prepare results --
    results = results[results['group'] == "cboundary"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep"],axis=1)
    results_m = results.groupby(["group","id"]).mean().reset_index()
    results_s = results.groupby(["group","id"]).std().reset_index()
    if not("prop_nc" in results.columns): return

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Conditioned Boundary Updates.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','prop_nc','prop_icov','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])
    print("Standard Deviation:")
    view_table(results_s[['id','prop_nc','prop_icov','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d','ue3d','sa2d','sa3d','pooling']])


def optical_flow_exps(results):

    # -- prepare results --
    results = results[results['group'] == "flow"]
    nrep = len(results['rep'].unique())
    results = results.drop(columns=["vname","rep"],axis=1)
    results_m = results.groupby(["group","id","flow"]).mean().reset_index()
    results_s = results.groupby(["group","id","flow"]).std().reset_index()

    # -- view results --
    terminal_width = shutil.get_terminal_size().columns
    print("\n"*2)
    print("Optical Flow.".center(terminal_width))
    print("Averages:")
    view_table(results_m[['id','flow','ave_nsp','tex','szv']])
    view_table(results_m[['id','ue2d','ue3d','sa2d','sa3d','pooling']])
    print("Standard Deviation:")
    view_table(results_s[['id','flow','ave_nsp','tex','szv']])
    view_table(results_s[['id','ue2d','ue3d','sa2d','sa3d','pooling']])


def main():

    # -- read experiment configs --
    dname = "davis"
    # dname = "segtrackv2"
    save_root = Path("results/")/dname
    exps = read_exps(save_root / "info")
    refresh = False

    # -- [optionally] evaluate only a group (or some subset) --
    # exps = [exp for exp in exps if (exp['group'] in ['bass','bist'] )]
    # exps = [exp for exp in exps if (exp['group'] in ['cboundary'] )]
    # exps = [exp for exp in exps if (exp['group'] in ['flow'] )]

    # -- evaluate over a grid --
    eval_root = save_root / "eval"
    nreps = 3
    results = []
    for exp in exps:
        for rep in range(nreps):
            spix_root = save_root / exp['group'] / exp['id'] / ("rep%d"%rep)
            res = evaluate_exp(dname,exp,spix_root,eval_root,refresh)
            res['rep'] = rep
            results.append(res)
    results = pd.concat(results)

    # -- view results --
    bist_bass_exps(results)
    split_step_exps(results)
    relabeling_exps(results)
    boundary_shape_exps(results)
    conditioned_boundary_updates_exps(results)
    optical_flow_exps(results)


if __name__ == "__main__":
    main()
