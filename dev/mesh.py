
# -- imports --
import numpy as np
import pandas as pd
from collections import OrderedDict
from easydict import EasyDict as edict


#
# -- Top Level Function --
#

def mesh(fields):
    return mesh_pydicts(fields)

def mesh_pydicts(fields):

    # -- corner case --
    if len(fields) == 0:
        return [{}]

    # -- split names and lists --
    names,lists = zip(*fields.items())

    # -- correct the order --
    names = list(names)
    lists = list(lists)
    lists.reverse()

    # -- create meshgrid --
    return create_named_meshgrid(lists,names)

def mesh_groups(fields,groups):
    # -- corner case --
    if len(groups) == 0:
        return mesh_pydicts(fields)

    # -- mesh sets of groups --
    names = OrderedDict()
    for gid,group_i in enumerate(groups):
        G = len(group_i[list(group_i.keys())[0]])
        names["%d" % gid] = [i for i in range(G)]
    groups_mesh = mesh(names)

    # -- create sets of mesh --
    _mesh = []
    for group_nums_d in groups_mesh:
        group_nums = [group_nums_d[g] for g in names.keys()]
        for gnum,group in zip(group_nums,groups):
            # -- set all fields --
            for field in group:
                elem = group[field][int(gnum)]
                elem = elem if isinstance(elem,list) else [elem]
                fields[field] = elem
        # -- append --
        _mesh += mesh(fields)
    return _mesh


def add_cfg(cfg_list,cfg2append):
    return append_configs(cfg_list,cfg2append)

def append_configs(cfg_list,cfg2append):
    for e,exp in enumerate(cfg_list):
        cfg_list[e] = edict(cfg_list[e])
        for key in cfg2append:
            if key in cfg_list[e]: continue
            cfg_list[e][key] = cfg2append[key]
    return cfg_list
        # cfg_list[e] = edict(dict(exp,**cfg2append))
    # print(cfg_list[0])


#
#
# -- Internal Logic --
#
#

def create_named_meshgrid(lists,names,use_pd=False):
    named_mesh = []
    mesh = create_meshgrid(lists,use_pd)
    for elem in mesh:
        elem = reversed(elem)
        named_elem = edict(OrderedDict(dict(zip(names,elem))))
        named_mesh.append(named_elem)
    return named_mesh

def create_meshgrid(lists,use_pd=True):
    # -- num lists --
    L = len(lists)

    # -- tokenize each list --
    codes,uniques = [],[]
    for l in lists:
        l_codes,l_uniques = factorize(l,use_pd)
        codes.append(l_codes)
        uniques.append(l_uniques)

    # -- meshgrid and flatten --
    lmesh = np.meshgrid(*codes)
    int_mesh = [grid.ravel() for grid in lmesh]

    # -- convert back to tokens --
    mesh = [[uniques[i][j] for j in int_mesh[i]] for i in range(L)]

    # -- "transpose" the axis to iter goes across original lists --
    mesh_T = []
    L,M = len(mesh),len(mesh[0])
    for m in range(M):
        mesh_m = []
        for l in range(L):
            elem = mesh[l][m]
            if isinstance(elem,np.int64):
                elem = int(elem)
            elif isinstance(elem,np.bool_):
                elem = bool(elem)
            elif isinstance(elem,np.float64):
                elem = float(elem)
            elif isinstance(elem,np.ndarray):
                elem = elem.tolist()
            mesh_m.append(elem)
        mesh_T.append(mesh_m)
    return mesh_T

def factorize(l,use_pd=True):
    if use_pd:
        codes,uniques = pd.factorize(l)
    else:
        codes = list(np.arange(len(l)))
        uniques = np.array(l,dtype=object)
        # assert set(l) == set(np.unique(l))
    return codes,uniques

def apply_mesh_filter(mesh,mfilter,ftype="keep"):
    filtered_mesh = []
    fields_str = list(mfilter.keys())[0]
    values = mfilter[fields_str]
    field1,field2 = fields_str.split("-")
    for elem in mesh:
        match_any = False
        match_none = True
        for val in values:
            eq1 = (elem[field1] == val[0])
            eq2 = (elem[field2] == val[1])
            if eq1 and eq2:
                match_any = True
                match_none = False
        if ftype == "keep":
            if match_any: filtered_mesh.append(elem)
        elif ftype == "remove":
            if match_none: filtered_mesh.append(elem)
        else: raise ValueError(f"[pyutils.mesh] Uknown ftype [{ftype}]")
    return filtered_mesh


def create_list_pairs(fields):
    pairs = []
    for f1,field1 in enumerate(fields):
        for f2,field2 in enumerate(fields):
            if f1 >= f2: continue
            pairs.append([field1,field2])
    return pairs


#
# -- read all fields with "picked"
#

def read_rm_picked(edata):
    picked = {key:edata[key] for key in edata.keys() if "pick" in key}
    for key in picked.keys():
        del edata[key]
    return picked

def append_picked(exps,picked):

    # -- order by num --
    keys = list(picked.keys())
    nums = [int(p[4:].split("_")[0]) for p in keys]
    keys = [k for _, k in sorted(zip(nums, keys))]

    # -- fill exps each time --
    for picked_key in keys:
        pcfg = picked[picked_key]
        exps = append_picked_i(exps,pcfg,picked_key)
    return exps

def append_picked_i(exps,pcfg,picked_key):
    full_exps = []
    picked_field = "_".join(picked_key.split("_")[1:])
    for e in exps:
        fval = e[picked_field]
        pmenu = pcfg[picked_field]
        if fval in pmenu:
            pindex = pmenu.index(fval)
        else:
            pindex = pmenu.index("_def_")
        pcfg_ = {key:pcfg[key][pindex] for key in pcfg.keys()}
        for key in pcfg_:
            if not(isinstance(pcfg_[key],list)):
                pcfg_[key] = [pcfg_[key]]
            else:
                pcfg_[key] = pcfg_[key]
        pexps = mesh(pcfg_)
        add_cfg(pexps,e)
        full_exps.extend(pexps)
    return full_exps

