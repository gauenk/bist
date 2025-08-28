
from plyfile import PlyData
import numpy as np

def get_edges_from_faces(face_data):
    data = face_data
    u0 = np.minimum(data[:,0],data[:,1])
    v0 = np.maximum(data[:,0],data[:,1])
    u1 = np.minimum(data[:,1],data[:,2])
    v1 = np.maximum(data[:,1],data[:,2])
    u2 = np.minimum(data[:,2],data[:,0])
    v2 = np.maximum(data[:,2],data[:,0])
    u = np.r_[u0,u1,u2]
    v = np.r_[v0,v1,v2]
    gt_edges = np.c_[u,v]
    _inds = np.lexsort((gt_edges[:,0],gt_edges[:,1]))
    gt_edges = gt_edges[_inds]
    gt_edges = np.unique(gt_edges,axis=0)

    vertex_counts = np.bincount(gt_edges.flatten())
    #print(vertex_counts)
    print(np.min(vertex_counts),np.max(vertex_counts))

    # k_thresh = 2
    # print(np.mean(1.0*(vertex_counts<=2)))
    # print(np.sum(1.0*(vertex_counts<=2)))

    return gt_edges



def main():
    scene_names = ["scene0001_00","scene0002_00"]
    for scene_name in scene_names:
        gt_fn = "data/scenes/scannetv2/%s/%s_vh_clean_2.ply" % (scene_name,scene_name)
        plydata = PlyData.read(gt_fn)
        data = np.stack(np.array(plydata['face'].data)['vertex_indices'],axis=0)
        gt_edges = get_edges_from_faces(data)
        print(gt_edges.shape)
        vertex_counts = np.bincount(gt_edges.flatten())
        print("gt degree: ",np.min(vertex_counts),np.max(vertex_counts))

        fn = "output/scannetv2/%s/%s_vh_clean_2.ply" % (scene_name,scene_name)
        plydata = PlyData.read(fn)
        spix_fn = "output/scannetv2/%s/%s_spix.ply" % (scene_name,scene_name)
        spix_plydata = PlyData.read(spix_fn)
        edges = np.array(plydata['edge'].data)
        edges = np.c_[edges['vertex1'],edges['vertex2']]
        data = np.array(plydata['vertex'].data)
        x = data['x']
        print(data.dtype.names)

        gcolor = data['gcolor']
        spix = data['label']
        print(spix.min(),spix.max())
        print(gcolor)
        #exit()
        nnodes = len(x)
        #print(edges.shape[0] - nnodes)
        print(edges.shape)
        vertex_counts = np.bincount(edges.flatten())
        print(len(vertex_counts))
        print("degree: ",np.min(vertex_counts),np.max(vertex_counts))


        # -- spix --
        data = np.array(spix_plydata['vertex'].data)
        ftr = np.c_[data['red'],data['green'],data['blue']]
        pos = np.c_[data['x'],data['y'],data['z']]
        var = np.c_[data['var_x'],data['var_y'],data['var_z']]
        cov = np.c_[data['cov_xy'],data['cov_xz'],data['cov_yz']]
        print(ftr)
        print(pos)
        print(var)
        print(cov)
        print(ftr.shape,pos.shape,var.shape,cov.shape)
        print(data.dtype.names)
        exit()
        # v_min = np.min(np.where(vertex_counts == 4)[0])
        # print(v_min)
        # print(edges)
        # m_inds = np.where(np.any(edges == v_min,axis=1))[0]
        # print(m_inds)
        # print(edges[m_inds])
        # print(np.min(vertex_counts),np.max(vertex_counts))
        # print("quants: ",np.quantile(vertex_counts,[0.001,0.01,0.1,0.25,0.5,0.75,0.9,0.95]))



        labels = data['label']
        max_label = labels.max()
        uniq,cnts = np.unique(labels,return_counts=True)
        quants = np.quantile(cnts,[0.1,0.25,0.5,0.75,0.9,0.95])
        fraction = (1.0*nnodes)/len(uniq)
        print(scene_name,fraction,max_label,len(uniq),nnodes,quants)


def inspect_scannetv2():
    pass


if __name__ == "__main__":
    main()