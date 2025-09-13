
# -- .. --
# import numpy as np
# import torch
# import torch as th

# from spconv.utils import Point2VoxelGPU3d
# import cumm.tensorview as tv

# from spconv.core_cc.csrc.sparse.all import ops3d
# import spconv.algo as algo
# from spconv.core import ConvAlgo
# from spconv.cppconstants import CPU_ONLY_BUILD

# import torch 

# -- read/write data --
from plyfile import PlyData
import struct

# -- sample mesh --
import kaolin

# -- spconv --
import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from spconv.pytorch import ops
from spconv.pytorch.hash import HashTable

# -- sparse tensor --
from torch_sparse import SparseTensor

# -- csr edges --
from torch_geometric.utils import to_torch_csr_tensor

# -- basic torch/numpy --
import torch as th
import numpy as np
import torch
from torch_scatter import scatter_mean


def create_sparse_voxel_grid_gpu():
    """
    Create sparse voxel grid from point cloud using GPU acceleration
    """
    print("=== Creating Sparse Voxel Grid with spconv GPU ===")
    
    # Create sample point cloud (replace with your actual point cloud)
    np.random.seed(42)
    num_points = 50000
    points = np.random.uniform(-3, 3, size=[num_points, 3]).astype(np.float32)
    
    # Add some structure (create a sphere-like shape)
    center = np.array([0, 0, 0])
    distances = np.linalg.norm(points - center, axis=1)
    points = points[distances < 3]  # Keep points within sphere
    points = points[:10000]  # Limit for demo
    
    print(f"Input points shape: {points.shape}")
    
    # Setup voxelizer
    voxel_generator = Point2VoxelGPU3d(
        vsize_xyz=[0.2, 0.2, 0.2],  # Voxel size [x, y, z]
        coors_range_xyz=[-5, -5, -5, 5, 5, 5],  # [x_min, y_min, z_min, x_max, y_max, z_max]
        num_point_features=3,  # [x, y, z] - 3 features per point
        max_num_voxels=5000,   # Maximum number of voxels
        max_num_points_per_voxel=2  # Max points per voxel
    )
    
    # Convert to GPU tensor
    points_tv = tv.from_numpy(points).cuda()  # NOT torch.from_numpy()    
    points = torch.from_numpy(points).cuda()
    print(points_tv)
    
    # Generate voxels
    voxels, coordinates, num_points_per_voxel = voxel_generator.point_to_voxel_hash(points_tv)
    
    # Convert back to numpy for inspection
    voxels_np = voxels.cpu().numpy()
    coordinates_np = coordinates.cpu().numpy()
    num_points_np = num_points_per_voxel.cpu().numpy()
    print(voxels_np.shape)
    print(coordinates_np.shape)
    print(num_points_np.shape)
    print(voxels_np)
    
    print(f"Generated {voxels_np.shape[0]} voxels")
    print(f"Voxels shape: {voxels_np.shape}")  # (num_voxels, max_points_per_voxel, 3)
    print(f"Coordinates shape: {coordinates_np.shape}")  # (num_voxels, 3) -> [x, y, z] indices
    print(f"Coordinate ranges: X[{coordinates_np[:, 0].min()}:{coordinates_np[:, 0].max()}], Y[{coordinates_np[:, 1].min()}:{coordinates_np[:, 1].max()}], Z[{coordinates_np[:, 2].min()}:{coordinates_np[:, 2].max()}]")
    

    # -- ... --
    voxels = th.from_numpy(voxels_np).cuda()
    coordinates = th.from_numpy(coordinates_np).cuda()
    num_points_np = th.from_numpy(num_points_np).cuda()
    return voxels,coordinates,num_points_np 

def create_sparse_tensor():
    """
    Create spconv SparseConvTensor using vectorized operations (much faster!)
    """
    print("\n=== Creating spconv SparseConvTensor (Vectorized) ===")
    
    # Get voxelized data
    voxels, coordinates, num_points_per_voxel = create_sparse_voxel_grid_gpu()
    
    if voxels is None:
        return None
    
    # Method 1: Simple mean aggregation using masks (fastest)
    # Create mask for valid points: (num_voxels, max_points_per_voxel)
    max_points = voxels.shape[1]
    num_voxels = voxels.shape[0]
    
    # Create indices for each position in voxel
    point_indices = torch.arange(max_points, device=voxels.device).unsqueeze(0).expand(num_voxels, -1)
    
    # Mask: True where we have valid points
    valid_mask = point_indices < num_points_per_voxel.unsqueeze(1)  # (num_voxels, max_points_per_voxel)
    
    # Zero out invalid points and compute mean
    masked_voxels = voxels * valid_mask.unsqueeze(-1).float()  # (num_voxels, max_points, 3)
    
    # Sum valid points and divide by actual count
    point_sums = masked_voxels.sum(dim=1)  # (num_voxels, 3)
    valid_counts = num_points_per_voxel.clamp(min=1).float().unsqueeze(1)  # Avoid division by zero
    features = point_sums / valid_counts  # (num_voxels, 3)
    
    print(f"Features computed vectorized: {features.shape}")
    
    # Add batch dimension to coordinates
    batch_size = 1
    batch_coords = torch.zeros((coordinates.shape[0], 4), dtype=coordinates.dtype, device=coordinates.device)
    batch_coords[:, 0] = 0  # Batch index
    batch_coords[:, 1:] = coordinates  # [batch_idx, x, y, z]
    
    # Define spatial shape
    spatial_shape = [100, 100, 100]
    
    # Create SparseConvTensor
    sparse_tensor = spconv.SparseConvTensor(
        features=features,
        indices=batch_coords,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )
    
    print(f"SparseConvTensor created with vectorized ops!")
    print(f"Features shape: {sparse_tensor.features.shape}")
    
    return sparse_tensor

def simple_sparse_conv_example():
    """
    Simple example of sparse convolution on the voxel grid
    """
    print("\n=== Simple Sparse Convolution Example ===")
    
    # Create sparse tensor
    sparse_tensor = create_sparse_tensor()
    
    # Define a simple sparse 3D convolution
    conv3d = spconv.SparseConv3d(
        in_channels=3,     # Input features (x, y, z coordinates)
        out_channels=16,   # Output features  
        kernel_size=3,     # 3x3x3 kernel
        stride=1,
        padding=1
    ).cuda()
    
    # Apply convolution
    output = conv3d(sparse_tensor)
    
    print(f"Input features: {sparse_tensor.features.shape}")
    print(f"Output features: {output.features.shape}")
    print(f"Convolution successful!")
    
    return output

def kernel_map_example():

    # Create sparse tensor
    sparse_tensor = create_sparse_tensor()

    kernel_size = 1
    stride = 1
    padding = 0

    inds,pairs,nums = get_kernel_map(sparse_tensor, kernel_size, stride, padding)

    print("\n"*10)
    print("CSR Edges.")
    get_csr_edges(pairs)
    print("\n"*10)

def get_csr_edges(pairs):

    # Example edge list
    edge_index = torch.tensor([
        [0, 0, 1, 2],
        [1, 2, 2, 0]
    ])
    print(edge_index.shape)
    print(pairs.shape)
    edge_index = pairs[:,0].long()
    
    num_nodes = edge_index.max().item() + 1
    
    # Convert to CSR using PyG utility (returns a torch_sparse.SparseTensor)
    # csr_tensor = to_torch_csr_tensor(edge_index, size=(num_nodes, num_nodes))
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

    # # Create SparseTensor from edge_index
    # adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    
    # Access CSR components
    rowptr, col, value = adj.csr()
    
    print("Row pointers:", rowptr)
    print("Column indices:", col)
    print("Values:", value)

    return rowptr, col, value

def get_kernel_map(sparse_tensor,
                   kernel_size,
                   stride=1,
                   padding=0,
                   dilation=1,
                   subm=False,
                   transposed=False,
                   algo=None):
    """
    Generate kernel map (indice pairs) for a sparse convolution.
    Returns: (out_indices, indice_pairs, indice_pair_num)
    """

    if algo is None:
        algo = ConvAlgo.Native

    # Ensure inputs are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if isinstance(stride, int):
        stride = (stride,) * 3
    if isinstance(padding, int):
        padding = (padding,) * 3
    if isinstance(dilation, int):
        dilation = (dilation,) * 3

    indices = sparse_tensor.indices  # (N, 4) [batch, z, y, x]
    spatial_shape = sparse_tensor.spatial_shape
    batch_size = sparse_tensor.batch_size
    output_padding = (0, 0, 0)  # typically 0 for normal conv

    out_indices, indice_pairs, indice_pair_num = ops.get_indice_pairs(
        indices, batch_size, spatial_shape, algo,
        kernel_size, stride, padding, dilation,
        output_padding, subm, transposed
    )

    # print("."*10)
    # print(indice_pair_num)
    # print("."*10)

    return out_indices, indice_pairs, indice_pair_num



def check_hash():
    """Fixed-Size CUDA Hash Table:
    this hash table can't delete keys after insert, and can't resize.
    You need to pre-define a fixed-length of hash table, recommend 2x size
    of your key num.

    """

    max_size = 1000
    k_dtype = torch.int32 
    v_dtype = torch.int64

    dev = torch.device("cuda:0")
    table = HashTable(dev, k_dtype, v_dtype, max_size=max_size)

    keys = torch.tensor([5, 3, 7, 4, 6, 2, 10, 8, 8], dtype=k_dtype, device=dev)
    values = torch.tensor([1, 6, 4, 77, 23, 756, 12, 14, 12], dtype=v_dtype, device=dev)
    keys_query = torch.tensor([8, 10, 2, 6, 4, 7, 3, 5], dtype=k_dtype, device=dev)

    table.insert(keys, values)

    vq, _ = table.query(keys_query)
    print(vq)
    print("."*10)
    ks, vs, cnt = table.items()
    cnt_item = cnt.item()
    print(cnt, ks[:cnt_item], vs[:cnt_item])

    table.assign_arange_()
    ks, vs, cnt = table.items()
    cnt_item = cnt.item()
    print(cnt, ks[:cnt_item], vs[:cnt_item])

    # print("----------Insert Exist Keys----------")
    # is_empty = table.insert_exist_keys(keys, values)
    # ks, vs, cnt = table.items()
    # cnt_item = cnt.item()
    # print(cnt, ks[:cnt_item], vs[:cnt_item], is_empty.dtype)


def get_contiguous_voxel_ids(indices):
    """
    Convert voxel coordinates (N,4) into contiguous voxel IDs using a GPU hash table.
    indices: (N, 4) int32 tensor [batch, z, y, x]
    Returns:
        voxel_ids: (N,) int32 tensor with contiguous IDs [0,...,K-1]
        num_unique: number of unique voxels (K)
    """
    assert indices.is_cuda, "indices must be on GPU"
    assert indices.dtype == torch.int32, "indices must be int32"

    N = indices.shape[0]

    # Flatten 4D coordinates to 1D keys
    # Can also just view as bytes or do a manual hash if needed
    # Here we use a simple linearization: batch*Z*Y*X + z*Y*X + y*X + x
    # For simplicity, just view as a single int by treating rows as keys
    keys = indices.view(-1)  # spconv HashTable can accept int32 keys

    # Create hash table
    # max_size = 1000
    dev = torch.device("cuda:0")
    k_dtype = torch.int32 
    v_dtype = torch.int64
    table = HashTable(dev, k_dtype, v_dtype, max_size=2*N)

    
    dummy_values = torch.zeros_like(keys)
    table.insert(keys, dummy_values)  # insert all keys
    table.assign_arange_()  # assign contiguous IDs 0,...,K-1

    # Query keys to get contiguous IDs
    voxel_ids, _ = table.query(keys)

    # Number of unique voxels
    _, _, cnt = table.items()
    num_unique = cnt.item()

    return voxel_ids, num_unique

def get_faces(voxels):
    
    # Suppose voxel coordinates Nx3
    # voxels = torch.tensor(voxel_coords, dtype=torch.float32).cuda()  # shape: (N, 3)
    voxel_size = 1.0  # adjust as needed
    
    # Cube corners relative to origin
    cube_offsets = torch.tensor([
        [0,0,0],
        [1,0,0],
        [1,1,0],
        [0,1,0],
        [0,0,1],
        [1,0,1],
        [1,1,1],
        [0,1,1]
    ], dtype=torch.float32, device='cuda') * voxel_size
    
    # Expand for all voxels
    vertices = voxels[:, None, :] + cube_offsets[None, :, :]  # (N, 8, 3)
    vertices = vertices.reshape(-1, 3)  # flatten to (8*N, 3)
    
    # Cube faces (triangles, relative to 8 cube corners)
    faces_template = torch.tensor([
        [0,1,2], [0,2,3],  # bottom
        [4,5,6], [4,6,7],  # top
        [0,1,5], [0,5,4],  # front
        [2,3,7], [2,7,6],  # back
        [1,2,6], [1,6,5],  # right
        [0,3,7], [0,7,4]   # left
    ], device='cuda')
    
    # Shift faces for all voxels
    N = voxels.shape[0]
    faces = faces_template[None, :, :] + torch.arange(N, device='cuda')[:, None, None] * 8
    faces = faces.reshape(-1, 3)

    return faces

def export_obj_with_colors(vertices, faces, colors, filename="voxels.obj"):
    with open(filename, 'w') as f:
        for v, c in zip(vertices.cpu().numpy(), colors.cpu().numpy()):
            f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        for face in faces.cpu().numpy():
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def export_binary_ply(vertices, colors, edges, filename="voxels_binary.ply"):

    print(filename)

    vertices = vertices.cpu().numpy()
    colors = colors.cpu().numpy()
    edges = edges.cpu().numpy()
    # faces = faces.cpu().numpy()

    with open(filename, 'wb') as f:
        # Write header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(vertices)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n".encode())
        f.write(b"property int vertex1\nproperty int vertex2\n")
        # f.write(f"element face {len(faces)}\n".encode())
        # f.write(b"property list uchar int vertex_indices\n")
        f.write(b"end_header\n")

        # Write vertex data
        for v, c in zip(vertices, colors):
            f.write(struct.pack('<fffBBB', v[0], v[1], v[2], c[0], c[1], c[2]))

        # Write edges
        for e in edges:
            f.write(struct.pack('<ii', e[0], e[1]))

        # # Write faces (as lists: uchar=3 vertices per face, followed by 3 ints)
        # for f_ in faces:
        #     f.write(struct.pack('<Biii', 3, f_[0], f_[1], f_[2]))

    print("Binary PLY export complete!")

def get_sparse_voxels(verts,colors,res):

    
    # verts: (V, 3)
    # colors: (V, C) e.g. C=3 for RGB
    
    # -- normalize verts --
    # print(verts.shape)
    # verts = verts - verts.mean(dim=1,keepdim=True)
    verts = verts - th.min(verts,dim=0).values
    verts = verts / th.max(th.max(verts,dim=0).values)
    # print(th.max(verts,dim=0).values)
    # print(th.min(verts,dim=0).values)
    box = (res*(th.max(verts,dim=0).values*1.1)).int()

    # Convert to voxel indices
    idx = (verts * (res - 1)).long()   # (V, 3)
    
    # Flatten voxel indices into linear index
    lin_idx = idx[:, 0] * res * res + idx[:, 1] * res + idx[:, 2]
    print(colors.shape,lin_idx.shape)
    
    # Aggregate features per voxel (mean of all verts that fall into that voxel)
    feat_per_voxel = scatter_mean(colors, lin_idx, dim=0, dim_size=res**3)
    
    # Get coordinates of non-empty voxels
    unique_lin_idx, inv = torch.unique(lin_idx, return_inverse=True)
    coords = torch.stack([
        unique_lin_idx // (res * res),
        (unique_lin_idx % (res * res)) // res,
        unique_lin_idx % res
    ], dim=1)  # (M, 3)
    
    # Add batch index
    batch_idx = torch.zeros(coords.size(0), 1, dtype=torch.long, device=verts.device)
    sparse_coords = torch.cat([batch_idx, coords], dim=1).int()  # (M, 4)
    
    # Features: select the aggregated ones
    features = feat_per_voxel[unique_lin_idx]  # (M, C)
    
    # # Now build Kaolin SparseTensor
    # from kaolin.ops import sparse
    # voxels = sparse.SparseTensor(sparse_coords, features, spatial_shape=(res, res, res))
    # return voxels
    return features,sparse_coords,box

    

def get_neighbors(coords,box):

    # -- .. --
    # coords: (N,3), integers
    N = coords.shape[0]
    coords = coords.int()
    print("All >=0 ? ",th.all(coords>=0))
    scale = 10000  # make sure it's larger than max grid size per axis
    X,Y,Z = box
    keys = coords[:,1] * Y*Z + coords[:,2] * Y + coords[:,3]
    print(keys.min(),keys.max())

    # -- keep hash --
    dev = torch.device("cuda:0")
    k_dtype = torch.int32 
    v_dtype = torch.int64
    table = HashTable(dev, torch.int32, torch.int64, max_size=N*2)
    table.insert(keys.to(torch.int32), torch.arange(N, device=dev, dtype=torch.int64))

    # -- all offsets --
    all_offsets = torch.tensor([[dx, dy, dz] 
                                for dx in [-1,0,1] 
                                for dy in [-1,0,1] 
                                for dz in [-1,0,1]], device=dev, dtype=torch.int32)

    # Remove the zero offset (0,0,0)
    offsets = all_offsets[~((all_offsets == 0).all(dim=1))]  # shape (26,3)
    offsets = offsets.cuda()
    zeros = th.zeros_like(offsets[:,:1])
    offsets = th.cat([zeros,offsets],axis=-1)
    
    # compute neighbor coordinates
    coords_exp = coords[:, None, :] + offsets[None, :, :]  # (N,26,3)
    neighbor_keys = coords_exp[:,:,1] * Y*Z + coords_exp[:,:,2] * Y + coords_exp[:,:,3]
    neighbor_keys = neighbor_keys.to(torch.int32)
    
    # query table
    neighbor_idx, _ = table.query(neighbor_keys.flatten())
    neighbor_idx = neighbor_idx.view(N, offsets.shape[0])  # (N,26)
    # print(neighbor_idx[:2])
    # print(th.sum(neighbor_idx < N))

    return neighbor_idx

def get_edges(neigh):
    N,K = neigh.shape
    valid_mask = th.logical_and(neigh < N,neigh >= 0)
    edge0 = torch.arange(N, device=neigh.device).view(N, 1).expand(-1, K)
    edge0 = edge0[valid_mask]
    edge1 = neigh[valid_mask]
    # print("All edges1 >= 0? ",th.all(edge1>=0))
    return th.stack([edge0, edge1],dim=1)

def get_csr(edges):

    # 0. Split
    N = edges.shape[0]
    edge0,edge1 = edges[:,0],edges[:,1]

    # 1. Count neighbors per voxel
    row_counts = torch.bincount(edge0, minlength=N)  # # of neighbors per input voxel
    
    # 2. Compute rowptr
    rowptr = torch.zeros(N+1, dtype=torch.long, device=edge0.device)
    rowptr[1:] = torch.cumsum(row_counts, dim=0)
    
    # 3. col array: Sort edges by input index to align with rowptr
    perm = torch.argsort(edge0)
    col = edge1[perm]

    return rowptr, col


def mesh_to_voxel():

    #
    #
    # -- mesh -> voxels --
    #
    #

    # -- read --
    scene_names = ["scene0030_00","scene0002_00"]
    scene_name = scene_names[0]
    fname = "data/scenes/scannetv2/%s/%s_vh_clean_2.ply" % (scene_name,scene_name)
    plydata = PlyData.read(fname)

    # -- unpack --
    data = np.array(plydata['vertex'].data)
    pos = np.c_[data['x'],data['y'],data['z']]
    colors = np.c_[data['red'],data['green'],data['blue']]
    faces = np.stack(np.array(plydata['face'].data)['vertex_indices'],axis=0)

    # -- to torch --
    V = pos.shape[0]
    F = faces.shape[0]
    pos = th.from_numpy(pos).cuda()
    colors = th.from_numpy(colors).cuda()
    faces = th.from_numpy(faces).cuda()
    print(pos.shape,faces.shape,colors.shape)

    #
    #
    # -- encase in spconv --
    #
    #

    # # -- convert position --
    # res = 256
    # tri2vox = kaolin.ops.conversions.trianglemeshes_to_voxelgrids
    # voxels = tri2vox(vertices=pos,faces=faces,resolution=res,return_sparse=True)
    # indices = voxels.coalesce().indices().T

    # -- format for sample points --
    faces_r = faces.view(-1,1).expand(-1,3)
    colors = th.gather(colors,0,faces_r).reshape(F,3,3).round()

    # -- sample vertices on faces --
    nsamples = 2*F
    sample_mesh = kaolin.ops.mesh.sample_points
    pos,_,colors = sample_mesh(pos[None,:],faces,nsamples,
                                 face_features=colors[None,:])
    pos = pos[0]
    colors = colors[0].round().int()
    # print(pos.shape,colors.shape)
    # exit()
    # print(pos.shape,colors.shape)

    # -- sparse voxel grid --
    res = 256
    ftrs,indices,box = get_sparse_voxels(pos,colors,res)
    sparse_tensor = spconv.SparseConvTensor(
        features=ftrs,indices=indices,
        spatial_shape=[res,res,res],batch_size=1)

    # -- get edges --
    neigh = get_neighbors(indices,box)
    edges = get_edges(neigh)
    # print(edges)
    # print(th.sum(edges>=edges.shape[0]))
    print("."*10)
    rowptr, col = get_csr(edges)
    print(edges)


    # -- info --
    print(edges.shape)
    print(neigh.shape,rowptr.shape,col.shape)
    print(rowptr.shape)
    print(col.shape)
    print(col[rowptr[0]:rowptr[1]])
    print(col[rowptr[1]:rowptr[2]])

    #
    # -- save for viz --
    # 

    # faces = get_faces(indices)
    export_binary_ply(pos,colors,edges,scene_name+"_voxel.ply")

    #
    # -- run clustering --
    # 

    # kwargs = {"sp_size":15}
    # labels = bist.cluster_sparse_voxel(pos,colors,col,rowptr,**kwargs)


    #
    # -- save for viz --
    #

    # ...


    #
    # -- extract uniq tiles --
    #
    

    # kernel_size, stride, padding = 3, 1, 0
    # _,pairs,_ = get_kernel_map(sparse_tensor, kernel_size, stride, padding)
    # klabels = th.gather(labels,pairs[1,:],...)
    # hash = apply_label_hash(klabels)
    # new_label = compactify_hash(hash)

    #
    # -- save for viz --
    #

    # ...

def apply_label_has(labels):
    hash_comp = th.tensor([100,1222,13433,1231]) # some hash
    hash_val = th.sum(hash_comp * labels,0)
    return hash_val
    


# Example usage
if __name__ == "__main__":
    # # Basic voxelization
    # voxels, coordinates, num_points = create_sparse_voxel_grid_gpu()
    
    # # # Create sparse tensor
    # # sparse_tensor = create_sparse_tensor()
    
    # # Run sparse convolution
    # output = simple_sparse_conv_example()

    # kernel_map_example()
    # check_hash()

    mesh_to_voxel()
    

    print("\n=== Success! ===")
    print("Sparse voxel grid created and processed successfully!")
