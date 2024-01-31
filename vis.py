import skimage.measure
import trimesh
import open3d as o3d
import numpy as np

def marching_cubes(sdf,scale,level=0.5):#0.5
    try:
        vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(    #marching_cubes_lewiner(    #marching_cubes(
        sdf, level=level)#gradient_direction='ascent'
        
    except (RuntimeError, ValueError):
        return None
    print(vertices)
    print(faces)
    print(vertex_normals)
    dim = sdf.shape[0]
    vertices = vertices / (dim - 1)
    mesh = trimesh.Trimesh(vertices=vertices,
                           vertex_normals=vertex_normals,
                           faces=faces)

    return mesh

def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst