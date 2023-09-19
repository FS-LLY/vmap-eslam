import torch
import model
import embedding
import render_rays
import numpy as np
import vis
import ESLAM
import torch.nn as nn
from tqdm import tqdm
from pynvml import *
import trimesh
import skimage
from packaging import version
import numpy as np
import open3d as o3d

class Trainer:
    def __init__(self, cfg):
        self.obj_id = cfg.obj_id
        self.cfg =cfg
        self.bound = cfg.bound
        self.device = cfg.training_device
        self.hidden_feature_size = cfg.hidden_feature_size #32 for obj  # 256 for iMAP, 128 for seperate bg
        self.obj_scale = cfg.obj_scale # 10 for bg and iMAP
        self.n_unidir_funcs = cfg.n_unidir_funcs
        self.emb_size1 = 21*(3+1)+3
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1
        self.resolution = cfg.resolution
        self.marching_cubes_bound = cfg.marching_cubes_bound
        self.level_set = cfg.level_set
        self.points_batch_size=500000

        self.load_network(cfg)

        if self.obj_id == 0:
            self.bound_extent = 0.995
        else:
            self.bound_extent = 0.9

    def load_network(self,cfg):
        #这里，开始为背景配置ESLAM特征平面：
        if cfg.do_bg and self.obj_id == 0:
            eslam = ESLAM.ESLAM(cfg)
            self.decoders = model.ESLAMdecoder(c_dim=cfg.c_dim, truncation=cfg.truncation, learnable_beta=cfg.learnable_beta,bound = cfg.bound)
            self.decoders = self.decoders.to(cfg.data_device)
            decoders_para_list = []
            decoders_para_list += list(self.decoders.parameters())
            planes_para = []
            for planes in [eslam.shared_planes_xy, eslam.shared_planes_xz, eslam.shared_planes_yz]:
                for i, plane in enumerate(planes):
                    plane = nn.Parameter(plane)
                    planes_para.append(plane)
                    planes[i] = plane

            c_planes_para = []
            for c_planes in [eslam.shared_c_planes_xy,eslam.shared_c_planes_xz, eslam.shared_c_planes_yz]:
                for i, c_plane in enumerate(c_planes):
                    c_plane = nn.Parameter(c_plane)
                    c_planes_para.append(c_plane)
                    c_planes[i] = c_plane
            self.eslam = eslam
            self.decoders_para_list = decoders_para_list
            self.planes_para = planes_para
            self.c_planes_para = c_planes_para
        else:
            self.fc_occ_map = model.OccupancyMap(
                self.emb_size1,
                self.emb_size2,
                hidden_size=self.hidden_feature_size
            )
            self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)
    #渲染3D模型
    def meshing(self, bound, obj_center, grid_dim=256):
        with torch.no_grad():
            occ_range = [-1., 1.]
            range_dist = occ_range[1] - occ_range[0]
            scene_scale_np = bound.extent / (range_dist * self.bound_extent)
            scene_scale = torch.from_numpy(scene_scale_np).float().to(self.device)
            transform_np = np.eye(4, dtype=np.float32)
            transform_np[:3, 3] = bound.center
            transform_np[:3, :3] = bound.R
            # transform_np = np.linalg.inv(transform_np)  #
            transform = torch.from_numpy(transform_np).to(self.device)
            grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                                scale=scene_scale, transform=transform).view(-1, 3)
            grid_pc -= obj_center.to(grid_pc.device)
            ret = self.eval_points(grid_pc)#内存爆了
            if ret is None:
                return None

            occ, _ = ret
            mesh = vis.marching_cubes(occ.view(grid_dim, grid_dim, grid_dim).cpu().detach().numpy())
            if mesh is None:
                print("marching cube failed")
                return None
            # Transform to [-1, 1] range
            mesh.apply_translation([-0.5, -0.5, -0.5])
            mesh.apply_scale(2)

            # Transform to scene coordinates
            mesh.apply_scale(scene_scale_np)
            mesh.apply_transform(transform_np)
            vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
            ret = self.eval_points(vertices_pts)
            if ret is None:
                return None
            _, color = ret
            mesh_color = color * 255
            vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
            mesh.visual.vertex_colors = vertex_colors
            return mesh

    def eval_points(self, points,chunk_size=100000):
        # 256^3 = 16777216
        if self.obj_id == 0 and self.cfg.do_bg:
            self.points_batch_size = 500000
            p_split = torch.split(points, self.points_batch_size)
            bound = self.bound##################################
            rets = []
            for pi in p_split:
                # mask for points out of bound
                mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
                mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
                mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
                mask = mask_x & mask_y & mask_z

                ret = self.decoders(pi, self.eslam.all_planes)

                ret[~mask, -1] = -1
                rets.append(ret)

            ret = torch.cat(rets, dim=0)
            alpha = self.sdf2alpha(ret[..., -1], self.decoders.beta)#volume densities
            
            occ = 1 - torch.exp(-alpha)
            return (occ,ret[...,:3])#occ,rgb
            #return (alpha,ret[...,:3])##?S
        else:
            alpha, color = [], []
            n_chunks = int(np.ceil(points.shape[0] / chunk_size))
            with torch.no_grad():
                for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts
                    chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size)
                    embedding_k = self.pe(points[chunk_idx, ...])#位置编码
                    alpha_k, color_k = self.fc_occ_map(embedding_k)#MLP
                    alpha.extend(alpha_k.detach().squeeze())
                    color.extend(color_k.detach().squeeze())
            alpha = torch.stack(alpha)
            color = torch.stack(color)
            #把点按sigmoid 投到[-1,1]
            occ = render_rays.occupancy_activation(alpha).detach()#这里有个detach
            if occ.max() == 0:
                print("no occ")
                return None
            return (occ, color)

    def bg_get_mesh(self, all_planes , decoders,mesh_bound, device='cuda:0', color=True):
        """
        Get mesh from keyframes and feature planes and save to file.
        Args:
            mesh_out_file (str): output mesh file.
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            keyframe_dict (dict): keyframe dictionary.
            device (str): device to run the model.
            color (bool): whether to use color.
        Returns:
            None

        """

        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            

            z = []
            mask = []
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                mask.append(mesh_bound.contains(pnts.cpu().numpy()))
            mask = np.concatenate(mask, axis=0)

            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                z.append(self.eval_points(pnts.to(device), all_planes, decoders).cpu().numpy()[:, -1])
            z = np.concatenate(z, axis=0)
            z[~mask] = -1

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print('marching_cubes error. Possibly no surface extracted from the level set.')
                return

            # convert back to world coordinates
            vertices = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if color:
                # color is extracted by passing the coordinates of mesh vertices through the network
                points = torch.from_numpy(vertices)
                z = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    z_color = self.eval_points(pnts.to(device).float(), all_planes, decoders).cpu()[..., :3]
                    z.append(z_color)
                z = torch.cat(z, dim=0)
                vertex_colors = z.numpy()
            else:
                vertex_colors = None

            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            return mesh

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound

        padding = 0.05

        nsteps_x = ((bound[0][1] - bound[0][0] + 2 * padding) / resolution).round().int().item()
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding, nsteps_x)
        
        nsteps_y = ((bound[1][1] - bound[1][0] + 2 * padding) / resolution).round().int().item()
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding, nsteps_y)
        
        nsteps_z = ((bound[2][1] - bound[2][0] + 2 * padding) / resolution).round().int().item()
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding, nsteps_z)

        x_t, y_t, z_t = torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(z).float()
        grid_x, grid_y, grid_z = torch.meshgrid(x_t, y_t, z_t, indexing='xy')
        grid_points_t = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)

        return {"grid_points": grid_points_t, "xyz": [x, y, z]}
    
    def sdf2alpha(self, sdf, beta=10):
        """

        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))
    

    











