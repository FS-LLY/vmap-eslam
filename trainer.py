import torch
import model
import embedding
import render_rays
import numpy as np
import vis
import ESLAM
import torch.nn as nn
from tqdm import tqdm

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
        ret = self.eval_points(grid_pc)
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

    def eval_points(self, points, chunk_size=100000):
        # 256^3 = 16777216
        if self.obj_id == 0 and self.cfg.do_bg:
            self.points_batch_size = 500000
            p_split = torch.split(points, self.points_batch_size)
            bound = self.bound
            rets = []
            for pi in p_split:
                # mask for points out of bound
                mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
                mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
                mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
                mask = mask_x & mask_y & mask_z

                ret = self.decoders(pi, all_planes=self.eslam.all_planes)

                ret[~mask, -1] = -1
                rets.append(ret)

            ret = torch.cat(rets, dim=0)
            return (ret[...,3:],ret[...,:3])#sdf,rgb
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
            occ = render_rays.occupancy_activation(alpha).detach()
            if occ.max() == 0:
                print("no occ")
                return None
            return (occ, color)











